"""
============================================================
STRATEGY 1: CROSS-SECTIONAL MOMENTUM (Statistical Model)
============================================================

HOW IT WORKS:
  - Each day, rank all 100 ETFs by a composite momentum signal
  - Go LONG top quintile (20 ETFs), SHORT bottom quintile (20 ETFs)
  - Weight positions by rank strength (stronger signal = larger position)
  - Rebalance daily, respecting 5bp transaction costs

SIGNALS USED (from preliminary analysis):
  - mom_10  : 10-day price momentum  (ICIR ~2.4)
  - rsi_14  : 14-day RSI             (ICIR ~2.9)  <- strongest
  - close_loc: close position in daily range (ICIR ~1.6)

This is a STATISTICAL model — no training required.
It exploits the cross-sectional rank structure injected into the data.
============================================================
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_rsi(close_series, n=14):
    """Compute RSI for a single symbol's close price series."""
    delta = close_series.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df):
    """
    Engineer all features for the momentum composite signal.
    Works on a full history dataframe (sorted by symbol, date).
    Returns the same dataframe with feature columns added.
    """
    g = df.groupby('symbol')

    # --- Momentum signals ---
    df['mom_5']   = g['close'].transform(lambda x: x.pct_change(5))
    df['mom_10']  = g['close'].transform(lambda x: x.pct_change(10))
    df['mom_20']  = g['close'].transform(lambda x: x.pct_change(20))

    # --- RSI ---
    df['rsi_14']  = g['close'].transform(lambda x: compute_rsi(x, 14))

    # --- Candle close location: where did price close within the day's range? ---
    rng = df['high'] - df['low']
    df['close_loc'] = np.where(rng > 0, (df['close'] - df['low']) / rng, 0.5)

    # --- Overnight gap ---
    df['gap'] = g['open'].transform(lambda x: x.shift(0)) / \
                g['close'].transform(lambda x: x.shift(1)) - 1

    # --- Volume surprise ---
    df['vol_ma20']    = g['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['vol_surprise']= df['volume'] / df['vol_ma20'].replace(0, np.nan)

    return df


def compute_composite_score(day_df):
    """
    For a single day's cross-section of ETFs, compute a composite
    momentum score by z-scoring and averaging the individual signals.

    Returns a Series indexed by symbol with composite scores.
    """
    signals = ['mom_10', 'rsi_14', 'close_loc']
    scores  = pd.DataFrame(index=day_df['symbol'])

    for sig in signals:
        vals = day_df.set_index('symbol')[sig]
        mu   = vals.mean()
        sd   = vals.std()
        if sd > 1e-8:
            scores[sig] = (vals - mu) / sd
        else:
            scores[sig] = 0.0

    # Equal-weight average of z-scores
    composite = scores.mean(axis=1)
    return composite


def compute_target_weights(composite_scores, top_frac=0.20, leverage=1.0):
    """
    Convert composite scores into target portfolio weights.

    Strategy:
      - Long  top    `top_frac` of ETFs  (equal weight within group)
      - Short bottom `top_frac` of ETFs  (equal weight within group)
      - Zero  for the middle

    leverage=1.0 means 100% long + 100% short = 200% gross exposure,
    which is common for long-short equity. Reduce if you want less risk.

    Returns a Series indexed by symbol with target weights.
    """
    n       = len(composite_scores)
    n_side  = max(1, int(np.floor(n * top_frac)))

    ranked  = composite_scores.rank(ascending=True)
    weights = pd.Series(0.0, index=composite_scores.index)

    # Long: top n_side ranks
    long_mask  = ranked >= (n - n_side + 1)
    # Short: bottom n_side ranks
    short_mask = ranked <= n_side

    weights[long_mask]  =  leverage / n_side
    weights[short_mask] = -leverage / n_side

    # Normalise so longs sum to 1 and shorts sum to -1
    long_sum  = weights[weights > 0].sum()
    short_sum = weights[weights < 0].sum()

    if long_sum  > 0: weights[weights > 0] /= long_sum
    if short_sum < 0: weights[weights < 0] /= abs(short_sum)

    return weights


# ─────────────────────────────────────────────────────────────
# CORE STRATEGY FUNCTIONS  (mirrors the R submission structure)
# ─────────────────────────────────────────────────────────────

def initialise_state(data: pd.DataFrame) -> dict:
    """
    Pre-compute everything we need from training data.

    Parameters
    ----------
    data : full training dataframe (date, symbol, open, close, low, high, volume)

    Returns
    -------
    state : dict with:
        - 'history'        : recent price/feature history (rolling window)
        - 'positions'      : current ETF positions {symbol: weight}
        - 'wealth'         : current wealth scalar
        - 'symbols'        : list of all ETF symbols
        - 'window'         : number of days of history to keep
    """
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Compute features over full training history
    data = compute_features(data)

    # Keep a rolling window of the most recent 30 days per symbol
    # (enough for all features: RSI needs 14, mom_20 needs 20)
    WINDOW = 30
    all_dates  = sorted(data['date'].unique())
    last_dates = all_dates[-WINDOW:]
    history    = data[data['date'].isin(last_dates)].copy()

    symbols    = sorted(data['symbol'].unique().tolist())

    # Start with zero positions
    positions  = {sym: 0.0 for sym in symbols}

    print(f"[initialise_state] Done. {len(symbols)} symbols, "
          f"history window: {history['date'].min().date()} → {history['date'].max().date()}")

    state = {
        'history'  : history,
        'positions': positions,
        'wealth'   : 1.0,
        'symbols'  : symbols,
        'window'   : WINDOW,
        'tx_cost'  : 0.0005,   # 5 basis points
    }
    return state


def trading_algorithm(new_data: pd.DataFrame, state: dict) -> tuple:
    """
    Called once per trading day. Decides what to buy/sell.

    Parameters
    ----------
    new_data : single-day market data for all symbols
    state    : dict returned by initialise_state (or previous call)

    Returns
    -------
    trades    : dict {symbol: dollar_amount_to_trade}
                positive = buy, negative = sell
    new_state : updated state for next call
    """
    new_data  = new_data.copy()
    new_data['date'] = pd.to_datetime(new_data['date'])

    # ── 1. Append today's data to rolling history ──────────────
    history   = pd.concat([state['history'], new_data], ignore_index=True)
    history   = history.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Trim to window (keep last `window` unique dates per symbol)
    all_dates = sorted(history['date'].unique())
    keep_dates= all_dates[-state['window']:]
    history   = history[history['date'].isin(keep_dates)].copy()

    # ── 2. Recompute features on rolling window ─────────────────
    history   = compute_features(history)

    # ── 3. Extract today's cross-section ────────────────────────
    today     = new_data['date'].iloc[0]
    today_df  = history[history['date'] == today].copy()

    # Drop symbols with missing features (insufficient history)
    signals   = ['mom_10', 'rsi_14', 'close_loc']
    today_df  = today_df.dropna(subset=signals)

    if len(today_df) < 10:
        # Not enough data yet — hold current positions, no trades
        new_state = state.copy()
        new_state['history'] = history
        return {sym: 0.0 for sym in state['symbols']}, new_state

    # ── 4. Compute composite signal & target weights ─────────────
    composite      = compute_composite_score(today_df)
    target_weights = compute_target_weights(composite, top_frac=0.20, leverage=1.0)

    # Fill zeros for symbols not in today's cross-section
    all_symbols    = state['symbols']
    target_weights = target_weights.reindex(all_symbols, fill_value=0.0)

    # ── 5. Compute trades (change in position × wealth) ──────────
    wealth         = state['wealth']
    current_pos    = pd.Series(state['positions']).reindex(all_symbols, fill_value=0.0)

    # Current weights (positions already in wealth units)
    current_weights = current_pos  # positions stored as wealth fractions

    delta_weights   = target_weights - current_weights
    trades          = (delta_weights * wealth).to_dict()

    # ── 6. Estimate transaction costs & update wealth ─────────────
    tx_cost   = state['tx_cost']
    total_cost= sum(abs(v) for v in trades.values()) * tx_cost
    new_wealth= wealth - total_cost

    # ── 7. Update positions ───────────────────────────────────────
    # After trades execute at close, positions shift to target_weights
    new_positions = target_weights.to_dict()

    # ── 8. Pack new state ─────────────────────────────────────────
    new_state = {
        'history'  : history,
        'positions': new_positions,
        'wealth'   : new_wealth,
        'symbols'  : all_symbols,
        'window'   : state['window'],
        'tx_cost'  : tx_cost,
    }

    return trades, new_state


# ─────────────────────────────────────────────────────────────
# BACKTESTER  (walk-forward simulation on training data)
# ─────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, warmup_days: int = 30) -> pd.DataFrame:
    """
    Walk-forward backtest of the momentum strategy.

    Parameters
    ----------
    df          : full training dataframe
    warmup_days : days of history fed to initialise_state before live trading

    Returns
    -------
    results : DataFrame with columns [date, wealth, log_wealth, n_trades, turnover]
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    all_dates = sorted(df['date'].unique())
    train_dates= all_dates[:warmup_days]
    test_dates = all_dates[warmup_days:]

    print(f"Warmup : {train_dates[0].date()} → {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Live   : {test_dates[0].date()} → {test_dates[-1].date()} ({len(test_dates)} days)")

    # Initialise on warmup period
    warmup_data = df[df['date'].isin(train_dates)]
    state       = initialise_state(warmup_data)

    results = []

    for i, date in enumerate(test_dates):
        new_data = df[df['date'] == date].copy()

        trades, state = trading_algorithm(new_data, state)

        # Compute daily P&L from position returns
        # P&L = sum over held positions of (weight × daily_return × wealth)
        today_returns = new_data.set_index('symbol')['close'].pct_change()  # approximate
        # More accurately: use open-to-close or previous-close-to-close
        prev_date = all_dates[all_dates.index(date) - 1] if date != all_dates[0] else date
        prev_close= df[df['date'] == prev_date].set_index('symbol')['close']
        curr_close= new_data.set_index('symbol')['close']
        day_ret   = (curr_close / prev_close - 1).fillna(0)

        pos       = pd.Series(state['positions'])
        pnl       = (pos * day_ret).sum() * state['wealth']
        state['wealth'] = max(0.0, state['wealth'] + pnl)

        turnover  = sum(abs(v) for v in trades.values())
        n_nonzero = sum(1 for v in trades.values() if abs(v) > 1e-8)

        results.append({
            'date'       : date,
            'wealth'     : state['wealth'],
            'log_wealth' : np.log(max(state['wealth'], 1e-10)),
            'n_trades'   : n_nonzero,
            'turnover'   : turnover,
        })

        if (i + 1) % 100 == 0:
            print(f"  Day {i+1:4d}/{len(test_dates)} | "
                  f"Wealth: {state['wealth']:.4f} | "
                  f"Log-W: {np.log(max(state['wealth'],1e-10)):.4f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("Loading data...")
    df = pd.read_csv('df_train.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    print("\nRunning walk-forward backtest...")
    results = run_backtest(df, warmup_days=30)

    print("\n" + "="*50)
    print("BACKTEST RESULTS — MOMENTUM STRATEGY")
    print("="*50)
    final_wealth = results['wealth'].iloc[-1]
    final_logw   = results['log_wealth'].iloc[-1]
    max_dd_wealth= (results['wealth'] / results['wealth'].cummax() - 1).min()
    avg_turnover = results['turnover'].mean()

    print(f"  Final wealth       : {final_wealth:.4f}")
    print(f"  Final log-wealth   : {final_logw:.4f}")
    print(f"  Max drawdown       : {max_dd_wealth:.2%}")
    print(f"  Avg daily turnover : {avg_turnover:.4f}")
    print(f"  Total trading days : {len(results)}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Cross-Sectional Momentum Strategy — Backtest", fontsize=14, fontweight='bold')

    axes[0].plot(results['date'], results['wealth'], color='steelblue', linewidth=1.5)
    axes[0].axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel("Wealth")
    axes[0].set_title("Portfolio Wealth")

    axes[1].plot(results['date'], results['log_wealth'], color='darkorange', linewidth=1.5)
    axes[1].axhline(0.0, color='grey', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel("Log Wealth")
    axes[1].set_title("Log Wealth (performance metric)")

    dd = results['wealth'] / results['wealth'].cummax() - 1
    axes[2].fill_between(results['date'], dd, 0, color='red', alpha=0.4)
    axes[2].set_ylabel("Drawdown")
    axes[2].set_title("Drawdown from Peak")

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/backtest_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nChart saved to backtest_momentum.png")
