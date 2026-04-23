"""
SPX Variance Risk Premium — Bull Put Spread Backtest
=====================================================
Downloads daily SPY and VIX data, computes the VRP (IV - RV), runs a
monthly bull put credit spread backtest with a VRP entry filter, and
produces four publication-quality figures plus a stats JSON file.

Data:   SPY prices + CBOE VIX via yfinance (2010–2025)
Signal: VRP = VIX - RV_21d > threshold  →  enter spread at month-end
Spread: short put @ -2%, long put @ -4%, 35% premium, 20% capital/trade
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Strategy parameters ───────────────────────────────────────────────────────
SPREAD_WIDTH   = 0.02   # short strike at -2%, long strike at -4%
PREMIUM_RATIO  = 0.35   # net credit as fraction of spread width
VRP_THRESHOLD  = 3.0    # minimum VRP (%) required to enter
CAPITAL_ALLOC  = 0.20   # portfolio fraction deployed per trade
RV_WINDOW      = 21     # rolling window for realized vol (trading days)
START          = '2010-01-01'
END            = '2025-12-31'

# Five major volatility events used for stress analysis
STRESS_EVENTS = [
    ('2011 Credit\nDowngrade', '2011-08-08', '2011-07-01', '2011-09-30'),
    ('2015 Flash\nCrash',      '2015-08-24', '2015-07-15', '2015-10-15'),
    ('2018\nVolmageddon',      '2018-02-05', '2018-01-01', '2018-03-31'),
    ('2020 COVID\nCrash',      '2020-03-16', '2020-02-01', '2020-05-31'),
    ('2022 Fed\nRate Shock',   '2022-06-16', '2022-04-01', '2022-09-30'),
]

# ── Colour palette ────────────────────────────────────────────────────────────
C_IV   = '#2C7BB6'
C_RV   = '#D7191C'
C_POS  = '#4DAC26'
C_NEG  = '#D01C8B'
C_GREY = '#AAAAAA'


# =============================================================================
# 1.  DATA
# =============================================================================

print("Downloading SPY and VIX data ...")
spy = yf.download('SPY',  start=START, end=END, auto_adjust=True, progress=False)
vix = yf.download('^VIX', start=START, end=END, auto_adjust=True, progress=False)

spy_close = spy['Close'].squeeze()
vix_close = vix['Close'].squeeze()

df = pd.DataFrame({'SPY': spy_close, 'VIX': vix_close}).dropna()
df['ret'] = np.log(df['SPY'] / df['SPY'].shift(1))
# Annualised realized vol: rolling std of log-returns * sqrt(252) * 100
df['RV']  = df['ret'].rolling(RV_WINDOW).std() * np.sqrt(252) * 100
df['IV']  = df['VIX']
df['VRP'] = df['IV'] - df['RV']
df = df.dropna()

print(f"  {len(df)} trading days  ({df.index[0].date()} to {df.index[-1].date()})")

# Summary statistics
vrp_mean   = df['VRP'].mean()
vrp_median = df['VRP'].median()
vrp_std    = df['VRP'].std()
iv_gt_rv   = (df['VRP'] > 0).mean() * 100
iv_mean    = df['IV'].mean()
rv_mean    = df['RV'].mean()
n_days     = len(df)
date_start = df.index[0].strftime('%Y-%m-%d')
date_end   = df.index[-1].strftime('%Y-%m-%d')
n_years    = (df.index[-1] - df.index[0]).days / 365.25


# =============================================================================
# 2.  BACKTEST
# =============================================================================

def spread_pnl(entry: float, exit_: float, premium: float) -> float:
    """P&L as a fraction of spread width for a bull put credit spread.

    Zones:
      exit >= short_strike            →  full premium kept
      long_strike <= exit < short     →  partial loss (intrinsic)
      exit < long_strike              →  max loss (1 - premium)
    """
    short_strike = entry * (1 - SPREAD_WIDTH)
    long_strike  = entry * (1 - 2 * SPREAD_WIDTH)

    if exit_ >= short_strike:
        return premium
    elif exit_ >= long_strike:
        intrinsic = (short_strike - exit_) / (entry * SPREAD_WIDTH)
        return premium - intrinsic
    else:
        return premium - 1.0


# Use last trading day of each month as entry/exit dates
month_ends = df.resample('BME').last().index
month_ends = [d for d in month_ends if d in df.index]

trades, ao_trades   = [], []
portfolio           = 1.0
always_on_pf        = 1.0
portfolio_ts        = []
always_on_ts        = []

for i, entry_date in enumerate(month_ends[:-1]):
    exit_date = month_ends[i + 1]
    if exit_date not in df.index:
        continue

    entry_spy = float(df.loc[entry_date, 'SPY'])
    exit_spy  = float(df.loc[exit_date,  'SPY'])
    vrp_entry = float(df.loc[entry_date, 'VRP'])

    pnl = spread_pnl(entry_spy, exit_spy, PREMIUM_RATIO)

    # Always-on: trade every month regardless of VRP
    always_on_pf *= (1 + pnl * CAPITAL_ALLOC)
    always_on_ts.append({'date': exit_date, 'portfolio': always_on_pf})
    ao_trades.append({'pnl_pct': pnl * 100, 'win': pnl > 0})

    # VRP-filtered: only enter when VRP exceeds threshold
    if vrp_entry >= VRP_THRESHOLD:
        portfolio *= (1 + pnl * CAPITAL_ALLOC)
        trades.append({
            'entry_date': entry_date,
            'exit_date':  exit_date,
            'entry_spy':  entry_spy,
            'exit_spy':   exit_spy,
            'vrp_entry':  vrp_entry,
            'spy_ret':    (exit_spy / entry_spy) - 1,
            'pnl_pct':    pnl * 100,
            'win':        pnl > 0,
        })
    portfolio_ts.append({'date': exit_date, 'portfolio': portfolio})

trades_df    = pd.DataFrame(trades)
ao_trades_df = pd.DataFrame(ao_trades)
pf_ts        = pd.DataFrame(portfolio_ts).set_index('date')
ao_ts        = pd.DataFrame(always_on_ts).set_index('date')

# Performance metrics
n_trades     = len(trades_df)
win_rate     = trades_df['win'].mean() * 100
avg_pnl      = trades_df['pnl_pct'].mean()
avg_pnl_wins = trades_df.loc[trades_df['win'],  'pnl_pct'].mean()
avg_pnl_loss = trades_df.loc[~trades_df['win'], 'pnl_pct'].mean()
cum_ret_filt = (portfolio - 1) * 100
cum_ret_ao   = (always_on_pf - 1) * 100
ann_ret_filt = ((1 + cum_ret_filt / 100) ** (1 / n_years) - 1) * 100
ann_ret_ao   = ((1 + cum_ret_ao   / 100) ** (1 / n_years) - 1) * 100
ao_win_rate  = ao_trades_df['win'].mean() * 100

# Drawdown series
pf_ts['peak'] = pf_ts['portfolio'].cummax()
pf_ts['dd']   = (pf_ts['portfolio'] - pf_ts['peak']) / pf_ts['peak'] * 100
max_dd        = pf_ts['dd'].min()

ao_ts['peak'] = ao_ts['portfolio'].cummax()
ao_ts['dd']   = (ao_ts['portfolio'] - ao_ts['peak']) / ao_ts['peak'] * 100
ao_max_dd     = ao_ts['dd'].min()

# Annualised Sharpe using trade-month returns only
trade_rets    = trades_df['pnl_pct'] / 100 * CAPITAL_ALLOC
sharpe_annual = (trade_rets.mean() / trade_rets.std()) * np.sqrt(12)

print(f"\nBacktest  |  n={n_trades}  win={win_rate:.1f}%  avg_pnl={avg_pnl:.2f}%")
print(f"  Filtered:  {ann_ret_filt:.1f}% ann.  maxDD={max_dd:.2f}%  Sharpe={sharpe_annual:.2f}")
print(f"  Always-on: {ann_ret_ao:.1f}% ann.  maxDD={ao_max_dd:.2f}%")


# =============================================================================
# 3.  FIGURES
# =============================================================================

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})


# ── Figure 1: IV / RV time series + VRP bar ──────────────────────────────────
print("\nGenerating figures ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['IV'], color=C_IV, lw=1.2, label='IV (VIX)', zorder=3)
ax1.plot(df.index, df['RV'], color=C_RV, lw=1.2, label='21-day RV', zorder=3)
ax1.fill_between(df.index, df['IV'], df['RV'],
                 where=(df['IV'] >= df['RV']), alpha=0.15, color=C_IV, label='IV > RV')
ax1.fill_between(df.index, df['IV'], df['RV'],
                 where=(df['IV'] <  df['RV']), alpha=0.20, color=C_RV, label='RV > IV')

for label, peak_date, _, _ in STRESS_EVENTS:
    try:
        ts  = pd.Timestamp(peak_date)
        idx = ts if ts in df.index else df.index[df.index.searchsorted(ts)]
        ax1.annotate(label, xy=(ts, df.loc[idx, 'IV']),
                     xytext=(0, 18), textcoords='offset points',
                     fontsize=7.5, ha='center',
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))
    except Exception:
        pass

ax1.set_ylabel('Volatility (%)')
ax1.set_title('Implied vs. Realized Volatility and the Variance Risk Premium (2010-2025)')
ax1.legend(loc='upper right', fontsize=8)

colors_bar = [C_IV if v > 0 else C_RV for v in df['VRP']]
ax2.bar(df.index, df['VRP'], color=colors_bar, width=2, alpha=0.7)
ax2.axhline(0, color='black', lw=0.8)
ax2.axhline(VRP_THRESHOLD, color=C_GREY, lw=1, ls='--',
            label=f'Entry threshold ({VRP_THRESHOLD}%)')
ax2.set_ylabel('VRP (%)')
ax2.set_xlabel('Date')
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig1_vrp_time.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# ── Figure 2: VRP distribution + quintile forward returns ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(df['VRP'], bins=80, color=C_IV, alpha=0.75, edgecolor='none')
ax1.axvline(vrp_mean,   color='black', lw=1.5, ls='-',  label=f'Mean ({vrp_mean:.2f}%)')
ax1.axvline(vrp_median, color=C_RV,   lw=1.5, ls='--', label=f'Median ({vrp_median:.2f}%)')
ax1.axvline(0,          color='grey', lw=1,   ls=':')
ax1.set_xlabel('VRP (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Daily VRP')
ax1.legend()

df['fwd_ret_21'] = df['SPY'].pct_change(RV_WINDOW).shift(-RV_WINDOW) * 100
df_valid = df.dropna(subset=['fwd_ret_21', 'VRP']).copy()
df_valid['quintile'] = pd.qcut(
    df_valid['VRP'], 5,
    labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)']
)
quintile_means = df_valid.groupby('quintile', observed=True)['fwd_ret_21'].mean()

bar_colors = [C_RV if v < 0 else C_IV for v in quintile_means.values]
ax2.bar(quintile_means.index.astype(str), quintile_means.values,
        color=bar_colors, alpha=0.8)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel('VRP Quintile')
ax2.set_ylabel('Avg 21-day Forward SPY Return (%)')
ax2.set_title('VRP Quintile vs. Forward SPY Returns')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig2_vrp_dist.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# ── Figure 3: Backtest ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Log scale so both strategies are legible despite ~8x return difference
ax1.plot(pf_ts.index, pf_ts['portfolio'], color=C_IV,  lw=2,   label='VRP-filtered')
ax1.plot(ao_ts.index, ao_ts['portfolio'], color=C_GREY, lw=1.5, ls='--', label='Always-on')
ax1.axhline(1, color='black', lw=0.5)
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, _: f'{(y - 1) * 100:.0f}%')
)
ax1.set_ylabel('Cumulative Return (log scale)')
ax1.set_title('Cumulative Portfolio P&L')
ax1.legend()

bar_cols = [C_POS if w else C_NEG for w in trades_df['win']]
ax2.bar(range(n_trades), trades_df['pnl_pct'], color=bar_cols, alpha=0.8)
ax2.axhline(0, color='black', lw=0.8)
ax2.legend(handles=[mpatches.Patch(color=C_POS, label='Win'),
                    mpatches.Patch(color=C_NEG, label='Loss')])
ax2.set_xlabel('Trade #')
ax2.set_ylabel('P&L (% of spread width)')
ax2.set_title('Monthly Trade P&L')

short_ret      = -SPREAD_WIDTH * 100
long_ret       = -2 * SPREAD_WIDTH * 100
scatter_colors = [C_POS if w else C_NEG for w in trades_df['win']]
ax3.scatter(trades_df['vrp_entry'], trades_df['spy_ret'] * 100,
            c=scatter_colors, alpha=0.7, s=40)
ax3.axhline(short_ret, color='orange', lw=1.5, ls='--',
            label=f'Short strike ({short_ret:.0f}%)')
ax3.axhline(long_ret,  color=C_RV,    lw=1.5, ls=':',
            label=f'Long strike ({long_ret:.0f}%)')
ax3.set_xlabel('VRP at Entry (%)')
ax3.set_ylabel('SPY Return over Trade Period (%)')
ax3.set_title('VRP at Entry vs. SPY Return')
ax3.legend(fontsize=8)

ax4.fill_between(pf_ts.index, pf_ts['dd'], 0, alpha=0.5, color=C_RV)
ax4.plot(pf_ts.index, pf_ts['dd'], color=C_RV, lw=1)
ax4.set_ylabel('Drawdown (%)')
ax4.set_xlabel('Date')
ax4.set_title('Strategy Drawdown Profile')

fig.suptitle('Backtest Results: VRP-Filtered Bull Put Spread Strategy',
             fontsize=13, y=1.01)
fig.savefig(OUTPUT_DIR / 'fig3_backtest.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# ── Figure 4: Tail risk ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

event_labels, vrp_before, vrp_during = [], [], []
for label, peak_date, window_start, window_end in STRESS_EVENTS:
    ws  = pd.Timestamp(window_start)
    we  = pd.Timestamp(window_end)
    pk  = pd.Timestamp(peak_date)
    pre = df[(df.index >= ws) & (df.index <  pk)]['VRP'].mean()
    dur = df[(df.index >= pk) & (df.index <= we)]['VRP'].mean()
    event_labels.append(label)
    vrp_before.append(pre)
    vrp_during.append(dur)

x     = np.arange(len(event_labels))
width = 0.35
ax1.bar(x - width / 2, vrp_before, width, label='Before event', color=C_IV, alpha=0.8)
ax1.bar(x + width / 2, vrp_during, width, label='During event', color=C_RV, alpha=0.8)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(event_labels, fontsize=8)
ax1.set_ylabel('Average VRP (%)')
ax1.set_title('VRP Before vs. During Major Stress Events')
ax1.legend()

thresholds   = np.arange(0, 12.5, 0.5)
win_rates_t  = []
trade_counts = []
for thr in thresholds:
    sub = trades_df[trades_df['vrp_entry'] >= thr]
    win_rates_t.append(sub['win'].mean() * 100 if len(sub) else np.nan)
    trade_counts.append(len(sub))

ax2_r = ax2.twinx()
ax2.plot(thresholds, win_rates_t, color=C_IV, lw=2, marker='o', ms=4, label='Win rate')
ax2.axvline(VRP_THRESHOLD, color='orange', lw=1.5, ls='--',
            label=f'Chosen threshold ({VRP_THRESHOLD}%)')
ax2.set_xlabel('VRP Entry Threshold (%)')
ax2.set_ylabel('Win Rate (%)', color=C_IV)
ax2.tick_params(axis='y', labelcolor=C_IV)
ax2.set_title('Win Rate vs. Entry Threshold')
ax2_r.bar(thresholds, trade_counts, width=0.4, alpha=0.25, color=C_GREY, label='Trade count')
ax2_r.set_ylabel('Number of Trades', color=C_GREY)
ax2_r.tick_params(axis='y', labelcolor=C_GREY)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig4_tail_risk.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"  Figures saved to {OUTPUT_DIR}/")


# =============================================================================
# 4.  STATS OUTPUT
# =============================================================================

vix_peaks = {}
for label, peak_date, window_start, window_end in STRESS_EVENTS:
    sub = df[(df.index >= pd.Timestamp(window_start)) &
             (df.index <= pd.Timestamp(window_end))]
    if len(sub):
        vix_peaks[label.replace('\n', ' ')] = {
            'peak_vix':  round(float(sub['IV'].max()), 1),
            'peak_date': sub['IV'].idxmax().strftime('%Y-%m-%d'),
        }

quintile_tbl = {
    str(q): {
        'vrp_mean': round(float(df_valid[df_valid['quintile'] == q]['VRP'].mean()), 2),
        'fwd_ret':  round(float(df_valid[df_valid['quintile'] == q]['fwd_ret_21'].mean()), 3),
    }
    for q in df_valid['quintile'].cat.categories
}

output = {
    'date_start':        date_start,
    'date_end':          date_end,
    'n_days':            int(n_days),
    'n_years':           round(n_years, 1),
    'vrp_mean':          round(vrp_mean,   2),
    'vrp_median':        round(vrp_median, 2),
    'vrp_std':           round(vrp_std,    2),
    'iv_gt_rv_pct':      round(iv_gt_rv,   1),
    'iv_mean':           round(iv_mean,    2),
    'rv_mean':           round(rv_mean,    2),
    'n_trades':          int(n_trades),
    'win_rate':          round(win_rate,      1),
    'avg_pnl':           round(avg_pnl,       2),
    'avg_pnl_wins':      round(avg_pnl_wins,  2),
    'avg_pnl_losses':    round(avg_pnl_loss,  2),
    'cum_ret_filtered':  round(cum_ret_filt,  2),
    'cum_ret_always_on': round(cum_ret_ao,    2),
    'ann_ret_filtered':  round(ann_ret_filt,  1),
    'ann_ret_always_on': round(ann_ret_ao,    1),
    'max_drawdown':      round(max_dd,        2),
    'ao_win_rate':       round(ao_win_rate,   1),
    'ao_max_dd':         round(ao_max_dd,     2),
    'n_ao_trades':       int(len(ao_trades_df)),
    'sharpe_annual':     round(sharpe_annual, 2),
    'vrp_threshold':     VRP_THRESHOLD,
    'spread_width_pct':  SPREAD_WIDTH * 100,
    'premium_ratio_pct': PREMIUM_RATIO * 100,
    'capital_alloc_pct': CAPITAL_ALLOC * 100,
    'rv_window':         RV_WINDOW,
    'vix_peaks':         vix_peaks,
    'quintile_fwd_ret':  quintile_tbl,
}

stats_path = BASE_DIR / 'stats.json'
with open(stats_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Stats saved to {stats_path}")
