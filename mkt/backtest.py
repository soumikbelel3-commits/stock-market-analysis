"""Cross-sectional signal validation: rank IC, quantile spreads, long/short curves.

This module answers the question the screen never asked itself -- *does the score
predict anything?* -- and it can only answer half of it.

Trap #4, and why half the screen is untestable
----------------------------------------------
``yfinance`` serves **only the current statements**. There is no point-in-time
history: the 2019 balance sheet you can download today is the one filed in 2026,
restatements included, for a Nifty 50 membership list that is today's, not 2019's.
Backtesting the fundamental leg on that data scores the past with knowledge of the
future, twice over -- once through the restatements, once through survivorship.

So this module deliberately does two different things to the two legs:

* The **technical leg is backtested properly.** Every input is derived from prices
  known on the scoring date -- rolling means, expanding drawdowns, trailing returns.
  Point-in-time by construction, and ``assert_point_in_time`` proves it per run.
* The **fundamental leg is not backtested at all.** Notebook 07 instead *quantifies*
  the bias: it scores the universe on today's fundamentals and measures the
  "predictive" power over a window that closed before those statements existed. The
  IC that comes back is an artefact, and seeing its size is the point.

The consequence for the mandate is stated rather than buried: any empirically-fitted
leg weight applies to the **technical leg only**. ``config.SCORE_WEIGHTS`` stays a
stated 50/50 prior, not a fitted result.

Statistical power
-----------------
``config.BACKTEST_UNIVERSE`` defaults to the Nifty 50 for consistency with the rest
of the chain. Fifty names means quintiles of ten and a rank-IC standard error near
1/sqrt(49) = 0.14, so a single date's IC of 0.03 is indistinguishable from zero.
Every conclusion here therefore rests on the **t-statistic of the IC series across
time**, never on one date's value.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import align, config, fetch, indicators as ind, score

TRADING_DAYS = ind.TRADING_DAYS


# ==========================================================================
# Trap #4 guard
# ==========================================================================
def assert_point_in_time(panel: pd.DataFrame, as_of, label: str = "panel",
                         tolerance_days: int = 0) -> pd.DataFrame:
    """Regression guard for trap #4. Raises if any column knows the future.

    A scoring panel is point-in-time only if no column carries an observation
    dated after the date being scored. This is cheap to check and catches the
    whole family of look-ahead bugs -- a mis-signed ``shift``, a forward-filled
    forward return, a frame accidentally sliced after the fact.

    Returns a per-column report so a passing run still shows its work.
    """
    cutoff = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=tolerance_days)
    rows, bad = [], []
    for col in panel.columns:
        last = panel[col].last_valid_index()
        ahead = 0 if last is None else int(
            (panel.loc[panel.index > cutoff, col].notna()).sum())
        rows.append({"series": str(col), "last_obs": None if last is None else last.date(),
                     "obs_after_cutoff": ahead})
        if ahead:
            bad.append((str(col), ahead, None if last is None else last.date()))

    if bad:
        raise AssertionError(
            f"{label}: {len(bad)} column(s) carry observations after {cutoff.date()} "
            f"-- look-ahead (trap #4) has reappeared. First offenders: {bad[:5]}")

    out = pd.DataFrame(rows)
    out.attrs["cutoff"] = cutoff
    out.attrs["label"] = label
    return out


# ==========================================================================
# Panel construction
# ==========================================================================
def build_panel(tickers: Iterable[str],
                period: str = config.BACKTEST_PERIOD,
                field: str = "Close",
                calendar: str = "NSE",
                max_nan_pct: float = 8.0) -> tuple[pd.DataFrame, dict, list]:
    """One aligned price panel plus the raw frames the OHLCV factors need.

    Returns ``(panel, frames, failures)``. The panel goes through
    ``align.align_to_calendar`` and ``assert_alignment`` like every other
    cross-market frame in the project -- a backtest run on a contaminated
    calendar (trap #1) produces confident nonsense.
    """
    frames, failures = fetch.price_histories(tickers, period=period)
    raw = fetch.close_frame(frames, field=field)
    if raw.empty:
        raise fetch.FetchError("build_panel: no usable price history")
    panel = align.align_to_calendar(raw, calendar=calendar)
    align.assert_alignment(panel, max_nan_pct=max_nan_pct, label="backtest panel")
    panel.attrs["as_of"] = panel.index.max()
    return panel, frames, failures


def _field_panel(frames: dict[str, pd.DataFrame], field: str,
                 index: pd.DatetimeIndex) -> pd.DataFrame:
    """One OHLCV field across tickers, reindexed onto the panel's calendar."""
    cols = {t: d[field] for t, d in frames.items()
            if isinstance(d, pd.DataFrame) and field in d.columns}
    if not cols:
        return pd.DataFrame(index=index)
    out = pd.DataFrame(cols).sort_index()
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].reindex(index)


def rebalance_dates(index: pd.DatetimeIndex,
                    freq: str = config.REBALANCE_FREQ,
                    warmup: int = 252) -> pd.DatetimeIndex:
    """Last available session of each period, after a warm-up.

    The warm-up exists because the 200-day mean and the 252-day range position
    are undefined before then; scoring on a half-formed factor is not a signal,
    it is noise with a date attached.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) <= warmup:
        raise ValueError(f"panel has {len(idx)} sessions, needs more than {warmup}")
    usable = idx[warmup:]
    marks = pd.Series(usable, index=usable).groupby(
        pd.Series(usable, index=usable).dt.to_period(
            {"ME": "M", "M": "M", "QE": "Q", "Q": "Q", "W": "W"}.get(freq, "M"))
    ).max()
    return pd.DatetimeIndex(marks.values)


# ==========================================================================
# Point-in-time technical factors
# ==========================================================================
# The live scorecard computes these one name at a time from the latest bar
# (`indicators.technical_snapshot`). A backtest needs the same numbers on every
# past date, so they are recomputed here panel-wide and vectorised. Two
# deliberate differences, both stated rather than hidden:
#
#   * trailing returns use trading-day windows (63/126/252) where the live
#     snapshot uses calendar-day offsets (91/182/365). On an NSE calendar these
#     are the same horizon to within a day or two.
#   * every window is backward-looking, and drawdown uses an *expanding* peak,
#     so no factor can see past its own date.
def factor_history(panel: pd.DataFrame,
                   frames: dict[str, pd.DataFrame],
                   bench: pd.Series,
                   dma_windows: Sequence[int] = config.DMA_WINDOWS) -> dict[str, pd.DataFrame]:
    """Every ``score.TECHNICAL_SPEC`` factor, as a date x ticker frame."""
    close = panel
    high = _field_panel(frames, "High", panel.index)
    low = _field_panel(frames, "Low", panel.index)
    vol = _field_panel(frames, "Volume", panel.index)
    bench = bench.reindex(panel.index).ffill(limit=3)

    out: dict[str, pd.DataFrame] = {}
    smas = {}
    for w in dma_windows:
        smas[w] = close.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"px_vs_DMA{w}_%"] = (close / smas[w] - 1) * 100
    out["n_dma_above"] = sum((close > smas[w]).astype(float) for w in dma_windows)

    for days, name in ((63, "ret_3m_%"), (126, "ret_6m_%"), (252, "ret_12m_%")):
        out[name] = close.pct_change(days) * 100

    bench_ret = {n: bench.pct_change(n) for n in (63, 126)}
    for n in (63, 126):
        out[f"rs_{n}d_pp"] = (close.pct_change(n).sub(bench_ret[n], axis=0)) * 100

    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    span = (roll_max - roll_min).replace(0, np.nan)
    out["pos_52w_%"] = (close - roll_min) / span * 100
    # Expanding peak: only past prices, which is what a trader actually knows.
    out["dd_from_high_%"] = (close / close.cummax() - 1) * 100

    out["RSI14"] = close.apply(ind.rsi)
    out["vol_ann_%"] = close.apply(lambda s: ind.realized_vol(s, 63))

    if not high.empty and not low.empty:
        atr_cols = {}
        for t in close.columns:
            if t not in high.columns or t not in low.columns:
                continue
            ohlc = pd.DataFrame({"High": high[t], "Low": low[t], "Close": close[t]})
            atr_cols[t] = ind.atr_pct(ohlc)
        out["ATR14_%"] = pd.DataFrame(atr_cols).reindex(
            index=close.index, columns=close.columns)

    if not vol.empty:
        short = vol.rolling(20, min_periods=10).mean()
        long = vol.rolling(63, min_periods=30).mean()
        out["vol_trend_%"] = (short / long.replace(0, np.nan) - 1) * 100

    for k, v in out.items():
        v.attrs["factor"] = k
    return out


def cross_sectional_z(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Robust (median/MAD) z-score **within each date**, across names.

    Row-wise, not column-wise: the question at every rebalance is which name
    looks best relative to its peers *that day*, which is what a rank IC then
    measures. A time-series z would answer a different question.
    """
    return df.apply(lambda row: score.robust_z(row, clip=clip), axis=1)


def score_history(factors: dict[str, pd.DataFrame],
                  dates: pd.DatetimeIndex,
                  spec: dict[str, tuple[float, int]] | None = None,
                  clip: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Composite technical score on each rebalance date.

    Mirrors ``score.score_block`` exactly -- weighted sum of signed robust
    z-scores, renormalised by the weight actually available -- so a name with a
    missing factor is scored on what it has rather than penalised to zero.

    Returns ``(scores, coverage_%)``, both date x ticker.
    """
    spec = spec or score.TECHNICAL_SPEC
    dates = pd.DatetimeIndex(dates)
    contrib, present, weights = None, None, {}
    for col, (w, direction) in spec.items():
        if col not in factors:
            continue
        raw = factors[col].reindex(dates)
        z = cross_sectional_z(raw, clip=clip) * direction
        avail = raw.notna()
        part = (z * w).where(avail, 0.0)
        contrib = part if contrib is None else contrib.add(part, fill_value=0.0)
        pw = avail.astype(float) * w
        present = pw if present is None else present.add(pw, fill_value=0.0)
        weights[col] = w

    if contrib is None:
        raise ValueError("score_history: none of the spec's factors are available")

    total_w = sum(weights.values())
    scores = contrib / present.replace(0, np.nan) * total_w
    coverage = present / total_w * 100
    scores.attrs["factors_used"] = list(weights)
    scores.attrs["n_factors_in_spec"] = len(spec)
    return scores, coverage


# ==========================================================================
# Forward returns -- the only place the future is allowed to appear
# ==========================================================================
def forward_returns(panel: pd.DataFrame,
                    horizons: Sequence[int] = config.IC_HORIZONS) -> dict[int, pd.DataFrame]:
    """Return over the *next* h sessions, dated at the start of the window.

    This is evaluation data and must never reach a score. Everything that feeds
    ``score_history`` is backward-looking; everything here is forward-looking,
    and the two are joined only inside ``rank_ic``.
    """
    return {int(h): (panel.shift(-int(h)) / panel - 1) * 100 for h in horizons}


# ==========================================================================
# Information coefficient
# ==========================================================================
def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame,
            min_names: int = 10) -> pd.Series:
    """Spearman rank correlation between score and forward return, per date.

    Rank rather than Pearson because the score is an ordinal view -- it claims
    the top name beats the bottom one, not that a score of 2.0 earns twice a
    score of 1.0.
    """
    common = scores.index.intersection(fwd.index)
    out = {}
    for d in common:
        j = pd.concat([scores.loc[d], fwd.loc[d]], axis=1,
                      keys=["s", "r"]).dropna()
        if len(j) >= min_names:
            out[d] = float(j["s"].corr(j["r"], method="spearman"))
    return pd.Series(out, name="rank_ic").sort_index()


def newey_west_se(x: pd.Series, lag: int) -> float:
    """Standard error of the mean, corrected for serial correlation.

    Needed because a 126-day forward return sampled monthly overlaps its five
    neighbours: the same price move is counted six times, the IC series is
    autocorrelated, and the naive ``std/sqrt(n)`` understates the error by
    roughly sqrt(overlap). Uncorrected, a 126d IC looks about twice as
    significant as it is.

    Bartlett kernel, the standard choice. ``lag=0`` reduces to the plain OLS
    standard error.
    """
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 3:
        return float("nan")
    dev = v - v.mean()
    gamma0 = float(dev @ dev) / n
    total = gamma0
    for j in range(1, min(int(lag), n - 1) + 1):
        gamma_j = float(dev[j:] @ dev[:-j]) / n
        total += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    # A badly-behaved sample can drive the corrected variance negative; fall
    # back to the uncorrected one rather than returning a NaN t-stat.
    if total <= 0:
        total = gamma0
    return float(np.sqrt(total / n))


def ic_summary(ic: pd.Series, periods_per_year: float = 12.0,
               overlap_lag: int = 0) -> dict:
    """Mean, dispersion and the statistic that actually matters: the t-stat.

    ``t = mean / se`` over the IC **time series**. A single date's IC carries a
    standard error near 1/sqrt(N-1) -- with 50 names that is 0.14 -- which is
    why no single date is ever quoted as evidence here.

    ``overlap_lag`` switches the standard error to Newey-West. Pass the number
    of rebalances a forward-return window overlaps; ``ic_table`` derives it.
    """
    x = pd.to_numeric(ic, errors="coerce").dropna()
    n = len(x)
    if n < 3:
        return {"n_periods": n, "mean_ic": np.nan, "std_ic": np.nan,
                "t_stat": np.nan, "ir": np.nan, "hit_rate_%": np.nan,
                "se_method": "n/a", "overlap_lag": overlap_lag}
    mean, std = float(x.mean()), float(x.std(ddof=1))
    se = (newey_west_se(x, overlap_lag) if overlap_lag > 0
          else (std / np.sqrt(n) if std > 0 else np.nan))
    return {
        "n_periods": n,
        "mean_ic": mean,
        "std_ic": std,
        "t_stat": mean / se if se and np.isfinite(se) and se > 0 else np.nan,
        "ir": (mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "hit_rate_%": float((x > 0).mean() * 100),
        "se_method": "Newey-West" if overlap_lag > 0 else "iid",
        "overlap_lag": int(overlap_lag),
    }


def overlap_lag_for(horizon_days: int, rebalance_days: float = 21.0) -> int:
    """How many rebalances a forward window of `horizon_days` overlaps."""
    return max(0, int(np.ceil(horizon_days / max(rebalance_days, 1.0))) - 1)


def ic_table(scores: pd.DataFrame, fwd_by_h: dict[int, pd.DataFrame],
             periods_per_year: float = 12.0,
             rebalance_days: float = 21.0) -> pd.DataFrame:
    """IC summary at each horizon -- this table *is* the decay curve.

    Each horizon gets the Newey-West lag its own overlap implies, so the
    longer-horizon t-stats are comparable with the 21-day one rather than
    flattered by counting the same price move several times.
    """
    rows = []
    for h, fwd in sorted(fwd_by_h.items()):
        lag = overlap_lag_for(h, rebalance_days)
        s = ic_summary(rank_ic(scores, fwd), periods_per_year, overlap_lag=lag)
        rows.append({"horizon_days": h, **s})
    return pd.DataFrame(rows).set_index("horizon_days")


def factor_ic_table(factors: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                    fwd: pd.DataFrame,
                    spec: dict[str, tuple[float, int]] | None = None) -> pd.DataFrame:
    """Per-factor IC: which of the spec's factors actually earn their weight."""
    spec = spec or score.TECHNICAL_SPEC
    rows = []
    for col, (w, direction) in spec.items():
        if col not in factors:
            continue
        z = cross_sectional_z(factors[col].reindex(pd.DatetimeIndex(dates))) * direction
        s = ic_summary(rank_ic(z, fwd))
        rows.append({"factor": col, "spec_weight": w, "direction": direction, **s})
    out = pd.DataFrame(rows).set_index("factor")
    return out.sort_values("t_stat", ascending=False)


def regime_label_history(lookback: int = 756, min_periods: int = 252,
                         threshold: float = 0.5) -> tuple[pd.Series, pd.DataFrame]:
    """Reconstruct notebook 02's Risk-On / Neutral / Risk-Off label as a series.

    Notebook 02 emits only *today's* label to the dashboard -- there is no label
    history to read -- so the series is rebuilt here from the same six inputs:
    VIX, MOVE, OVX and SKEW inverted, gold/copper inverted, and HY versus IG.

    One deliberate difference from 02: each input is z-scored on a **rolling**
    3-year window rather than the trailing 10 years, so the label on any date
    uses only data available on that date. A label built from the full-sample
    mean would be look-ahead of exactly the kind this module exists to prevent.

    Returns ``(labels, components)``.
    """
    series: dict[str, pd.Series] = {}
    for tkr, name in list(config.RISK_TICKERS.items()):
        try:
            series[name] = fetch.price_history(
                tkr, period=config.BACKTEST_PERIOD, quiet=True)["Close"]
        except Exception:                                 # noqa: BLE001
            continue
    for tkr, name in (("GC=F", "Gold"), ("HG=F", "Copper"),
                      ("HYG", "US High Yield"), ("LQD", "US Investment Grade")):
        try:
            series[name] = fetch.price_history(
                tkr, period=config.BACKTEST_PERIOD, quiet=True)["Close"]
        except Exception:                                 # noqa: BLE001
            continue
    if not series:
        return pd.Series(dtype=object), pd.DataFrame()

    px = align.align_to_calendar(pd.DataFrame(series), calendar="B")

    def roll_z(s: pd.Series) -> pd.Series:
        mu = s.rolling(lookback, min_periods=min_periods).mean()
        sd = s.rolling(lookback, min_periods=min_periods).std()
        return (s - mu) / sd.replace(0, np.nan)

    comp = {}
    for name in ("VIX (equity vol)", "MOVE (bond vol)", "OVX (oil vol)",
                 "SKEW (tail risk)"):
        if name in px.columns:
            comp[name.split(" ")[0]] = -roll_z(px[name])      # high vol = risk-off
    if {"Gold", "Copper"} <= set(px.columns):
        comp["gold/copper"] = -roll_z(px["Gold"] / px["Copper"])
    if {"US High Yield", "US Investment Grade"} <= set(px.columns):
        comp["HY vs IG"] = roll_z(px["US High Yield"] / px["US Investment Grade"])

    components = pd.DataFrame(comp)
    composite = components.mean(axis=1, skipna=True)
    labels = pd.Series(
        np.where(composite >= threshold, "Risk-On",
                 np.where(composite <= -threshold, "Risk-Off", "Neutral")),
        index=composite.index, name="regime").where(composite.notna())
    components["composite"] = composite
    return labels.dropna(), components


def conditional_stats(ic: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """IC split by regime label. Answers 'does it work when it matters?'"""
    lab = labels.reindex(ic.index).ffill()
    rows = []
    for name, grp in ic.groupby(lab):
        if len(grp) < 3:
            rows.append({"regime": name, "n_periods": len(grp),
                         "mean_ic": float(grp.mean()) if len(grp) else np.nan,
                         "t_stat": np.nan, "hit_rate_%": np.nan})
            continue
        s = ic_summary(grp)
        rows.append({"regime": name, "n_periods": s["n_periods"],
                     "mean_ic": s["mean_ic"], "t_stat": s["t_stat"],
                     "hit_rate_%": s["hit_rate_%"]})
    return pd.DataFrame(rows).set_index("regime")


# ==========================================================================
# Quantile portfolios
# ==========================================================================
def quantile_portfolios(scores: pd.DataFrame, fwd: pd.DataFrame,
                        n: int = config.N_QUANTILES,
                        min_names: int = 10) -> pd.DataFrame:
    """Mean forward return of each score quantile, per rebalance date.

    Q1 is the worst-scored bucket and Qn the best, so a working signal shows a
    monotone rise across the columns. ``top_minus_bottom`` is the market-neutral
    portfolio -- long the best quantile, short the worst -- which makes this the
    long/short proof of concept as well as a diagnostic.
    """
    common = scores.index.intersection(fwd.index)
    rows = {}
    for d in common:
        j = pd.concat([scores.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
        if len(j) < max(min_names, n):
            continue
        # rank -> bucket, so ties never collapse a whole quantile
        buckets = pd.qcut(j["s"].rank(method="first"), n,
                          labels=[f"Q{i + 1}" for i in range(n)])
        rows[d] = j.groupby(buckets, observed=True)["r"].mean()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T.sort_index()
    out["top_minus_bottom"] = out[f"Q{n}"] - out["Q1"]
    return out


def quantile_summary(qp: pd.DataFrame, n: int = config.N_QUANTILES) -> pd.DataFrame:
    """Mean forward return per quantile, with a monotonicity verdict."""
    cols = [f"Q{i + 1}" for i in range(n)]
    means = qp[cols].mean()
    out = means.to_frame("mean_fwd_ret_%")
    out["hit_rate_%"] = (qp[cols] > 0).mean() * 100
    out["n_periods"] = qp[cols].notna().sum()
    diffs = means.diff().dropna()
    out.attrs["monotone"] = bool((diffs > 0).all())
    out.attrs["monotone_steps"] = int((diffs > 0).sum())
    out.attrs["n_steps"] = int(len(diffs))
    out.attrs["spread_%"] = float(means.iloc[-1] - means.iloc[0])
    return out


# ==========================================================================
# Long/short portfolio and turnover
# ==========================================================================
def quantile_weights(scores: pd.DataFrame, n: int = config.N_QUANTILES,
                     min_names: int = 10) -> pd.DataFrame:
    """Equal-weight top quintile long, bottom quintile short, per date.

    Weights sum to +1 on the long side and -1 on the short side, i.e. gross 2,
    net 0. That is the same shape notebook 10 builds for real, so the turnover
    and cost numbers here transfer.
    """
    rows = {}
    for d, row in scores.iterrows():
        s = row.dropna()
        if len(s) < max(min_names, n):
            continue
        buckets = pd.qcut(s.rank(method="first"), n, labels=False)
        w = pd.Series(0.0, index=scores.columns)
        top, bot = s.index[buckets == n - 1], s.index[buckets == 0]
        if len(top) and len(bot):
            w[top] = 1.0 / len(top)
            w[bot] = -1.0 / len(bot)
        rows[d] = w
    return pd.DataFrame(rows).T.sort_index()


def turnover(weights_by_date: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance: half the sum of absolute weight changes.

    Halved because selling one name to buy another is one round trip, not two.
    """
    w = weights_by_date.fillna(0.0)
    return (w.diff().abs().sum(axis=1) / 2.0).rename("turnover")


def long_short_curve(scores: pd.DataFrame, panel: pd.DataFrame,
                     n: int = config.N_QUANTILES,
                     cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    """Realised long/short equity curve, gross and net of costs.

    Positions are formed on the rebalance close and held to the next one, so the
    return earned between two rebalances uses the weights set at the *earlier*
    of them -- the shift below is the difference between a backtest and a
    fantasy.
    """
    w = quantile_weights(scores, n=n)
    if w.empty:
        return pd.DataFrame()
    px = panel.reindex(w.index).ffill()
    period_ret = (px / px.shift(1) - 1)

    held = w.shift(1)                       # last rebalance's book earns this period
    gross = (held * period_ret).sum(axis=1, min_count=1) * 100
    to = turnover(w)
    # Charge the cost against the period the trade paid for. Turnover at date t
    # establishes the book held from t to t+1, and that period's return lands on
    # row t+1 -- so the cost is shifted to sit alongside the return it bought.
    cost = to.shift(1) * (cost_bps / 100.0)
    net = gross - cost

    out = pd.DataFrame({"gross_%": gross, "cost_%": cost, "net_%": net,
                        "turnover": to}).dropna(subset=["gross_%"])
    out["equity_gross"] = (1 + out["gross_%"] / 100).cumprod()
    out["equity_net"] = (1 + out["net_%"] / 100).cumprod()
    return out


def curve_stats(curve: pd.DataFrame, periods_per_year: float = 12.0) -> dict:
    """Headline numbers for a rebalance-frequency return series."""
    if curve.empty:
        return {}
    r = curve["net_%"].dropna() / 100
    g = curve["gross_%"].dropna() / 100
    yrs = len(r) / periods_per_year
    eq = curve["equity_net"].dropna()
    return {
        "n_periods": int(len(r)),
        "years": round(yrs, 2),
        "cagr_net_%": float((eq.iloc[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 and len(eq) else np.nan,
        "mean_period_gross_%": float(g.mean() * 100),
        "mean_period_net_%": float(r.mean() * 100),
        "vol_ann_%": float(r.std(ddof=1) * np.sqrt(periods_per_year) * 100),
        "sharpe_net": float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
        if r.std(ddof=1) > 0 else np.nan,
        "hit_rate_%": float((r > 0).mean() * 100),
        "max_dd_%": float(ind.max_drawdown(eq)) if len(eq) > 1 else np.nan,
        "avg_turnover": float(curve["turnover"].mean()),
        "cost_drag_ann_%": float(curve["cost_%"].mean() * periods_per_year),
    }


# ==========================================================================
# Controls
# ==========================================================================
def shuffle_control(scores: pd.DataFrame, fwd: pd.DataFrame,
                    n_trials: int = 50, seed: int = 0) -> pd.DataFrame:
    """Random-score control. If a shuffled signal scores well, the harness is broken.

    Each trial reshuffles the scores *within each date*, which destroys the
    cross-sectional information while preserving every other property of the
    experiment -- the dates, the coverage, the forward returns. The resulting
    t-stats should straddle zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_trials):
        shuffled = scores.apply(
            lambda row: pd.Series(rng.permutation(row.values), index=row.index),
            axis=1)
        s = ic_summary(rank_ic(shuffled, fwd))
        rows.append({"trial": i, "mean_ic": s["mean_ic"], "t_stat": s["t_stat"]})
    return pd.DataFrame(rows)


# ==========================================================================
# Trap #4 demonstration
# ==========================================================================
def lookahead_demo(static_scores: pd.Series, panel: pd.DataFrame,
                   statement_date, window_days: int = 252) -> dict:
    """Quantify the look-ahead bias rather than describing it.

    Scores the universe with **today's** fundamentals, then measures how well
    that ranking "predicted" a window that closed *before* those statements were
    published. Any IC here is impossible information, so its size is a direct
    read on how much a naive fundamental backtest would have flattered itself.
    """
    end = pd.Timestamp(statement_date).normalize()
    px = panel.loc[:end]
    if len(px) < window_days + 2:
        window_days = max(21, len(px) - 2)
    start = px.index[-window_days - 1]
    past_ret = (px.iloc[-1] / px.loc[start] - 1) * 100

    j = pd.concat([static_scores, past_ret], axis=1, keys=["s", "r"]).dropna()
    ic = float(j["s"].corr(j["r"], method="spearman")) if len(j) >= 10 else np.nan
    return {
        "window_start": start.date(),
        "window_end": px.index[-1].date(),
        "window_days": int(window_days),
        "statements_published_after": str(end.date()),
        "n_names": int(len(j)),
        "spurious_ic": ic,
        "note": ("fundamentals dated on or after the window end were used to rank a "
                 "period that had already finished -- any non-zero IC here is an "
                 "artefact of look-ahead, not skill"),
    }
