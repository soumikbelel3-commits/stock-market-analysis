"""Performance measurement: excess-return statistics, attribution, capture.

Why this module exists
----------------------
The chain computed Sharpe ratios against **zero**. ``risk.drawdown_summary`` did
it, and so did ``backtest.curve_stats``. At an Indian risk-free rate near 6.25%
and a book at 13% volatility that overstates Sharpe by about 0.48 -- which is the
whole distance between a strategy that beats cash and one that does not. Every
statistic here is therefore an **excess-return** statistic by construction, and
the rate used is reported alongside the number rather than assumed.

The one defensible exception is a genuinely market-neutral book: it is funded on
collateral that earns the rate, so its spread is already excess. That case is
handled explicitly by ``rf=0`` with ``config.NEUTRAL_BOOK_RF_IS_ZERO`` as the
documented reason, never by silence.

What is here that was not
-------------------------
* Sortino, Calmar, Omega and the tail ratio -- because a left-skewed equity book
  is badly described by volatility alone, and an allocator will ask for all four.
* Tracking error, information ratio and active share -- the benchmark-relative
  vocabulary the chain had no way to speak.
* Up/down capture -- which separates "made money in a rising market" from
  "made money".
* Brinson attribution -- allocation versus selection, so a good year can be
  traced to the right cause instead of claimed for the wrong one.
* A drawdown table with recovery times, because peak-to-trough depth without
  time-to-recover is half the answer.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config, dashboard, indicators as ind

TRADING_DAYS = ind.TRADING_DAYS


# ==========================================================================
# The risk-free rate
# ==========================================================================
def risk_free_annual(default: float | None = None,
                     data: dict | None = None) -> tuple[float, str]:
    """Annual risk-free rate in percent, and where it came from.

    Prefers a live Indian short rate published to ``dashboard.json`` by notebook
    03; falls back to ``config.RISK_FREE_ANNUAL_PCT``, which is a stated prior
    with an as-of date rather than a fetched number -- there is no keyless daily
    source for the 91-day T-bill.

    Returning the provenance with the value is deliberate. A Sharpe ratio whose
    denominator nobody can trace is a marketing number.
    """
    fallback = config.RISK_FREE_ANNUAL_PCT if default is None else float(default)
    try:
        data = dashboard.read() if data is None else data
    except Exception:                                       # noqa: BLE001
        data = {}
    node = (data or {}).get("03_rates", {})
    for key in ("india_short_rate_%", "india_repo_%", "risk_free_%"):
        v = node.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v) and 0 < float(v) < 30:
            return float(v), f"dashboard 03_rates.{key}"
    return float(fallback), f"{config.RISK_FREE_SOURCE} (as of {config.RISK_FREE_ASOF})"


def rf_per_period(rf_annual_pct: float, periods_per_year: float) -> float:
    """Compound the annual rate down to one period. Not a division.

    ``rf_annual/12`` is wrong by the amount of the compounding, which is small at
    monthly frequency and not small over ten years of it.
    """
    if not np.isfinite(rf_annual_pct):
        return 0.0
    return float((1.0 + rf_annual_pct / 100.0) ** (1.0 / periods_per_year) - 1.0)


def excess(rets: pd.Series, rf_annual_pct: float | None = None,
           periods_per_year: float = TRADING_DAYS) -> pd.Series:
    """Return series net of the risk-free rate, period by period."""
    r = pd.to_numeric(rets, errors="coerce").dropna()
    rf_a = config.RISK_FREE_ANNUAL_PCT if rf_annual_pct is None else rf_annual_pct
    return r - rf_per_period(rf_a, periods_per_year)


# ==========================================================================
# Core statistics
# ==========================================================================
def perf_stats(rets: pd.Series,
               rf_annual_pct: float | None = None,
               periods_per_year: float = TRADING_DAYS,
               mar_annual_pct: float | None = None) -> dict:
    """The full excess-return statistics block for one return series.

    ``mar_annual_pct`` is the minimum acceptable return used by Sortino and
    Omega; it defaults to the risk-free rate, which is the convention that makes
    Sortino comparable with Sharpe.

    Every ratio in the output is computed on **excess** returns and the rate used
    is echoed back in the dict, so the number and its assumption travel together.
    """
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if len(r) < 3:
        return {"n_periods": int(len(r)), "note": "too few observations"}

    rf_a = (config.RISK_FREE_ANNUAL_PCT if rf_annual_pct is None
            else float(rf_annual_pct))
    mar_a = rf_a if mar_annual_pct is None else float(mar_annual_pct)
    rf_p = rf_per_period(rf_a, periods_per_year)
    mar_p = rf_per_period(mar_a, periods_per_year)

    ex = r - rf_p
    years = len(r) / periods_per_year
    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else np.nan

    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    down = r[r < mar_p] - mar_p
    downside_dev = (float(np.sqrt((down ** 2).mean()) * np.sqrt(periods_per_year))
                    if len(down) else 0.0)

    dd = ind.drawdown(eq)
    max_dd = float(dd.min()) / 100.0 if len(dd) else np.nan

    gains = r[r > mar_p] - mar_p
    losses = mar_p - r[r <= mar_p]
    omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else np.inf

    p95, p05 = float(r.quantile(0.95)), float(r.quantile(0.05))
    tail_ratio = abs(p95 / p05) if p05 != 0 else np.nan

    ann_excess = float((1.0 + cagr) / (1.0 + rf_a / 100.0) - 1.0) if np.isfinite(cagr) else np.nan

    return {
        "n_periods": int(len(r)),
        "years": round(years, 2),
        "total_return_%": total * 100,
        "cagr_%": cagr * 100 if np.isfinite(cagr) else np.nan,
        "excess_cagr_%": ann_excess * 100 if np.isfinite(ann_excess) else np.nan,
        "ann_vol_%": vol * 100,
        "downside_dev_%": downside_dev * 100,
        "sharpe": float(ex.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
        if r.std(ddof=1) > 0 else np.nan,
        "sortino": float((r.mean() - mar_p) / (downside_dev / np.sqrt(periods_per_year))
                         * np.sqrt(periods_per_year)) if downside_dev > 0 else np.nan,
        "calmar": float(ann_excess / abs(max_dd))
        if np.isfinite(ann_excess) and np.isfinite(max_dd) and max_dd < 0 else np.nan,
        "omega": omega,
        "max_drawdown_%": max_dd * 100 if np.isfinite(max_dd) else np.nan,
        "current_drawdown_%": float(dd.iloc[-1]) if len(dd) else np.nan,
        "hit_rate_%": float((r > 0).mean() * 100),
        "best_period_%": float(r.max() * 100),
        "worst_period_%": float(r.min() * 100),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "tail_ratio": tail_ratio,
        "gain_to_pain": float(r[r > 0].sum() / abs(r[r < 0].sum()))
        if abs(r[r < 0].sum()) > 0 else np.inf,
        "rf_annual_%": rf_a,
        "periods_per_year": periods_per_year,
    }


def sharpe_correction(rets: pd.Series, rf_annual_pct: float | None = None,
                      periods_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Side-by-side: Sharpe against zero versus Sharpe against the real rate.

    Exists so the defect this module fixes is visible rather than merely fixed.
    The gap is roughly ``rf / vol`` and it is not small.
    """
    r = pd.to_numeric(rets, errors="coerce").dropna()
    rf_a = (config.RISK_FREE_ANNUAL_PCT if rf_annual_pct is None
            else float(rf_annual_pct))
    sd = float(r.std(ddof=1))
    if sd <= 0 or len(r) < 3:
        return pd.DataFrame()
    naive = float(r.mean() / sd * np.sqrt(periods_per_year))
    rf_p = rf_per_period(rf_a, periods_per_year)
    true = float((r.mean() - rf_p) / sd * np.sqrt(periods_per_year))
    return pd.DataFrame([{
        "sharpe_vs_zero": naive,
        "sharpe_vs_rf": true,
        "overstatement": naive - true,
        "rf_annual_%": rf_a,
        "ann_vol_%": sd * np.sqrt(periods_per_year) * 100,
        "note": "sharpe_vs_zero is the number the chain reported before mkt.perf",
    }])


# ==========================================================================
# Benchmark-relative
# ==========================================================================
def relative_stats(rets: pd.Series, bench: pd.Series,
                   rf_annual_pct: float | None = None,
                   periods_per_year: float = TRADING_DAYS) -> dict:
    """Everything measured against a benchmark rather than against cash.

    ``information_ratio`` is active return over tracking error -- the statistic an
    allocator uses to decide whether the active decisions were worth the fee. It
    is *not* the Sharpe ratio of the active return, and conflating the two is a
    common and flattering error.
    """
    j = pd.concat([rets, bench], axis=1, keys=["p", "b"]).dropna()
    if len(j) < 30:
        return {"n": int(len(j)), "note": "too few overlapping observations"}

    rf_a = (config.RISK_FREE_ANNUAL_PCT if rf_annual_pct is None
            else float(rf_annual_pct))
    rf_p = rf_per_period(rf_a, periods_per_year)

    p, b = j["p"], j["b"]
    active = p - b
    var_b = float(b.var(ddof=1))
    beta = float(p.cov(b) / var_b) if var_b > 0 else np.nan
    alpha_p = float((p.mean() - rf_p) - beta * (b.mean() - rf_p)) if np.isfinite(beta) else np.nan
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    corr = float(p.corr(b))

    up = b > 0
    dn = b < 0
    up_cap = (float((1 + p[up]).prod() ** (1 / max(up.sum(), 1)) - 1) /
              float((1 + b[up]).prod() ** (1 / max(up.sum(), 1)) - 1) * 100
              if up.sum() > 1 and abs((1 + b[up]).prod() ** (1 / up.sum()) - 1) > 1e-12
              else np.nan)
    dn_cap = (float((1 + p[dn]).prod() ** (1 / max(dn.sum(), 1)) - 1) /
              float((1 + b[dn]).prod() ** (1 / max(dn.sum(), 1)) - 1) * 100
              if dn.sum() > 1 and abs((1 + b[dn]).prod() ** (1 / dn.sum()) - 1) > 1e-12
              else np.nan)

    return {
        "n": int(len(j)),
        "beta": beta,
        "jensen_alpha_ann_%": alpha_p * periods_per_year * 100 if np.isfinite(alpha_p) else np.nan,
        "r2": corr ** 2 if np.isfinite(corr) else np.nan,
        "correlation": corr,
        "active_return_ann_%": float(active.mean() * periods_per_year * 100),
        "tracking_error_ann_%": te * 100,
        "information_ratio": (float(active.mean() * periods_per_year / te)
                              if te > 0 else np.nan),
        "up_capture_%": up_cap,
        "down_capture_%": dn_cap,
        "capture_spread_pp": (up_cap - dn_cap
                              if np.isfinite(up_cap) and np.isfinite(dn_cap) else np.nan),
        "up_periods": int(up.sum()),
        "down_periods": int(dn.sum()),
        "rf_annual_%": rf_a,
    }


def active_share(weights: pd.Series, bench_weights: pd.Series) -> dict:
    """Half the sum of absolute active weights: how different the book really is.

    Runs 0-100%. A long-only fund below about 60% is closet-indexing and cannot
    justify an active fee; a long/short book routinely exceeds 100% because its
    shorts have no benchmark counterpart, which is why ``long_only_active_share``
    is reported separately -- the standard measure is only interpretable on the
    long side.
    """
    idx = weights.index.union(bench_weights.index)
    w = weights.reindex(idx).fillna(0.0)
    b = bench_weights.reindex(idx).fillna(0.0)
    total = float((w - b).abs().sum() / 2.0)
    wl = w.clip(lower=0)
    wl = wl / wl.sum() if wl.sum() > 0 else wl
    long_only = float((wl - b).abs().sum() / 2.0)
    return {
        "active_share_%": total * 100,
        "long_only_active_share_%": long_only * 100,
        "names_in_book": int((w != 0).sum()),
        "names_in_benchmark": int((b != 0).sum()),
        "overlap_names": int(((w != 0) & (b != 0)).sum()),
        "off_benchmark_weight_%": float(w[b == 0].abs().sum() * 100),
    }


# ==========================================================================
# Drawdowns
# ==========================================================================
def drawdown_table(rets: pd.Series, top: int = 5) -> pd.DataFrame:
    """The worst drawdowns with their dates, depth, length and recovery.

    Depth without duration is half a risk statistic: a 20% drawdown recovered in
    six weeks and a 20% drawdown still open after two years are different events,
    and only one of them ends careers. ``recovered`` is explicitly False for a
    drawdown that has not closed by the end of the sample rather than being
    silently dropped.
    """
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if len(r) < 3:
        return pd.DataFrame()
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    under = eq < peak * (1 - 1e-12)

    episodes, start = [], None
    for i, (dt, flag) in enumerate(under.items()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, len(eq)))

    rows = []
    for s, e in episodes:
        seg = eq.iloc[max(s - 1, 0):e]
        if len(seg) < 2:
            continue
        pk_val = float(seg.iloc[0])
        trough_dt = seg.idxmin()
        depth = float(seg.min() / pk_val - 1.0)
        recovered = e < len(eq)
        rows.append({
            "peak": seg.index[0].date(),
            "trough": trough_dt.date(),
            "recovery": eq.index[e].date() if recovered else None,
            "depth_%": depth * 100,
            "length_periods": int(e - s + 1),
            "to_trough_periods": int(seg.index.get_loc(trough_dt)),
            "to_recover_periods": int(e - seg.index.get_loc(trough_dt) - s + 1) if recovered else None,
            "recovered": recovered,
        })
    if not rows:
        return pd.DataFrame()
    out = (pd.DataFrame(rows).sort_values("depth_%").head(top)
           .reset_index(drop=True))
    out.index = [f"DD{i + 1}" for i in range(len(out))]
    out.attrs["n_episodes"] = len(rows)
    out.attrs["currently_underwater"] = bool(under.iloc[-1])
    return out


def monthly_table(rets: pd.Series) -> pd.DataFrame:
    """Year x month return grid with a year total. The standard tearsheet block.

    Reads a book's character faster than any single statistic: whether losses are
    isolated months or consecutive quarters is visible at a glance and invisible
    in an annualised number.
    """
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return pd.DataFrame()
    m = (1 + r).resample("ME").prod() - 1
    df = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.to_numpy() * 100})
    grid = df.pivot_table(index="y", columns="m", values="r", aggfunc="first")
    grid.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in grid.columns]
    yearly = (1 + m).groupby(m.index.year).prod() - 1
    grid["Year"] = (yearly * 100).reindex(grid.index)
    grid.index.name = None
    return grid.round(2)


def rolling_stats(rets: pd.Series, window: int = 252,
                  rf_annual_pct: float | None = None,
                  periods_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Rolling excess Sharpe, volatility and drawdown.

    A single full-sample Sharpe averages over regimes the book may never see
    again. The rolling series answers the question that matters -- was this
    consistent, or was it one good year -- and it is the input to any honest
    conversation about persistence.
    """
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if len(r) < window + 2:
        return pd.DataFrame()
    rf_a = (config.RISK_FREE_ANNUAL_PCT if rf_annual_pct is None
            else float(rf_annual_pct))
    rf_p = rf_per_period(rf_a, periods_per_year)
    ex = r - rf_p
    mu = ex.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    eq = (1 + r).cumprod()
    out = pd.DataFrame({
        "sharpe": (mu / sd.replace(0, np.nan)) * np.sqrt(periods_per_year),
        "vol_ann_%": sd * np.sqrt(periods_per_year) * 100,
        "return_ann_%": r.rolling(window).mean() * periods_per_year * 100,
        "drawdown_%": ind.drawdown(eq),
    }).dropna(how="all")
    out.attrs["window"] = window
    out.attrs["rf_annual_%"] = rf_a
    return out


# ==========================================================================
# Attribution
# ==========================================================================
def brinson_attribution(port_w: pd.Series, bench_w: pd.Series,
                        name_returns: pd.Series,
                        sectors: Mapping[str, str] | pd.Series,
                        normalise: bool = False) -> pd.DataFrame:
    """Brinson-Fachler: allocation, selection and interaction, by sector.

    Splits active return into the three decisions that produced it:

    * **allocation** -- the sector was over- or under-weighted, and the sector
      beat or lagged the benchmark overall;
    * **selection** -- within the sector, the names held beat the sector;
    * **interaction** -- the cross term, which is what an overweight in a sector
      where selection also worked earns on top.

    Reported separately because they are different skills. A year made entirely
    on allocation says the top-down chain worked; a year made on selection says
    the scorecard did. Claiming both for one number is how a process stops
    learning.

    **Long/short books.** The decomposition is exact only when both weight
    vectors sum to 1. A long/short book sums to its net exposure -- often near
    zero -- and the identity then fails by a residual that can exceed the active
    return itself, which makes the table worse than useless: it looks precise and
    reconciles to nothing.

    There is no way to fix that inside the arithmetic, so this does not pretend
    to. ``normalise=True`` rescales the portfolio weights to sum to 1 and
    attributes *that* book, which is the well-posed question for a long leg
    considered on its own; the short leg should then be reported separately as a
    contribution rather than folded in. ``reconciles`` in ``.attrs`` says whether
    the identity actually closed, and ``assert_attribution_reconciles`` refuses a
    table where it did not.
    """
    sec = pd.Series(sectors) if not isinstance(sectors, pd.Series) else sectors
    idx = port_w.index.union(bench_w.index)
    w_p = port_w.reindex(idx).fillna(0.0)
    w_b = bench_w.reindex(idx).fillna(0.0)
    raw_sum = float(w_p.sum())
    if normalise:
        if abs(raw_sum) < 1e-9:
            raise ValueError(
                "brinson_attribution: cannot normalise a book whose weights sum "
                "to zero. Attribute the long leg on its own, and report the short "
                "leg as a separate contribution.")
        w_p = w_p / raw_sum
    r_n = pd.to_numeric(name_returns.reindex(idx), errors="coerce")
    s = sec.reindex(idx).fillna("?")

    valid = r_n.notna()
    w_p, w_b, r_n, s = w_p[valid], w_b[valid], r_n[valid], s[valid]
    # Unlabelled names form their own bucket rather than vanishing. A groupby on a
    # series with NaN keys silently drops those rows, so a sector map that failed
    # to align would return an empty attribution -- a confidently blank table,
    # which is worse than a visible "?" row.
    s = s.fillna("?")
    if w_b.sum() > 0:
        w_b = w_b / w_b.sum()
    total_bench = float((w_b * r_n).sum())

    rows = []
    for name, members in s.groupby(s).groups.items():
        wp = float(w_p.loc[members].sum())
        wb = float(w_b.loc[members].sum())
        rp = (float((w_p.loc[members] * r_n.loc[members]).sum() / wp)
              if abs(wp) > 1e-12 else 0.0)
        rb = (float((w_b.loc[members] * r_n.loc[members]).sum() / wb)
              if abs(wb) > 1e-12 else 0.0)
        allocation = (wp - wb) * (rb - total_bench)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        rows.append({
            "sector": name,
            "port_weight_%": wp * 100,
            "bench_weight_%": wb * 100,
            "active_weight_pp": (wp - wb) * 100,
            "port_return_%": rp * 100,
            "bench_return_%": rb * 100,
            "allocation_pp": allocation * 100,
            "selection_pp": selection * 100,
            "interaction_pp": interaction * 100,
            "total_pp": (allocation + selection + interaction) * 100,
        })

    if not rows:
        raise ValueError(
            "brinson_attribution: no name has both a return and a weight. Check "
            "that the sector map, the weight vectors and the return series share "
            "an index.")
    out = pd.DataFrame(rows).set_index("sector").sort_values("total_pp", ascending=False)
    total_port = float((w_p * r_n).sum())
    out.attrs["portfolio_return_%"] = total_port * 100
    out.attrs["benchmark_return_%"] = total_bench * 100
    out.attrs["active_return_pp"] = (total_port - total_bench) * 100
    out.attrs["attributed_pp"] = float(out["total_pp"].sum())
    # Brinson-Fachler is exact when both weight vectors sum to 1. A long/short
    # book does not, so the residual is reported rather than hidden -- and
    # `reconciles` says outright whether the identity closed.
    resid = (total_port - total_bench) * 100 - float(out["total_pp"].sum())
    out.attrs["residual_pp"] = resid
    out.attrs["portfolio_weight_sum"] = raw_sum
    out.attrs["normalised"] = bool(normalise)
    out.attrs["reconciles"] = bool(abs(resid) < 1e-6)
    return out


def assert_attribution_reconciles(br: pd.DataFrame, tolerance_pp: float = 0.01,
                                  label: str = "attribution") -> pd.DataFrame:
    """Regression guard: refuse an attribution table that does not add up.

    Allocation + selection + interaction must equal the active return. When it
    does not, the usual cause is a portfolio that does not sum to 1 -- a
    long/short book, or a partially-invested one -- and the correct response is
    to attribute a well-posed sub-book rather than to publish a table with a
    residual larger than the number it is explaining.
    """
    resid = float(br.attrs.get("residual_pp", np.nan))
    if not np.isfinite(resid):
        raise AssertionError(f"{label}: no residual recorded; this table did not "
                             f"come from brinson_attribution")
    if abs(resid) > tolerance_pp:
        raise AssertionError(
            f"{label}: allocation + selection + interaction = "
            f"{br.attrs.get('attributed_pp', float('nan')):.2f}pp but the active "
            f"return is {br.attrs.get('active_return_pp', float('nan')):.2f}pp -- a "
            f"residual of {resid:.2f}pp. Portfolio weights sum to "
            f"{br.attrs.get('portfolio_weight_sum', float('nan')):.3f}, not 1. "
            f"Attribute the long leg with normalise=True and report the short leg "
            f"as a separate contribution.")
    return pd.DataFrame([{
        "active_return_pp": round(float(br.attrs.get("active_return_pp", np.nan)), 4),
        "attributed_pp": round(float(br.attrs.get("attributed_pp", np.nan)), 4),
        "residual_pp": round(resid, 8),
        "normalised": br.attrs.get("normalised", False),
        "result": "attribution reconciles to the active return",
    }])


def contribution(weights: pd.Series, name_returns: pd.Series,
                 sectors: Mapping[str, str] | pd.Series | None = None) -> pd.DataFrame:
    """Per-name contribution to the book's return: weight x return.

    Trivial arithmetic that nonetheless answers the first question asked in every
    review -- what made the money, and what cost it.
    """
    w = weights[weights != 0]
    r = pd.to_numeric(name_returns.reindex(w.index), errors="coerce")
    out = pd.DataFrame({"weight": w, "return_%": r * 100,
                        "contribution_pp": w * r * 100})
    if sectors is not None:
        sec = pd.Series(sectors) if not isinstance(sectors, pd.Series) else sectors
        out["sector"] = sec.reindex(out.index)
    out["side"] = np.where(out["weight"] < 0, "short", "long")
    out = out.sort_values("contribution_pp", ascending=False)
    out.attrs["total_pp"] = float(out["contribution_pp"].sum())
    out.attrs["long_pp"] = float(out.loc[out["side"] == "long", "contribution_pp"].sum())
    out.attrs["short_pp"] = float(out.loc[out["side"] == "short", "contribution_pp"].sum())
    out.attrs["n_winners"] = int((out["contribution_pp"] > 0).sum())
    out.attrs["n_losers"] = int((out["contribution_pp"] < 0).sum())
    return out


# ==========================================================================
# Guard
# ==========================================================================
def assert_excess_return_basis(stats: Mapping, label: str = "book",
                               allow_zero_rf: bool = False) -> pd.DataFrame:
    """Regression guard: refuse a performance block computed against zero.

    The defect this whole module exists to fix is easy to reintroduce -- one call
    that forgets to pass the rate and the Sharpe silently gains half a point. This
    fires on that.

    ``allow_zero_rf=True`` is the documented exception for a genuinely
    market-neutral book funded on collateral, per
    ``config.NEUTRAL_BOOK_RF_IS_ZERO``. It has to be asked for.
    """
    rf = stats.get("rf_annual_%")
    if rf is None:
        raise AssertionError(
            f"{label}: performance block carries no 'rf_annual_%' -- it was not "
            f"computed on excess returns, so its Sharpe/Sortino/Calmar are "
            f"overstated by roughly rf/vol.")
    if float(rf) == 0.0 and not allow_zero_rf:
        raise AssertionError(
            f"{label}: risk-free rate is 0. A long or net-long book must be "
            f"measured against cash. Pass allow_zero_rf=True only for a "
            f"market-neutral book funded on collateral "
            f"(config.NEUTRAL_BOOK_RF_IS_ZERO).")
    return pd.DataFrame([{
        "rf_annual_%": float(rf),
        "sharpe": round(float(stats.get("sharpe", np.nan)), 3),
        "sortino": round(float(stats.get("sortino", np.nan)), 3),
        "calmar": round(float(stats.get("calmar", np.nan)), 3),
        "basis": "excess of risk-free" if float(rf) else "neutral book, rf=0 by design",
        "result": "performance measured on an excess-return basis",
    }])


def summary_frame(blocks: Mapping[str, Mapping]) -> pd.DataFrame:
    """Several named performance blocks side by side, as one comparison table."""
    keep = ["cagr_%", "excess_cagr_%", "ann_vol_%", "sharpe", "sortino", "calmar",
            "max_drawdown_%", "hit_rate_%", "skew", "tail_ratio", "rf_annual_%"]
    out = pd.DataFrame({k: {m: v.get(m, np.nan) for m in keep}
                        for k, v in blocks.items()})
    return out.round(3)
