"""Cross-sectional scoring: robust z-scores, composites, diversified selection.

Kept separate from ``fundamentals`` and ``indicators`` because scoring spans
both legs. The company scorecard weights them 50/50 (``config.SCORE_WEIGHTS``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def robust_z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score on the median/MAD, clipped.

    The mean/stdev version is dominated by one or two outliers -- a single
    120x PE would flatten every other name's valuation score to nearly zero.
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale == 0:
        scale = x.std()
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(0.0, index=s.index)
    return ((x - med) / scale).clip(-clip, clip)


def score_block(df: pd.DataFrame, spec: dict[str, tuple[float, int]],
                label: str = "") -> tuple[pd.Series, pd.DataFrame]:
    """Weighted composite over several columns.

    ``spec`` maps column -> (weight, direction), direction +1 = higher is
    better, -1 = lower is better. Missing values score 0 (neutral) rather than
    dropping the name, and the per-name coverage is returned so a thinly
    covered score can be discounted rather than trusted blindly.
    """
    parts, used = {}, {}
    for col, (w, direction) in spec.items():
        if col not in df.columns:
            continue
        z = robust_z(df[col]) * direction
        parts[col] = z * w
        used[col] = df[col].notna()
    if not parts:
        return pd.Series(0.0, index=df.index), pd.DataFrame(index=df.index)

    contrib = pd.DataFrame(parts)
    present = pd.DataFrame(used)
    weights = pd.Series({c: spec[c][0] for c in contrib.columns})
    eff_weight = present.mul(weights, axis=1).sum(axis=1)
    raw = contrib.where(present, 0.0).sum(axis=1)
    total_w = weights.sum()
    score = raw / eff_weight.replace(0, np.nan) * total_w
    score = score.fillna(0.0)
    contrib["_coverage_%"] = (eff_weight / total_w * 100).round(0)
    contrib["_score"] = score
    if label:
        contrib.attrs["label"] = label
    return score, contrib


def to_percentile(s: pd.Series) -> pd.Series:
    """Map a score to 0-100 so the two legs combine on a common scale."""
    return s.rank(pct=True) * 100


def composite(fundamental: pd.Series, technical: pd.Series,
              weights: dict | None = None) -> pd.DataFrame:
    """Blend the two legs on a 0-100 percentile scale."""
    w = weights or config.SCORE_WEIGHTS
    f_pct = to_percentile(fundamental)
    t_pct = to_percentile(technical)
    total = w["fundamental"] + w["technical"]
    comp = (f_pct * w["fundamental"] + t_pct * w["technical"]) / total
    return pd.DataFrame({
        "fund_score": fundamental,
        "tech_score": technical,
        "fund_pct": f_pct,
        "tech_pct": t_pct,
        "composite": comp,
    })


def select_diversified(df: pd.DataFrame, n: int = config.TOP_N_DEEP_DIVE,
                       max_per_sector: int = config.MAX_PER_SECTOR,
                       sector_col: str = "sector",
                       score_col: str = "composite") -> pd.DataFrame:
    """Top n by score with a per-sector cap.

    Without the cap the ranking can return eight financials, which is a
    concentrated bet dressed up as a screen. If the cap cannot fill n names,
    the remaining slots are filled by score and flagged.
    """
    ranked = df.sort_values(score_col, ascending=False)
    picked, counts = [], {}
    for tkr, row in ranked.iterrows():
        sec = row.get(sector_col, "?")
        if counts.get(sec, 0) >= max_per_sector:
            continue
        picked.append(tkr)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picked) == n:
            break

    relaxed = []
    if len(picked) < n:
        for tkr in ranked.index:
            if tkr not in picked:
                picked.append(tkr)
                relaxed.append(tkr)
            if len(picked) == n:
                break

    out = ranked.loc[picked].copy()
    out["sector_cap_relaxed"] = out.index.isin(relaxed)
    out["rank"] = range(1, len(out) + 1)
    return out


# Scorecard specification: (weight, direction). Weights are relative within
# each leg; the two legs are then combined 50/50.
FUNDAMENTAL_SPEC = {
    "ROE_%":            (1.4, +1),
    "ROCE_%":           (1.0, +1),
    "op_margin_%":      (0.8, +1),
    "net_margin_%":     (0.6, +1),
    "revenue_cagr_%":   (1.2, +1),
    "earnings_cagr_%":  (1.2, +1),
    "rev_yoy_q_%":      (0.8, +1),
    "earn_yoy_q_%":     (0.8, +1),
    "debt_to_equity":   (0.9, -1),
    "interest_coverage": (0.7, +1),
    "fcf_margin_%":     (0.9, +1),
    "cash_conversion":  (0.5, +1),
    "PE":               (1.0, -1),
    "PB":               (0.6, -1),
    "EV_EBITDA":        (0.6, -1),
    "div_yield_%":      (0.4, +1),
}

TECHNICAL_SPEC = {
    "px_vs_DMA200_%":   (1.2, +1),
    "px_vs_DMA50_%":    (0.9, +1),
    "px_vs_DMA20_%":    (0.5, +1),
    "n_dma_above":      (0.8, +1),
    "rs_63d_pp":        (1.4, +1),
    "rs_126d_pp":       (1.0, +1),
    "ret_3m_%":         (0.9, +1),
    "ret_6m_%":         (0.7, +1),
    "ret_12m_%":        (0.5, +1),
    "pos_52w_%":        (0.8, +1),
    "dd_from_high_%":   (0.6, +1),
    "RSI14":            (0.4, +1),
    "vol_ann_%":        (0.5, -1),
    "ATR14_%":          (0.3, -1),
    "vol_trend_%":      (0.3, +1),
}
