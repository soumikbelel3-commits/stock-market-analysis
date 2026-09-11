"""Margin and capital adequacy for the futures short leg.

Why this module exists
----------------------
The README named SPAN+ELM as one of the three constraints that make trap #5 real,
and then never computed it. That leaves a gap with a specific shape: the chain
could prove a short leg was expressible in **whole lots**, and could not answer
whether the account could actually **carry** it.

Those are different questions with different answers. A short leg that quantises
perfectly into three lots is still undeployable if those three lots demand 70% of
capital as margin, because the long leg then has nothing left to buy with, and
the first adverse mark-to-market becomes a forced unwind rather than a drawdown.

What is approximated, and how honestly
--------------------------------------
SPAN is computed by the exchange from a proprietary daily risk array -- sixteen
scenarios across price and volatility, plus inter-month and short-option-minimum
charges. It **cannot** be reproduced from public data, and any module claiming
otherwise is lying about its inputs.

What can be done honestly is an approximation with the correct shape and clearly
stated error bars. SPAN's initial margin for a single futures position is close to
its worst-case scanned loss, which is close to a multiple of daily volatility with
a regulatory floor underneath it. So:

    initial_margin ~ max(scan_sigma * daily_vol, floor) * notional   (SPAN proxy)
                   + ELM_PCT * notional                              (exact, flat)

The ELM leg is exact -- it is a flat percentage by rule. The SPAN leg is an
estimate, and every function that returns one labels it ``approximate`` so the
number can never be mistaken for a broker's figure. Real margin is what the broker
charges; this is what tells you whether to expect a problem before you find out.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from . import config, indicators as ind

TRADING_DAYS = ind.TRADING_DAYS


# ==========================================================================
# Per-position margin
# ==========================================================================
def span_proxy(daily_vol: float | pd.Series,
               scan_sigma: float | None = None,
               floor_pct: float | None = None) -> float | pd.Series:
    """Scan-range proxy for SPAN initial margin, as a fraction of notional.

    ``max(scan_sigma * daily_vol, floor)``. The floor is what actually binds on a
    low-volatility large cap -- a name at 1.1% daily vol scans to 3.85% at 3.5
    sigma, below the 5% floor -- which is why omitting it would understate margin
    on exactly the names a conservative book holds most of.
    """
    s = config.SPAN_SCAN_RANGE_SIGMA if scan_sigma is None else float(scan_sigma)
    f = config.SPAN_MIN_SCAN_PCT if floor_pct is None else float(floor_pct)
    if np.isscalar(daily_vol):
        return float(max(s * float(daily_vol), f))
    v = pd.to_numeric(daily_vol, errors="coerce")
    return (v * s).clip(lower=f).rename("span_pct")


def position_margin(notional: float | pd.Series, daily_vol: float | pd.Series,
                    scan_sigma: float | None = None,
                    elm_pct: float | None = None) -> pd.DataFrame:
    """SPAN proxy + ELM for one or many futures positions.

    Returns the breakdown rather than a total, because the two components behave
    differently: ELM is fixed by rule and SPAN moves with volatility, so a
    vol spike raises margin on a position whose size never changed. That is the
    mechanism behind most forced unwinds, and it is invisible in a single number.
    """
    elm = config.ELM_PCT if elm_pct is None else float(elm_pct)
    span_pct = span_proxy(daily_vol, scan_sigma)

    if np.isscalar(notional) and np.isscalar(daily_vol):
        n = float(notional)
        return pd.DataFrame([{
            "notional": n,
            "span_pct": float(span_pct),
            "elm_pct": elm,
            "total_pct": float(span_pct) + elm,
            "span_INR": float(span_pct) * n,
            "elm_INR": elm * n,
            "margin_INR": (float(span_pct) + elm) * n,
            "basis": "SPAN approximated by scan range; ELM exact",
        }])

    n = pd.to_numeric(pd.Series(notional), errors="coerce")
    sp = pd.Series(span_pct).reindex(n.index)
    out = pd.DataFrame({
        "notional": n,
        "span_pct": sp,
        "elm_pct": elm,
        "total_pct": sp + elm,
        "span_INR": sp * n,
        "elm_INR": elm * n,
        "margin_INR": (sp + elm) * n,
    })
    out.attrs["basis"] = "SPAN approximated by scan range; ELM exact"
    out.attrs["scan_sigma"] = (config.SPAN_SCAN_RANGE_SIGMA if scan_sigma is None
                               else scan_sigma)
    return out


# ==========================================================================
# Book-level requirement
# ==========================================================================
def book_margin(book: pd.DataFrame, prices: pd.Series, lots: pd.Series,
                daily_vol: pd.Series, capital: float = config.DEFAULT_CAPITAL,
                units: pd.Series | None = None) -> pd.DataFrame:
    """Margin required by every short leg, name by name.

    Only shorts appear: the long leg is cash-market delivery, paid for in full,
    and carries no margin requirement. That asymmetry is the reason the two legs
    of a "market-neutral" book consume capital so differently, and the reason a
    book can be dollar-neutral and nowhere near capital-neutral.

    ``units`` accepts the realised whole-lot counts from
    ``portfolio.quantize_to_lots``; without it the notional is taken from target
    weights, which understates margin whenever rounding pushed a position up.
    """
    shorts = book[book["side"] == "short"] if "side" in book.columns else book[book["weight"] < 0]
    if shorts.empty:
        out = pd.DataFrame(columns=["notional", "margin_INR"])
        out.attrs["total_margin"] = 0.0
        out.attrs["note"] = "no short leg -- cash equity carries no margin"
        return out

    px = pd.to_numeric(prices.reindex(shorts.index), errors="coerce")
    lt = pd.to_numeric(lots.reindex(shorts.index), errors="coerce")
    dv = pd.to_numeric(daily_vol.reindex(shorts.index), errors="coerce")
    dv = dv.fillna(dv.median())

    if units is not None:
        u = pd.to_numeric(units.reindex(shorts.index), errors="coerce").abs()
        notional = (u * lt * px) if (u * lt * px).notna().any() else None
        basis = "realised whole lots"
        if notional is None:
            notional = shorts["weight"].abs() * capital
            basis = "target weights (lot counts unusable)"
    else:
        notional = shorts["weight"].abs() * capital
        basis = "target weights"

    m = position_margin(notional, dv)
    out = pd.DataFrame({
        "target_weight": shorts["weight"],
        "price": px,
        "lot": lt,
        "notional": m["notional"],
        "daily_vol_%": dv * 100,
        "span_pct": m["span_pct"] * 100,
        "elm_pct": m["elm_pct"] * 100,
        "margin_pct": m["total_pct"] * 100,
        "margin_INR": m["margin_INR"],
        "margin_%_of_capital": m["margin_INR"] / capital * 100,
    })
    out.attrs["notional_basis"] = basis
    out.attrs["total_margin"] = float(out["margin_INR"].sum())
    out.attrs["total_notional"] = float(out["notional"].sum())
    out.attrs["capital"] = capital
    out.attrs["approximate"] = True
    return out.sort_values("margin_INR", ascending=False)


def capital_adequacy(book: pd.DataFrame, margin_table: pd.DataFrame,
                     capital: float = config.DEFAULT_CAPITAL,
                     mtm_buffer_days: int | None = None,
                     book_daily_vol: float | None = None) -> dict:
    """Does the account have the cash to carry this book? The full accounting.

    Four claims on the same rupees, and a book is only deployable if they fit:

    1. **Long leg** -- paid in full, cash delivery.
    2. **Short-leg margin** -- SPAN + ELM, blocked for as long as the position is on.
    3. **Mark-to-market buffer** -- futures settle daily in cash. A short that moves
       against the book generates a *cash* call the same evening, whatever the
       long leg is doing. Holding nothing back for that is how a solvent book gets
       liquidated by its own broker.
    4. **Free cash** -- what is left, which had better not be negative.

    The buffer is sized at ``mtm_buffer_days`` daily standard deviations of the
    short leg's notional, which is a deliberately crude and deliberately generous
    rule: it is a liquidity reserve, not a risk estimate, and being wrong in the
    conservative direction costs a little return while being wrong in the other
    direction costs the book.
    """
    days = config.MARGIN_MTM_BUFFER_DAYS if mtm_buffer_days is None else int(mtm_buffer_days)
    longs = book[book["weight"] > 0]["weight"].sum() if len(book) else 0.0
    long_cash = float(longs) * capital
    margin = float(margin_table.attrs.get("total_margin", 0.0)) if len(margin_table) else 0.0
    short_notional = float(margin_table.attrs.get("total_notional", 0.0)) if len(margin_table) else 0.0

    vol = book_daily_vol
    if vol is None and len(margin_table) and "daily_vol_%" in margin_table.columns:
        w = margin_table["notional"]
        vol = float((margin_table["daily_vol_%"] / 100 * w).sum() / w.sum()) if w.sum() else 0.0
    vol = float(vol or 0.0)
    mtm_buffer = short_notional * vol * np.sqrt(max(days, 1))

    used = long_cash + margin + mtm_buffer
    free = capital - used
    util = margin / capital if capital else np.nan

    return {
        "capital": capital,
        "long_leg_cash": long_cash,
        "short_leg_notional": short_notional,
        "short_leg_margin": margin,
        "mtm_buffer": mtm_buffer,
        "mtm_buffer_days": days,
        "total_committed": used,
        "free_cash": free,
        "free_cash_%": free / capital * 100 if capital else np.nan,
        "margin_utilisation": util,
        "margin_utilisation_limit": config.MARGIN_UTILISATION_LIMIT,
        "leverage_on_capital": (long_cash + short_notional) / capital if capital else np.nan,
        "adequate": bool(free >= 0 and util <= config.MARGIN_UTILISATION_LIMIT),
        "basis": "SPAN approximated; ELM exact; buffer is a liquidity reserve",
    }


def margin_stress(margin_table: pd.DataFrame, capital: float = config.DEFAULT_CAPITAL,
                  vol_multiples: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """What margin becomes when volatility spikes.

    The scenario that ends books. Margin is not a fixed claim -- it scales with
    the scan range, so the day the market moves is the day the requirement rises,
    on positions that have simultaneously moved against you. A book at 55%
    utilisation in calm conditions can be at 100% after a doubling of volatility
    without a single trade, and the response demanded is forced selling into the
    move that caused it.

    The floor matters here too: names already pinned at the regulatory minimum
    absorb a 1.5x vol shock with no margin increase at all, which is why the
    escalation is non-linear.
    """
    if margin_table.empty:
        return pd.DataFrame()
    notional = margin_table["notional"]
    dv = margin_table["daily_vol_%"] / 100.0

    rows = []
    for m in vol_multiples:
        span_pct = span_proxy(dv * m)
        total_pct = span_pct + config.ELM_PCT
        req = float((total_pct * notional).sum())
        rows.append({
            "vol_multiple": m,
            "margin_INR": req,
            "margin_%_of_capital": req / capital * 100 if capital else np.nan,
            "utilisation": req / capital if capital else np.nan,
            "breaches_limit": bool(req / capital > config.MARGIN_UTILISATION_LIMIT)
            if capital else False,
            "n_at_floor": int((dv * m * config.SPAN_SCAN_RANGE_SIGMA
                               <= config.SPAN_MIN_SCAN_PCT).sum()),
        })
    out = pd.DataFrame(rows).set_index("vol_multiple")
    base = out["margin_INR"].iloc[0]
    out["increase_vs_base_%"] = (out["margin_INR"] / base - 1) * 100 if base else np.nan
    out.attrs["capital"] = capital
    out.attrs["approximate"] = True
    return out


# ==========================================================================
# Guard
# ==========================================================================
def assert_capital_adequate(adequacy: Mapping, label: str = "book") -> pd.DataFrame:
    """Regression guard: refuse a book the account cannot carry.

    The natural companion to ``portfolio.assert_implementable``. That guard proves
    the short leg can be *expressed* in whole lots; this one proves it can be
    *held*. A book can pass the first and fail the second, and the failure surfaces
    at the broker rather than in the notebook unless something checks.

    Reports the capital that would clear it, in the same style as trap #5 --
    "raise the book to INR 1.8 crore or drop a short" is actionable, "insufficient
    margin" is not.
    """
    free = float(adequacy.get("free_cash", np.nan))
    util = float(adequacy.get("margin_utilisation", np.nan))
    cap = float(adequacy.get("capital", np.nan))
    limit = float(adequacy.get("margin_utilisation_limit", config.MARGIN_UTILISATION_LIMIT))

    problems = []
    if np.isfinite(free) and free < 0:
        committed = float(adequacy.get("total_committed", np.nan))
        problems.append(
            f"committed {committed:,.0f} against capital {cap:,.0f} -- short by "
            f"{-free:,.0f}. Minimum capital that would clear: {committed:,.0f}")
    if np.isfinite(util) and util > limit:
        need = float(adequacy.get("short_leg_margin", 0.0)) / limit if limit > 0 else np.inf
        problems.append(
            f"margin utilisation {util:.1%} exceeds the {limit:.0%} limit -- a vol "
            f"spike becomes a forced unwind. Minimum capital at this short leg: "
            f"{need:,.0f}")

    if problems:
        raise AssertionError(f"{label}: " + "; ".join(problems))

    return pd.DataFrame([{
        "capital": cap,
        "long_leg_cash": round(float(adequacy.get("long_leg_cash", np.nan)), 0),
        "short_leg_margin": round(float(adequacy.get("short_leg_margin", np.nan)), 0),
        "mtm_buffer": round(float(adequacy.get("mtm_buffer", np.nan)), 0),
        "free_cash_%": round(float(adequacy.get("free_cash_%", np.nan)), 2),
        "margin_utilisation": round(util, 4),
        "result": "capital adequate to carry the book (SPAN approximated)",
    }])
