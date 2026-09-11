"""Transaction costs: the India statutory stack, plus market impact.

Why this module exists
----------------------
The original chain charged a flat ``COST_BPS = 25`` round trip against turnover.
That is a reasonable placeholder and a poor institutional answer, for three
reasons this module fixes:

1. **The two legs are different instruments.** The long leg is cash-market
   delivery -- STT at 0.10% on *both* sides, stamp duty on the buy. The short leg
   is stock futures -- STT at 0.05% on the *sell only* (0.02% before Budget
   2026), a lower exchange charge, negligible stamp. Statutory cost on the short
   leg is roughly a third of the long leg's. A single blended number cannot express that, and it is exactly the
   sort of thing that decides whether a marginal short is worth putting on.

2. **Cost is not linear in size.** Impact grows with the square root of
   participation, so a book that doubles its capital does not double its cost --
   it more than doubles it. Without that term there is no capacity question, and
   capacity is the first thing an allocator asks about.

3. **Direction matters.** Entering a long and entering a short are not the same
   cost, and neither is exiting. Charging a symmetric round trip to both blurs the
   asymmetry that trap #5 already established for sizing.

What is modelled
----------------
``explicit`` -- brokerage, STT, exchange transaction charge, SEBI turnover fee,
stamp duty, and GST on the taxable subset. All statutory rates live in
``config.COST_RATES`` with an as-of date, because they change by circular.

``impact`` -- half-spread plus the square-root law,
``c * sigma_daily * sqrt(value / ADV)``, the standard Almgren/Kissell form.
``config.IMPACT_COEF`` is the single free parameter. Above
``IMPACT_PARTICIPATION_WARN`` participation the model is extrapolating past its
calibration range and says so rather than returning a confident number.

What is not modelled
--------------------
Timing risk, opportunity cost, and the borrow that does not exist here because
the short leg is futures rather than stock loan. Those are real costs; they are
just not estimable from public data, and inventing them would be worse than
naming them.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from . import config

CASH = "cash_delivery"
FUT = "stock_futures"


# ==========================================================================
# Explicit (statutory + brokerage) costs
# ==========================================================================
def explicit_bps(instrument: str = CASH, side: str = "buy",
                 rates: Mapping | None = None) -> pd.Series:
    """Every statutory line item for one side of one trade, in basis points.

    Returned as a Series rather than a float so the breakdown is auditable: when
    a cost number is challenged, the answer should be a table naming STT and
    stamp duty, not a single figure nobody can reconstruct.

    GST applies to brokerage, the exchange transaction charge and the SEBI fee --
    **not** to STT or stamp duty, which are themselves taxes. Getting that subset
    wrong is the most common error in a hand-rolled India cost model.
    """
    table = dict(rates or config.COST_RATES)
    if instrument not in table:
        raise KeyError(f"explicit_bps: unknown instrument {instrument!r}; "
                       f"known: {sorted(table)}")
    r = table[instrument]
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"explicit_bps: side must be 'buy' or 'sell', got {side!r}")

    brokerage = float(r["brokerage"])
    exch = float(r["exchange_txn"])
    sebi = float(r["sebi_fee"])
    stt = float(r[f"stt_{side}"])
    stamp = float(r[f"stamp_{side}"])
    gst = (brokerage + exch + sebi) * float(r["gst_rate"])

    out = pd.Series({
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn": exch,
        "sebi_fee": sebi,
        "stamp_duty": stamp,
        "gst": gst,
    }) * 10_000.0
    out["total"] = out.sum()
    out.attrs["instrument"] = instrument
    out.attrs["side"] = side
    out.attrs["as_of"] = config.COST_ASOF
    # Deliberately unrounded. This feeds round_trip_bps, trade_cost and the
    # capacity curve; rounding a computational return value for display costs
    # composability and gains nothing -- the caller rounds when it prints.
    return out


def cost_matrix(rates: Mapping | None = None) -> pd.DataFrame:
    """Every instrument x side combination, in bps. The reference table.

    Printing this once in a notebook settles the question of why the short leg's
    entry is cheap and its exit is not, without anyone having to read the code.
    """
    rows = {}
    for inst in (rates or config.COST_RATES):
        for side in ("buy", "sell"):
            rows[(inst, side)] = explicit_bps(inst, side, rates)
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["instrument", "side"])
    out.attrs["as_of"] = config.COST_ASOF
    out.attrs["units"] = "basis points of traded value"
    return out


def round_trip_bps(instrument: str = CASH, is_short: bool = False,
                   rates: Mapping | None = None) -> float:
    """Explicit cost of a full round trip, in bps.

    A short opens with a sell and closes with a buy; a long is the reverse. On the
    cash leg that is symmetric because STT is charged both ways, but on futures it
    is not -- which is the whole point of not blending the two.
    """
    first, second = ("sell", "buy") if is_short else ("buy", "sell")
    return float(explicit_bps(instrument, first, rates)["total"]
                 + explicit_bps(instrument, second, rates)["total"])


# ==========================================================================
# Market impact
# ==========================================================================
def impact_bps(value: float | pd.Series, adv_value: float | pd.Series,
               daily_vol: float | pd.Series,
               instrument: str = CASH,
               coef: float | None = None) -> pd.Series | float:
    """Square-root market impact plus the half-spread, in bps.

    ``impact = half_spread + c * sigma_daily * sqrt(value / ADV)``

    The square root is the empirically robust part: doubling the order does not
    double the impact, it multiplies it by about 1.41. That concavity is what
    makes a capacity curve bend rather than run straight, and it is why a book
    that is comfortable at INR 1 crore can be uneconomic at INR 50 crore without
    any single position looking wrong.

    ``daily_vol`` is a fraction (0.018 for 1.8% a day), not a percent.
    """
    c = config.IMPACT_COEF if coef is None else float(coef)
    half = config.HALF_SPREAD_BPS.get(instrument, 2.0)

    v = pd.Series(value, dtype="float64") if not np.isscalar(value) else float(value)
    a = pd.Series(adv_value, dtype="float64") if not np.isscalar(adv_value) else float(adv_value)
    s = pd.Series(daily_vol, dtype="float64") if not np.isscalar(daily_vol) else float(daily_vol)

    participation = np.where(np.asarray(a, dtype=float) > 0,
                             np.asarray(v, dtype=float) / np.asarray(a, dtype=float),
                             np.nan)
    imp = c * np.asarray(s, dtype=float) * np.sqrt(np.clip(participation, 0, None)) * 10_000.0
    total = half + imp

    if np.isscalar(value) and np.isscalar(adv_value) and np.isscalar(daily_vol):
        return float(total)
    idx = getattr(v, "index", getattr(a, "index", None))
    return pd.Series(np.asarray(total, dtype=float), index=idx, name="impact_bps")


def trade_cost(book: pd.DataFrame, capital: float = config.DEFAULT_CAPITAL,
               adv_value: pd.Series | None = None,
               daily_vol: pd.Series | None = None,
               entry_only: bool = False) -> pd.DataFrame:
    """Per-name all-in cost of putting on -- and taking off -- the book.

    ``book`` needs a signed ``weight`` column and a ``side`` column, i.e. exactly
    what ``portfolio.build_long_short`` returns. Longs are priced as cash
    delivery, shorts as stock futures, which is the only correct mapping given
    trap #5: an overnight short in India *is* a future.

    Set ``entry_only=True`` to price the entry alone -- useful when the question
    is what today's rebalance costs rather than what the whole position will cost
    over its life.
    """
    if "weight" not in book.columns:
        raise ValueError("trade_cost: book needs a 'weight' column")
    side = (book["side"] if "side" in book.columns
            else pd.Series(np.where(book["weight"] < 0, "short", "long"),
                           index=book.index))

    rows = []
    for t, w in book["weight"].items():
        is_short = str(side.get(t, "long")).lower() == "short"
        inst = FUT if is_short else CASH
        value = abs(float(w)) * capital

        first, second = ("sell", "buy") if is_short else ("buy", "sell")
        exp_in = float(explicit_bps(inst, first)["total"])
        exp_out = 0.0 if entry_only else float(explicit_bps(inst, second)["total"])

        adv = float(adv_value.get(t, np.nan)) if adv_value is not None else np.nan
        vol = float(daily_vol.get(t, np.nan)) if daily_vol is not None else np.nan
        if np.isfinite(adv) and np.isfinite(vol) and adv > 0:
            imp_in = float(impact_bps(value, adv, vol, inst))
            imp_out = 0.0 if entry_only else imp_in
            participation = value / adv
        else:
            imp_in = imp_out = np.nan
            participation = np.nan

        total_bps = exp_in + exp_out + (0.0 if not np.isfinite(imp_in) else imp_in + imp_out)
        rows.append({
            "ticker": t,
            "side": "short" if is_short else "long",
            "instrument": inst,
            "weight": float(w),
            "value": value,
            "explicit_in_bps": exp_in,
            "explicit_out_bps": exp_out,
            "impact_in_bps": imp_in,
            "impact_out_bps": imp_out,
            "total_bps": total_bps,
            "total_INR": total_bps / 10_000.0 * value,
            "participation": participation,
            "extrapolating": bool(np.isfinite(participation)
                                  and participation > config.IMPACT_PARTICIPATION_WARN),
        })

    out = pd.DataFrame(rows).set_index("ticker")
    gross_value = float(out["value"].sum())
    out.attrs["capital"] = capital
    out.attrs["entry_only"] = entry_only
    out.attrs["total_INR"] = float(out["total_INR"].sum())
    out.attrs["total_bps_of_capital"] = (float(out["total_INR"].sum()) / capital * 10_000.0
                                         if capital else np.nan)
    out.attrs["weighted_avg_bps"] = (float((out["total_bps"] * out["value"]).sum()
                                           / gross_value) if gross_value > 0 else np.nan)
    out.attrs["n_extrapolating"] = int(out["extrapolating"].sum())
    out.attrs["as_of"] = config.COST_ASOF
    return out


# ==========================================================================
# Turnover-based cost for the backtest
# ==========================================================================
def blended_round_trip_bps(long_share: float = 0.5,
                           impact_bps_est: float = 0.0) -> float:
    """One round-trip number for the L/S backtest, derived rather than assumed.

    ``backtest.long_short_curve`` charges a single bps figure against turnover,
    which is the right shape for a quantile backtest -- the names change every
    month and there is no ADV panel behind them. What was wrong was that the
    figure was a guess. This derives it from the same statutory table the live
    book uses, weighted by how much of the turnover is long.

    Pass ``impact_bps_est`` to add an assumed impact term; leaving it at zero
    gives the pure statutory floor, which is the honest lower bound on what a
    quantile strategy would have paid.
    """
    ls = float(np.clip(long_share, 0.0, 1.0))
    cash_rt = round_trip_bps(CASH, is_short=False)
    fut_rt = round_trip_bps(FUT, is_short=True)
    return float(ls * cash_rt + (1 - ls) * fut_rt + impact_bps_est)


def cost_drag(turnover: pd.Series, cost_bps: float,
              periods_per_year: float = 12.0) -> dict:
    """What a turnover series costs, annualised.

    Turnover here is one-way per rebalance, matching ``backtest.turnover``.
    """
    t = pd.to_numeric(turnover, errors="coerce").dropna()
    if t.empty:
        return {}
    per_period = t * cost_bps / 10_000.0
    return {
        "avg_turnover_one_way": float(t.mean()),
        "cost_bps_per_round_trip": float(cost_bps),
        "cost_per_period_%": float(per_period.mean() * 100),
        "cost_ann_%": float(per_period.mean() * periods_per_year * 100),
        "n_periods": int(len(t)),
    }


# ==========================================================================
# Capacity
# ==========================================================================
def capacity_curve(book: pd.DataFrame, adv_value: pd.Series, daily_vol: pd.Series,
                   capitals: list[float] | None = None,
                   gross_alpha_bps_ann: float = 300.0,
                   turnovers_per_year: float = 12.0) -> pd.DataFrame:
    """Cost as a function of book size -- where the strategy stops paying.

    This is the allocator's question, and it has an answer only because impact is
    concave: explicit cost is linear in size and vanishes as a share, while impact
    grows as the square root of participation and eventually eats everything.
    The crossing point is capacity.

    ``gross_alpha_bps_ann`` is the annual gross alpha the book is assumed to earn.
    It is a stated input, not an estimate from this project -- notebook 07's
    verdict on the technical leg was "not proven", and the honest use of this
    function is to ask "how much alpha would this need to survive its own costs",
    which is what the ``breakeven_alpha_bps`` column reports.
    """
    caps = capitals or [1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9]
    rows = []
    for cap in caps:
        tc = trade_cost(book, capital=cap, adv_value=adv_value,
                        daily_vol=daily_vol, entry_only=False)
        rt_bps = tc.attrs["weighted_avg_bps"]
        ann_bps = rt_bps * turnovers_per_year
        rows.append({
            "capital": cap,
            "round_trip_bps": rt_bps,
            "ann_cost_bps": ann_bps,
            "ann_cost_%": ann_bps / 100.0,
            "net_alpha_bps": gross_alpha_bps_ann - ann_bps,
            "breakeven_alpha_bps": ann_bps,
            "max_participation": float(tc["participation"].max()),
            "n_extrapolating": int(tc["extrapolating"].sum()),
        })
    out = pd.DataFrame(rows).set_index("capital")
    viable = out[out["net_alpha_bps"] > 0]
    out.attrs["gross_alpha_bps_ann"] = gross_alpha_bps_ann
    out.attrs["capacity_INR"] = float(viable.index.max()) if len(viable) else np.nan
    out.attrs["turnovers_per_year"] = turnovers_per_year
    return out


# ==========================================================================
# Guard
# ==========================================================================
def assert_cost_sanity(tc: pd.DataFrame, max_bps: float = 200.0,
                       label: str = "book") -> pd.DataFrame:
    """Regression guard: refuse a cost estimate that is not a cost estimate.

    Two failures this catches, both silent otherwise:

    * a per-name round trip above ``max_bps`` (2%) -- almost always a units bug,
      a rate entered as a percent where a fraction was expected, or an ADV in
      shares where rupees were meant;
    * a short leg priced as cash equity, which understates nothing but *overstates*
      STT fourfold and would send sizing the wrong way.
    """
    problems = []
    bad = tc[tc["total_bps"] > max_bps]
    if len(bad):
        problems.append(
            f"{len(bad)} name(s) cost more than {max_bps:.0f}bps round trip "
            f"(worst {bad['total_bps'].max():.0f}bps on {bad['total_bps'].idxmax()}) "
            f"-- check that ADV is in rupees and rates are fractions")

    mismapped = tc[(tc["side"] == "short") & (tc["instrument"] != FUT)]
    if len(mismapped):
        problems.append(
            f"{len(mismapped)} short leg(s) priced as {CASH!r} rather than {FUT!r} "
            f"-- an overnight short in India is a future (trap #5)")

    neg = tc[tc["total_bps"] < 0]
    if len(neg):
        problems.append(f"{len(neg)} name(s) have negative cost")

    if problems:
        raise AssertionError(f"{label}: " + "; ".join(problems))

    return pd.DataFrame([{
        "names": len(tc),
        "weighted_avg_bps": round(tc.attrs.get("weighted_avg_bps", np.nan), 2),
        "total_INR": round(tc.attrs.get("total_INR", np.nan), 0),
        "n_extrapolating": tc.attrs.get("n_extrapolating", 0),
        "result": "cost model internally consistent",
    }])
