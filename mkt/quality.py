"""Forensic accounting: Piotroski F, Altman Z, Beneish M, accruals.

Separate from ``fundamentals.py`` on the ``score.py`` precedent. ``fundamentals``
is already 516 lines, and "is this ratio computed correctly?" is a different
question from "is this ratio describing a real business?". This module asks the
second one.

Two rules govern everything here, and both exist because the alternative is a
confident wrong number:

**Never a silent partial score.** Beneish needs eight ratios and Piotroski nine
tests. If one input is missing the score is ``unavailable`` and the missing
component is named -- it is never computed over whatever happened to be present.
``quality_table`` returns per-component coverage so that decision is auditable.

**Financials are excluded, not approximated.** F, Z and M are built on working
capital, gross margin, inventory turns and accruals. A bank has no meaningful
current ratio and no gross margin; its operating cash flow is dominated by
deposit flows. Every score is therefore ``n/a-sector`` for
``config.FINANCIAL_SECTORS``, using the same mechanism ``fundamentals`` uses.

That exclusion is not merely conceptual. Measured across all 50 cached bundles on
2026-09-08: ``Current Assets`` resolves for 38/38 non-financials and 1/12
financials, ``Working Capital`` 38/38 and 1/12, ``Gross Profit`` 38/38 and 1/12.
Data availability and conceptual validity point the same way.

**These scores carry trap #4 too.** They are computed from the *current*
statements, so they describe the cross-section today and cannot be backtested.
Notebook 08 reports their IC as in-sample only, and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, fundamentals as fnd

COMPUTED = fnd.COMPUTED
NA_SECTOR = fnd.NA_SECTOR
MISSING = fnd.MISSING

# Beneish's published thresholds. -1.78 is the manipulation flag from the
# original 1999 paper; -2.22 is the more conservative cut used since.
BENEISH_FLAG = -1.78
BENEISH_WATCH = -2.22

# Altman Z''-score zones for the emerging-market variant.
ALTMAN_SAFE, ALTMAN_DISTRESS = 2.60, 1.10

PIOTROSKI_STRONG, PIOTROSKI_WEAK = 7, 3


def _is_financial(sector: str) -> bool:
    return sector in config.FINANCIAL_SECTORS


def _pair(df: pd.DataFrame, key: str) -> tuple[float, float]:
    """Latest and prior annual value of a statement line.

    Returns ``(nan, nan)`` unless *both* periods exist -- every score here is a
    year-on-year comparison, so one period is not a partial answer, it is no
    answer.
    """
    s = fnd.series_for(df, key).dropna()
    if len(s) < 2:
        return np.nan, np.nan
    return float(s.iloc[-1]), float(s.iloc[-2])


def _fin(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


# ==========================================================================
# Piotroski F-score
# ==========================================================================
def piotroski_f(bundle: dict, sector: str = "") -> tuple[float, dict, str]:
    """Nine binary tests of fundamental improvement. 0-9, higher is better.

    Profitability (4), leverage/liquidity (3), operating efficiency (2), scored
    year-on-year off the annual statements. Quarterly cash flow is not usable --
    ``q_cashflow`` is empty for 45 of the 50 cached bundles -- so this is
    deliberately an annual score, which is how Piotroski defined it anyway.
    """
    if _is_financial(sector):
        return np.nan, {}, NA_SECTOR

    inc = bundle.get("income", pd.DataFrame())
    bal = bundle.get("balance", pd.DataFrame())
    cfs = bundle.get("cashflow", pd.DataFrame())

    ni_t, ni_p = _pair(inc, "net_income")
    rev_t, rev_p = _pair(inc, "revenue")
    gp_t, gp_p = _pair(inc, "gross_profit")
    ta_t, ta_p = _pair(bal, "total_assets")
    ca_t, ca_p = _pair(bal, "current_assets")
    cl_t, cl_p = _pair(bal, "current_liab")
    sh_t, sh_p = _pair(bal, "shares_out")
    ocf_t, ocf_p = _pair(cfs, "ocf")

    # Long-term debt resolves for only 34/38 non-financials; total debt for
    # 38/38. Fall back rather than dropping the whole score for four names, and
    # record which measure was used.
    ltd_t, ltd_p = _pair(bal, "long_term_debt")
    lever_basis = "long_term_debt"
    if not (_fin(ltd_t) and _fin(ltd_p)):
        ltd_t, ltd_p = _pair(bal, "total_debt")
        lever_basis = "total_debt"

    # Piotroski uses beginning-of-period assets as the denominator.
    roa_t = fnd._safe_div(ni_t, ta_p)
    roa_p = fnd._safe_div(ni_p, ta_t if not _fin(ta_p) else ta_p)
    # prior-year ROA needs the year before that; approximate with same-year
    # assets when a third period is unavailable.
    s_ta = fnd.series_for(bal, "total_assets").dropna()
    if len(s_ta) >= 3:
        roa_p = fnd._safe_div(ni_p, float(s_ta.iloc[-3]))

    cfo_t = fnd._safe_div(ocf_t, ta_p)
    lever_t = fnd._safe_div(ltd_t, ta_t)
    lever_p = fnd._safe_div(ltd_p, ta_p)
    liquid_t = fnd._safe_div(ca_t, cl_t)
    liquid_p = fnd._safe_div(ca_p, cl_p)
    margin_t = fnd._safe_div(gp_t, rev_t)
    margin_p = fnd._safe_div(gp_p, rev_p)
    turn_t = fnd._safe_div(rev_t, ta_p)
    turn_p = fnd._safe_div(rev_p, float(s_ta.iloc[-3])) if len(s_ta) >= 3 else np.nan

    tests = {
        "ROA > 0": roa_t,
        "CFO > 0": cfo_t,
        "d ROA > 0": (roa_t - roa_p) if _fin(roa_t) and _fin(roa_p) else np.nan,
        "CFO > ROA (accrual)": (cfo_t - roa_t) if _fin(cfo_t) and _fin(roa_t) else np.nan,
        "d leverage < 0": (lever_p - lever_t) if _fin(lever_t) and _fin(lever_p) else np.nan,
        "d liquidity > 0": (liquid_t - liquid_p) if _fin(liquid_t) and _fin(liquid_p) else np.nan,
        "no new shares": (sh_p - sh_t) if _fin(sh_t) and _fin(sh_p) else np.nan,
        "d gross margin > 0": (margin_t - margin_p) if _fin(margin_t) and _fin(margin_p) else np.nan,
        "d asset turnover > 0": (turn_t - turn_p) if _fin(turn_t) and _fin(turn_p) else np.nan,
    }

    components, missing = {}, []
    for name, v in tests.items():
        if not _fin(v):
            components[name] = np.nan
            missing.append(name)
        else:
            components[name] = 1.0 if v > 0 else 0.0

    components["_lever_basis"] = lever_basis
    components["_missing"] = "; ".join(missing)
    if missing:
        return np.nan, components, MISSING

    total = float(sum(v for k, v in components.items() if not k.startswith("_")))
    if not fnd.in_bounds("piotroski_f", total):
        return np.nan, components, MISSING
    return total, components, COMPUTED


# ==========================================================================
# Altman Z-score, emerging-market variant
# ==========================================================================
def altman_z(bundle: dict, sector: str = "",
             mcap: float | None = None) -> tuple[float, dict, str]:
    """Altman Z''-score for emerging markets.

    ``Z'' = 3.25 + 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4``

    The EM variant deliberately uses **book** equity in X4 rather than market
    value, which is what makes it comparable across markets where the equity
    risk premium differs. Market cap is accepted anyway and reported as a
    separate ``X4_market`` component, so the sensitivity to that choice is
    visible instead of assumed.

    Zones: > 2.6 safe, 1.1-2.6 grey, < 1.1 distress.
    """
    if _is_financial(sector):
        return np.nan, {}, NA_SECTOR

    inc = bundle.get("income", pd.DataFrame())
    bal = bundle.get("balance", pd.DataFrame())

    ta, _ = fnd.stock_value(bal, "total_assets")
    wc, _ = fnd.stock_value(bal, "working_capital")
    re, _ = fnd.stock_value(bal, "retained_earnings")
    eq, _ = fnd.stock_value(bal, "equity")
    tl, _ = fnd.stock_value(bal, "total_liabilities")
    ebit = fnd.latest(inc, "ebit")

    if not _fin(wc):
        ca, _ = fnd.stock_value(bal, "current_assets")
        cl, _ = fnd.stock_value(bal, "current_liab")
        wc = ca - cl if _fin(ca) and _fin(cl) else np.nan
    if not _fin(tl):
        tl = ta - eq if _fin(ta) and _fin(eq) else np.nan

    x1 = fnd._safe_div(wc, ta)
    x2 = fnd._safe_div(re, ta)
    x3 = fnd._safe_div(ebit, ta)
    x4 = fnd._safe_div(eq, tl)

    components = {"X1 WC/TA": x1, "X2 RE/TA": x2, "X3 EBIT/TA": x3,
                  "X4 BookEq/TL": x4}
    if mcap is not None and _fin(mcap):
        components["X4_market MCap/TL"] = fnd._safe_div(mcap, tl)

    missing = [k for k in ("X1 WC/TA", "X2 RE/TA", "X3 EBIT/TA", "X4 BookEq/TL")
               if not _fin(components[k])]
    components["_missing"] = "; ".join(missing)
    if missing:
        return np.nan, components, MISSING

    z = 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    components["zone"] = zone_for_z(z)
    if not fnd.in_bounds("altman_z", z):
        return np.nan, components, MISSING
    return float(z), components, COMPUTED


def zone_for_z(z: float) -> str:
    if not _fin(z):
        return ""
    if z >= ALTMAN_SAFE:
        return "safe"
    if z <= ALTMAN_DISTRESS:
        return "distress"
    return "grey"


# ==========================================================================
# Beneish M-score
# ==========================================================================
_BENEISH_COEF = {
    "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
    "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327,
}
_BENEISH_CONST = -4.84


def beneish_m(bundle: dict, sector: str = "") -> tuple[float, dict, str]:
    """Eight-ratio earnings-manipulation score. Above -1.78 is the flag.

    Every one of the eight is a year-on-year ratio-of-ratios, so all sixteen
    underlying inputs must exist. The row names Yahoo uses are unobvious rather
    than inconsistent -- ``Selling General And Administration`` (not
    "Administrative"), and depreciation filed as ``Reconciled Depreciation`` on
    the income statement. Measured 2026-09-08, all sixteen resolve for 38/38
    non-financials.
    """
    if _is_financial(sector):
        return np.nan, {}, NA_SECTOR

    inc = bundle.get("income", pd.DataFrame())
    bal = bundle.get("balance", pd.DataFrame())
    cfs = bundle.get("cashflow", pd.DataFrame())

    rev_t, rev_p = _pair(inc, "revenue")
    ar_t, ar_p = _pair(bal, "receivables")
    cogs_t, cogs_p = _pair(inc, "cogs")
    ca_t, ca_p = _pair(bal, "current_assets")
    ppe_t, ppe_p = _pair(bal, "net_ppe")
    ta_t, ta_p = _pair(bal, "total_assets")
    sga_t, sga_p = _pair(inc, "sga")
    cl_t, cl_p = _pair(bal, "current_liab")
    ltd_t, ltd_p = _pair(bal, "long_term_debt")
    ni_t, _ = _pair(inc, "net_income")
    ocf_t, _ = _pair(cfs, "ocf")

    dep_t, dep_p = _pair(cfs, "depreciation")
    if not (_fin(dep_t) and _fin(dep_p)):
        dep_t, dep_p = _pair(inc, "depreciation")

    # Same fallback chain as `piotroski_f`: long-term debt resolves for 34/38
    # non-financials, total debt for 38/38. Falling back keeps four names
    # scoreable; treating a missing row as zero debt would be a guess, so that
    # is only the last resort and it is recorded.
    lvgi_basis = "long_term_debt"
    if not (_fin(ltd_t) and _fin(ltd_p)):
        ltd_t, ltd_p = _pair(bal, "total_debt")
        lvgi_basis = "total_debt"
    if not (_fin(ltd_t) and _fin(ltd_p)):
        ltd_t, ltd_p = 0.0, 0.0
        lvgi_basis = "assumed debt-free (no debt row resolved)"

    c: dict[str, float] = {}
    # Days sales in receivables: receivables growing faster than sales.
    c["DSRI"] = fnd._safe_div(fnd._safe_div(ar_t, rev_t), fnd._safe_div(ar_p, rev_p))
    # Gross margin deterioration.
    gm_t = fnd._safe_div(rev_t - cogs_t, rev_t) if _fin(rev_t) and _fin(cogs_t) else np.nan
    gm_p = fnd._safe_div(rev_p - cogs_p, rev_p) if _fin(rev_p) and _fin(cogs_p) else np.nan
    c["GMI"] = fnd._safe_div(gm_p, gm_t)
    # Asset quality: the share of assets that are neither current nor PP&E.
    aq_t = 1 - fnd._safe_div((ca_t or 0) + (ppe_t or 0), ta_t) if _fin(ta_t) else np.nan
    aq_p = 1 - fnd._safe_div((ca_p or 0) + (ppe_p or 0), ta_p) if _fin(ta_p) else np.nan
    c["AQI"] = fnd._safe_div(aq_t, aq_p)
    c["SGI"] = fnd._safe_div(rev_t, rev_p)
    # Depreciation rate slowing = assets being written off more slowly.
    dr_t = fnd._safe_div(dep_t, (dep_t or 0) + (ppe_t or 0))
    dr_p = fnd._safe_div(dep_p, (dep_p or 0) + (ppe_p or 0))
    c["DEPI"] = fnd._safe_div(dr_p, dr_t)
    c["SGAI"] = fnd._safe_div(fnd._safe_div(sga_t, rev_t), fnd._safe_div(sga_p, rev_p))
    lv_t = fnd._safe_div((ltd_t or 0) + (cl_t or 0), ta_t)
    lv_p = fnd._safe_div((ltd_p or 0) + (cl_p or 0), ta_p)
    c["LVGI"] = fnd._safe_div(lv_t, lv_p)
    # Total accruals to total assets -- the single heaviest term in the model.
    c["TATA"] = fnd._safe_div(ni_t - ocf_t, ta_t) if _fin(ni_t) and _fin(ocf_t) else np.nan

    missing = [k for k in _BENEISH_COEF if not _fin(c.get(k))]
    components = dict(c)
    components["_lvgi_basis"] = lvgi_basis
    components["_missing"] = "; ".join(missing)
    if missing:
        return np.nan, components, MISSING

    m = _BENEISH_CONST + sum(_BENEISH_COEF[k] * c[k] for k in _BENEISH_COEF)
    components["flag"] = "manipulation-flag" if m > BENEISH_FLAG else (
        "watch" if m > BENEISH_WATCH else "clean")
    if not fnd.in_bounds("beneish_m", m):
        return np.nan, components, MISSING
    return float(m), components, COMPUTED


# ==========================================================================
# Accruals
# ==========================================================================
def accruals_ratio(bundle: dict, sector: str = "") -> tuple[float, str]:
    """(Net income - operating cash flow) / average total assets.

    High and positive means earnings are being reported that the cash has not
    arrived for. Gated on financials for the same reason as the rest: a bank's
    operating cash flow is a deposit-flow number, not an earnings-quality one.
    """
    if _is_financial(sector):
        return np.nan, NA_SECTOR
    inc = bundle.get("income", pd.DataFrame())
    cfs = bundle.get("cashflow", pd.DataFrame())
    bal = bundle.get("balance", pd.DataFrame())
    ni = fnd.latest(inc, "net_income")
    ocf = fnd.latest(cfs, "ocf")
    ta = fnd.avg_two(bal, "total_assets")
    v = fnd._safe_div(ni - ocf, ta) if _fin(ni) and _fin(ocf) else np.nan
    if not _fin(v) or not fnd.in_bounds("accruals_ratio", v):
        return np.nan, MISSING
    return float(v), COMPUTED


# ==========================================================================
# The table
# ==========================================================================
def quality_table(bundles: dict[str, dict], sectors: dict[str, str],
                  mcaps: dict[str, float] | pd.Series | None = None) -> pd.DataFrame:
    """One row per name: F, Z, M, accruals, their zones and their provenance."""
    mcaps = {} if mcaps is None else dict(mcaps)
    rows, comp_store = [], {}
    for t, b in bundles.items():
        sec = sectors.get(t, "")
        f, f_c, f_p = piotroski_f(b, sec)
        z, z_c, z_p = altman_z(b, sec, mcaps.get(t))
        m, m_c, m_p = beneish_m(b, sec)
        acc, acc_p = accruals_ratio(b, sec)
        comp_store[t] = {"piotroski": f_c, "altman": z_c, "beneish": m_c}
        rows.append({
            "ticker": t, "sector": sec,
            "is_financial": _is_financial(sec),
            "piotroski_f": f, "piotroski_prov": f_p,
            "altman_z": z, "altman_zone": z_c.get("zone", ""), "altman_prov": z_p,
            "beneish_m": m, "beneish_flag": m_c.get("flag", ""), "beneish_prov": m_p,
            "accruals_ratio": acc, "accruals_prov": acc_p,
            "missing_components": "; ".join(
                x for x in (f_c.get("_missing", ""), z_c.get("_missing", ""),
                            m_c.get("_missing", "")) if x),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out.attrs["components"] = comp_store
    return out


def coverage_report(qtable: pd.DataFrame) -> pd.DataFrame:
    """Per-score provenance counts. Verification item: ``n/a-sector`` must equal
    the financials count exactly -- a mismatch means a score leaked past the gate."""
    n_fin = int(qtable["is_financial"].sum())
    rows = []
    for label, col in (("piotroski_f", "piotroski_prov"), ("altman_z", "altman_prov"),
                       ("beneish_m", "beneish_prov"),
                       ("accruals_ratio", "accruals_prov")):
        vals = qtable[col]
        rows.append({
            "score": label,
            COMPUTED: int((vals == COMPUTED).sum()),
            NA_SECTOR: int((vals == NA_SECTOR).sum()),
            MISSING: int((vals == MISSING).sum()),
            "non_financials": len(qtable) - n_fin,
            "coverage_of_eligible_%": round(
                (vals == COMPUTED).sum() / max(len(qtable) - n_fin, 1) * 100, 1),
        })
    out = pd.DataFrame(rows).set_index("score")
    out.attrs["n_financials"] = n_fin
    return out


def component_coverage(qtable: pd.DataFrame, which: str = "beneish") -> pd.DataFrame:
    """How often each individual component resolved. Never report a partial
    score; do report which part would have been missing."""
    store = qtable.attrs.get("components", {})
    eligible = [t for t in qtable.index if not qtable.loc[t, "is_financial"]]
    counts: dict[str, int] = {}
    for t in eligible:
        for k, v in store.get(t, {}).get(which, {}).items():
            if k.startswith("_") or k in ("zone", "flag"):
                continue
            counts[k] = counts.get(k, 0) + (1 if _fin(v) else 0)
    if not counts:
        return pd.DataFrame()
    out = pd.DataFrame({"resolved": pd.Series(counts)})
    out["eligible"] = len(eligible)
    out["coverage_%"] = (out["resolved"] / out["eligible"] * 100).round(1)
    return out.sort_values("coverage_%")


def assert_sector_gate(qtable: pd.DataFrame) -> pd.DataFrame:
    """Guard: the ``n/a-sector`` count must equal the financials count exactly.

    A financial that slipped through would carry a Piotroski score built on a
    current ratio that does not exist -- a number that looks fine and means
    nothing, which is precisely the failure mode this project keeps guarding
    against.
    """
    n_fin = int(qtable["is_financial"].sum())
    bad = []
    for label, col in (("piotroski_f", "piotroski_prov"), ("altman_z", "altman_prov"),
                       ("beneish_m", "beneish_prov"),
                       ("accruals_ratio", "accruals_prov")):
        n_na = int((qtable[col] == NA_SECTOR).sum())
        if n_na != n_fin:
            bad.append((label, n_na, n_fin))
    if bad:
        raise AssertionError(
            f"sector gate leaked: {bad} -- each score must be n/a-sector for exactly "
            f"the {n_fin} financial(s) in the universe")
    return pd.DataFrame([{"financials": n_fin, "scores_checked": 4,
                          "result": "every score is n/a-sector for exactly the financials"}])


# ==========================================================================
# Red flags and the short book
# ==========================================================================
def red_flags(row: pd.Series) -> list[str]:
    """Plain-language flags for one row of the quality table."""
    out = []
    f, z, m, a = (row.get("piotroski_f"), row.get("altman_z"),
                  row.get("beneish_m"), row.get("accruals_ratio"))
    if _fin(f) and f <= PIOTROSKI_WEAK:
        out.append(f"Piotroski {f:.0f}/9 -- fundamentals deteriorating")
    if _fin(z) and z <= ALTMAN_DISTRESS:
        out.append(f"Altman Z {z:.2f} -- distress zone")
    elif _fin(z) and z < ALTMAN_SAFE:
        out.append(f"Altman Z {z:.2f} -- grey zone")
    if _fin(m) and m > BENEISH_FLAG:
        out.append(f"Beneish M {m:+.2f} -- above the manipulation threshold")
    elif _fin(m) and m > BENEISH_WATCH:
        out.append(f"Beneish M {m:+.2f} -- watch band")
    if _fin(a) and a > 0.10:
        out.append(f"accruals {a:+.1%} of assets -- earnings ahead of cash")
    if row.get("is_financial"):
        out.append("financial -- forensic scores not applicable")
    return out


def quality_score(qtable: pd.DataFrame) -> pd.Series:
    """A single 0-100 forensic score, higher = cleaner.

    Equal-weighted percentile blend of F (up), Z (up), M (down) and accruals
    (down), renormalised by what is available -- the same coverage-aware
    treatment ``score.score_block`` gives the other two legs. Financials score
    NaN rather than 50: absent evidence is not neutral evidence.
    """
    parts, weights = [], []
    for col, direction in (("piotroski_f", 1), ("altman_z", 1),
                           ("beneish_m", -1), ("accruals_ratio", -1)):
        s = pd.to_numeric(qtable[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        parts.append(s.rank(pct=True, ascending=(direction > 0)) * 100)
        weights.append(s.notna().astype(float))
    if not parts:
        return pd.Series(np.nan, index=qtable.index, name="quality_score")
    stacked = pd.concat(parts, axis=1)
    avail = pd.concat(weights, axis=1)
    out = (stacked.fillna(0) * avail).sum(axis=1) / avail.sum(axis=1).replace(0, np.nan)
    return out.rename("quality_score")


def short_candidates(qtable: pd.DataFrame, screen: pd.DataFrame,
                     n: int = 8, max_per_sector: int = 2) -> pd.DataFrame:
    """Rank short candidates: weak forensics *and* a weak composite.

    A short needs both halves. Bad accounting alone is a stock that can stay
    expensive for years; a weak tape alone is a stock that is merely cheap. The
    ranking multiplies the two so a name has to fail on both to reach the top,
    and the sector cap stops the book becoming one sector bet inverted.
    """
    q = qtable.copy()
    q["quality_score"] = quality_score(q)
    joined = q.join(screen[[c for c in ("composite", "fund_pct", "tech_pct", "last_close",
                                        "vol_ann_%", "market_cap")
                            if c in screen.columns]], how="inner")
    elig = joined[(~joined["is_financial"]) & joined["quality_score"].notna()
                  & joined["composite"].notna()].copy()
    if elig.empty:
        return elig

    # Both legs on a 0-100 "how bad is it" scale, then averaged.
    elig["forensic_weakness"] = 100 - elig["quality_score"]
    elig["screen_weakness"] = 100 - elig["composite"].rank(pct=True) * 100
    elig["short_score"] = (elig["forensic_weakness"] + elig["screen_weakness"]) / 2
    elig = elig.sort_values("short_score", ascending=False)

    picked, counts = [], {}
    for t, r in elig.iterrows():
        sec = r.get("sector", "?")
        if counts.get(sec, 0) >= max_per_sector:
            continue
        picked.append(t)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picked) == n:
            break
    out = elig.loc[picked].copy()
    out["short_rank"] = range(1, len(out) + 1)
    out["flags"] = [ "; ".join(red_flags(out.loc[t])) for t in out.index ]
    return out


def do_not_own(qtable: pd.DataFrame) -> pd.DataFrame:
    """Names failing a hard forensic test, whatever the screen says about them."""
    q = qtable.copy()
    fail = (
        (pd.to_numeric(q["beneish_m"], errors="coerce") > BENEISH_FLAG)
        | (pd.to_numeric(q["altman_z"], errors="coerce") <= ALTMAN_DISTRESS)
        | (pd.to_numeric(q["piotroski_f"], errors="coerce") <= PIOTROSKI_WEAK)
    )
    out = q[fail.fillna(False)].copy()
    if len(out):
        out["reason"] = ["; ".join(red_flags(out.loc[t])) for t in out.index]
    return out
