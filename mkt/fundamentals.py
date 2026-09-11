"""Fundamental ratios computed from the statements, with provenance tracking.

Trap #2, measured: ``yf.Ticker('RELIANCE.NS').info['returnOnEquity']`` is
silently ``None``, and ``INFY.NS`` reports ``enterpriseToEbitda = 982``, which
is nonsense. ``.info`` is a convenience blob, not a source of truth.

So every ratio here is computed from the income statement, balance sheet and
cash-flow statement. ``.info`` is consulted only when the statements genuinely
lack an input, and any metric that had to fall back is tagged. The scorecard
asserts that every ratio it uses carries a provenance of ``computed`` or
``info-fallback`` -- never an untracked value.

Banks and NBFCs use a different vocabulary (no EBITDA, no meaningful D/E or
ROCE), so those metrics are marked ``n/a-sector`` rather than being computed
into a misleading number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COMPUTED = "computed"
INFO = "info-fallback"
NA_SECTOR = "n/a-sector"
MISSING = "unavailable"

# Yahoo renames statement rows between companies; try candidates in order.
ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income Common Stockholders", "Net Income",
                   "Net Income Including Noncontrolling Interests"],
    "pretax": ["Pretax Income"],
    "interest_exp": ["Interest Expense", "Interest Expense Non Operating",
                     "Total Other Finance Cost"],
    "net_interest_income": ["Net Interest Income"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "shares": ["Diluted Average Shares", "Basic Average Shares"],
    # balance sheet
    "equity": ["Stockholders Equity", "Common Stock Equity",
               "Total Equity Gross Minority Interest"],
    "total_assets": ["Total Assets"],
    "total_debt": ["Total Debt"],
    "current_assets": ["Current Assets"],
    "current_liab": ["Current Liabilities"],
    "cash": ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments"],
    "shares_out": ["Ordinary Shares Number", "Share Issued"],
    "inventory": ["Inventory"],
    "receivables": ["Accounts Receivable", "Receivables"],
    # Forensic inputs (mkt.quality). Names verified against all 50 cached bundles on
    # 2026-09-08: every one resolves for 38/38 non-financials. The naming is
    # unobvious rather than inconsistent -- Yahoo says "Administration", not
    # "Administrative", and files depreciation as "Reconciled Depreciation" on the
    # income statement.
    "retained_earnings": ["Retained Earnings"],
    "gross_ppe": ["Gross PPE"],
    "net_ppe": ["Net PPE"],
    "sga": ["Selling General And Administration",
            "Selling General And Administrative"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "working_capital": ["Working Capital"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    # cash flow
    "ocf": ["Operating Cash Flow"],
    "depreciation": ["Depreciation And Amortization", "Depreciation",
                     "Depreciation Amortization Depletion"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "fcf": ["Free Cash Flow"],
    "dividends_paid": ["Cash Dividends Paid"],
}


# --------------------------------------------------------------------------
# Statement access helpers
# --------------------------------------------------------------------------
def _sorted_cols(df: pd.DataFrame) -> list:
    """Statement periods, oldest first."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return sorted(df.columns)


def series_for(df: pd.DataFrame, key: str) -> pd.Series:
    """One statement line as a time series, oldest first, or empty.

    Candidates are listed in preference order, but the *best-covered* candidate
    wins rather than the first one carrying any data at all. Measured on
    2026-09-08: ``Net Income Common Stockholders`` exists for CIPLA.NS and
    WIPRO.NS with a single non-null period while ``Net Income`` has four, so a
    first-match rule silently truncated those two names to one period and made
    their earnings CAGR unavailable. Six such cases exist across the Nifty 50.

    Ties keep list order, so the preferred row still wins whenever coverage is
    equal -- which is the normal case.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    cols = _sorted_cols(df)
    best, best_n = None, 0
    for name in ROWS.get(key, [key]):
        if name not in df.index:
            continue
        s = pd.to_numeric(df.loc[name], errors="coerce")[cols]
        n = int(s.notna().sum())
        if n > best_n:
            best, best_n = s, n
    return best if best is not None else pd.Series(dtype=float)


def latest(df: pd.DataFrame, key: str, back: int = 0) -> float:
    """Most recent non-null value of a line (back=1 gives the prior period)."""
    s = series_for(df, key).dropna()
    if len(s) <= back:
        return np.nan
    return float(s.iloc[-(back + 1)])


def avg_two(df: pd.DataFrame, key: str) -> float:
    """Average of the latest two periods -- the right denominator for ROE/ROA."""
    s = series_for(df, key).dropna()
    if s.empty:
        return np.nan
    if len(s) == 1:
        return float(s.iloc[-1])
    return float(s.iloc[-2:].mean())


def ttm(df: pd.DataFrame, key: str, periods: int = 4) -> float:
    """Trailing-twelve-month sum from quarterly statements."""
    s = series_for(df, key).dropna()
    if len(s) < periods:
        return np.nan
    return float(s.iloc[-periods:].sum())


def _cagr(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    first, last = s.iloc[0], s.iloc[-1]
    years = len(s) - 1
    if first <= 0 or last <= 0 or years <= 0:
        return np.nan
    return float((last / first) ** (1 / years) - 1) * 100


def _safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


# --------------------------------------------------------------------------
# Bad-input defences
# --------------------------------------------------------------------------
# Measured on 2026-09-08: INFY.NS came out at PE 1317 and PB 447 against real
# values near 15 and 4.8. The cause was not a bad cell -- Yahoo serves INFY and
# HCLTECH statements in USD while quoting the share price in INR, so every
# ratio mixing price with statements was inflated by USDINR (~95x). That is
# handled properly by `currency_factor` below, which converts market cap into
# the statements' currency. Two further guards back it up: outlier repair on
# balance-sheet levels, and a plausibility gate on the finished ratios, so a
# bad input surfaces as `unavailable` rather than as a confident wrong number.

def candidate_fx(info: dict) -> tuple[float, str]:
    """The FX rate that *might* separate the statements from the share price.

    ``financialCurrency`` flags the candidate but is not trusted on its own:
    measured on 2026-09-08, it reads ``USD`` for both INFY.NS and HCLTECH.NS,
    yet only INFY's statements are actually in USD. The field tells us which
    rate to test; ``resolve_scale`` decides whether to apply it.
    """
    stmt_ccy = info.get("financialCurrency")
    price_ccy = info.get("currency")
    if not stmt_ccy or not price_ccy or stmt_ccy == price_ccy:
        return 1.0, ""
    from . import fetch as _fetch
    try:
        rate = _fetch.fx_rate(stmt_ccy, price_ccy)
        return rate, f"{stmt_ccy}/{price_ccy}={rate:,.2f}"
    except Exception as exc:                          # noqa: BLE001
        return 1.0, f"currency mismatch {stmt_ccy}/{price_ccy} UNRESOLVED ({exc})"


def resolve_scale(mcap: float, equity: float, earnings: float,
                  fx: float) -> tuple[float, str]:
    """Decide empirically whether market cap needs converting to statement units.

    Rather than trusting a metadata field, this tests the two candidate scales
    against the shape of the result: a real price-to-book sits roughly between
    0.2 and 40, and a real PE between 1 and 300. Whichever scale lands the
    ratio in a plausible band is the right one; unscaled wins ties because it
    is the normal case.
    """
    if not np.isfinite(mcap) or fx == 1.0:
        return 1.0, ""
    for probe, lo, hi in ((equity, 0.2, 40.0), (earnings, 1.0, 300.0)):
        if not np.isfinite(probe) or probe <= 0:
            continue
        for factor, tag in ((1.0, ""), (fx, "market cap converted to statement currency")):
            ratio = mcap / factor / probe
            if lo <= ratio <= hi:
                return factor, tag
    return 1.0, "currency scale unresolved -- valuation ratios may be off"


def stock_value(df: pd.DataFrame, key: str, max_dev: float = 8.0) -> tuple[float, bool]:
    """Latest balance-sheet level, with a same-line outlier check.

    Balance-sheet stocks move gradually. If the newest value sits more than
    `max_dev`x away from the median of its own history, it is treated as a data
    error and the most recent in-range period is used instead.

    Returns ``(value, repaired)``.
    """
    s = series_for(df, key).dropna()
    if s.empty:
        return np.nan, False
    val = float(s.iloc[-1])
    if len(s) < 3:
        return val, False
    med = float(s.median())
    if med == 0 or not np.isfinite(med):
        return val, False
    ratio = abs(val / med)
    if ratio > max_dev or ratio < 1 / max_dev:
        ok = s[(s / med).abs().between(1 / max_dev, max_dev)]
        if len(ok):
            return float(ok.iloc[-1]), True
    return val, False


# Ranges a real listed company can occupy. Outside these, an input was wrong.
RATIO_BOUNDS = {
    "ROE_%": (-150, 150),
    "ROA_%": (-100, 100),
    "ROCE_%": (-150, 200),
    "gross_margin_%": (-100, 100),
    "op_margin_%": (-200, 100),
    "net_margin_%": (-200, 100),
    "revenue_cagr_%": (-90, 300),
    "earnings_cagr_%": (-90, 400),
    "rev_yoy_q_%": (-99, 500),
    "earn_yoy_q_%": (-500, 900),
    "debt_to_equity": (0, 25),
    "interest_coverage": (-200, 5000),
    "current_ratio": (0, 30),
    "fcf_margin_%": (-300, 100),
    "cash_conversion": (-50, 50),
    # Deliberately loose. With the currency mismatch fixed at source, these are
    # a backstop against absurdity, not a valuation opinion -- Indian names like
    # NESTLEIND legitimately carry a PB in the 50s-60s on a small equity base,
    # and clipping that would replace a real number with a worse one.
    "PE": (0, 500),
    "PB": (0, 80),
    "EV_EBITDA": (0, 250),
    "PEG": (-50, 50),
    "div_yield_%": (0, 30),
    # Forensic scores (mkt.quality). Piotroski is 9 binary tests so its range is
    # exact; the other three are bounded by what a real listed company can produce --
    # an Altman Z of 400 means an input was corrupt, not that the balance sheet is
    # extraordinary.
    "piotroski_f": (0, 9),
    "altman_z": (-10, 20),
    "beneish_m": (-10, 5),
    "accruals_ratio": (-1, 1),
}


def in_bounds(name: str, value: float) -> bool:
    if value is None or not np.isfinite(value):
        return False
    lo, hi = RATIO_BOUNDS.get(name, (-np.inf, np.inf))
    return lo <= value <= hi


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def compute_ratios(bundle: dict, ticker: str, sector: str = "",
                   price: float | None = None) -> tuple[dict, dict]:
    """Compute the fundamental leg for one company.

    Returns ``(metrics, provenance)``. Every key in ``metrics`` has a matching
    key in ``provenance`` -- that pairing is what notebook 06 asserts on.
    """
    inc = bundle.get("income", pd.DataFrame())
    bal = bundle.get("balance", pd.DataFrame())
    cfs = bundle.get("cashflow", pd.DataFrame())
    qinc = bundle.get("q_income", pd.DataFrame())
    info = bundle.get("info", {}) or {}

    is_financial = sector in config.FINANCIAL_SECTORS
    m: dict[str, float] = {}
    p: dict[str, str] = {}
    flags: list[str] = []

    def put(name, value, prov, info_key: str | None = None,
            info_scale: float = 1.0):
        """Record a metric, gated on plausibility.

        A computed value outside ``RATIO_BOUNDS`` means an input was corrupt,
        so it is rejected in favour of ``.info`` (tagged ``info-fallback``) and
        the substitution is flagged. If ``.info`` is also implausible or
        absent, the metric is ``unavailable`` -- never a silent bad number.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = np.nan

        if np.isfinite(v) and in_bounds(name, v):
            m[name], p[name] = v, prov
            return

        if np.isfinite(v) and not in_bounds(name, v):
            flags.append(f"{name}={v:,.1f} out of bounds")

        iv = info.get(info_key) if info_key else None
        if iv is not None:
            try:
                iv = float(iv) * info_scale
            except (TypeError, ValueError):
                iv = None
        if iv is not None and np.isfinite(iv) and in_bounds(name, iv):
            m[name], p[name] = iv, INFO
            return

        m[name], p[name] = np.nan, MISSING

    def put_na(name, why=NA_SECTOR):
        m[name] = np.nan
        p[name] = why

    # ---- income statement -------------------------------------------------
    revenue = latest(inc, "revenue")
    net_income = latest(inc, "net_income")
    ebit = latest(inc, "ebit")
    ebitda = latest(inc, "ebitda")
    gross_profit = latest(inc, "gross_profit")
    interest_exp = latest(inc, "interest_exp")

    equity = avg_two(bal, "equity")
    equity_latest, eq_repaired = stock_value(bal, "equity")
    total_assets = avg_two(bal, "total_assets")
    total_debt, _ = stock_value(bal, "total_debt")
    cash, _ = stock_value(bal, "cash")
    cur_assets, _ = stock_value(bal, "current_assets")
    cur_liab, _ = stock_value(bal, "current_liab")
    shares_out, sh_repaired = stock_value(bal, "shares_out")
    if eq_repaired:
        flags.append("equity: latest period rejected as an outlier")
    if sh_repaired:
        flags.append("shares out: latest period rejected as an outlier")
    # ROE's denominator averages two periods, so a single corrupt cell must be
    # excluded there too or it silently halves the ratio.
    if eq_repaired and np.isfinite(equity_latest):
        equity = equity_latest

    ocf = latest(cfs, "ocf")
    capex = latest(cfs, "capex")
    fcf = latest(cfs, "fcf")
    if not np.isfinite(fcf) and np.isfinite(ocf) and np.isfinite(capex):
        fcf = ocf + capex          # capex is reported negative

    # ---- profitability ----------------------------------------------------
    put("ROE_%", _safe_div(net_income, equity) * 100, COMPUTED,
        info_key="returnOnEquity", info_scale=100)
    put("ROA_%", _safe_div(net_income, total_assets) * 100, COMPUTED,
        info_key="returnOnAssets", info_scale=100)

    if is_financial:
        put_na("ROCE_%")
    else:
        capital_employed = (total_assets - cur_liab
                            if np.isfinite(total_assets) and np.isfinite(cur_liab)
                            else np.nan)
        put("ROCE_%", _safe_div(ebit, capital_employed) * 100, COMPUTED)

    put("gross_margin_%", _safe_div(gross_profit, revenue) * 100, COMPUTED,
        info_key="grossMargins", info_scale=100)
    put("op_margin_%", _safe_div(ebit, revenue) * 100, COMPUTED,
        info_key="operatingMargins", info_scale=100)
    put("net_margin_%", _safe_div(net_income, revenue) * 100, COMPUTED,
        info_key="profitMargins", info_scale=100)

    # ---- growth -----------------------------------------------------------
    put("revenue_cagr_%", _cagr(series_for(inc, "revenue")), COMPUTED)
    put("earnings_cagr_%", _cagr(series_for(inc, "net_income")), COMPUTED)

    q_rev = series_for(qinc, "revenue").dropna()
    q_ni = series_for(qinc, "net_income").dropna()
    put("rev_yoy_q_%",
        (_safe_div(q_rev.iloc[-1], q_rev.iloc[-5]) - 1) * 100 if len(q_rev) >= 5 else np.nan,
        COMPUTED)
    put("earn_yoy_q_%",
        (_safe_div(q_ni.iloc[-1], q_ni.iloc[-5]) - 1) * 100 if len(q_ni) >= 5 else np.nan,
        COMPUTED)

    # ---- balance sheet quality -------------------------------------------
    if is_financial:
        put_na("debt_to_equity")
        put_na("interest_coverage")
        put_na("current_ratio")
    else:
        put("debt_to_equity", _safe_div(total_debt, equity_latest), COMPUTED,
            info_key="debtToEquity", info_scale=0.01)   # info reports it as a %
        put("interest_coverage",
            _safe_div(ebit, abs(interest_exp)) if np.isfinite(interest_exp) else np.nan,
            COMPUTED)
        put("current_ratio", _safe_div(cur_assets, cur_liab), COMPUTED,
            info_key="currentRatio")

    # ---- cash ------------------------------------------------------------
    put("fcf_abs", fcf, COMPUTED)
    fm = _safe_div(fcf, revenue)
    put("fcf_margin_%", fm * 100 if np.isfinite(fm) else np.nan, COMPUTED)
    put("cash_conversion", _safe_div(ocf, net_income), COMPUTED)

    # ---- valuation --------------------------------------------------------
    mcap = np.nan
    mcap_prov = MISSING
    if price is not None and np.isfinite(shares_out) and shares_out > 0:
        mcap, mcap_prov = price * shares_out, COMPUTED
    elif info.get("marketCap"):
        mcap, mcap_prov = float(info["marketCap"]), INFO
    put("market_cap", mcap, mcap_prov)

    # Market cap is in the price currency; the statements may not be. Test the
    # candidate rate against the data rather than trusting the metadata.
    ttm_ni_probe = ttm(qinc, "net_income")
    fx, fx_note = candidate_fx(info)
    scale, scale_note = resolve_scale(
        mcap, equity_latest,
        ttm_ni_probe if np.isfinite(ttm_ni_probe) else net_income, fx)
    if scale_note:
        flags.append(f"{scale_note} [{fx_note}]" if fx_note else scale_note)
    mcap_stmt = mcap / scale if np.isfinite(mcap) else np.nan

    ttm_ni = ttm(qinc, "net_income")
    earnings_for_pe = ttm_ni if np.isfinite(ttm_ni) else net_income
    put("PE", _safe_div(mcap_stmt, earnings_for_pe), COMPUTED, info_key="trailingPE")
    put("PB", _safe_div(mcap_stmt, equity_latest), COMPUTED, info_key="priceToBook")

    if is_financial:
        put_na("EV_EBITDA")
    else:
        ev = (mcap_stmt + (total_debt if np.isfinite(total_debt) else 0)
              - (cash if np.isfinite(cash) else 0)) if np.isfinite(mcap_stmt) else np.nan
        put("EV_EBITDA", _safe_div(ev, ebitda), COMPUTED,
            info_key="enterpriseToEbitda")

    # PEG from the computed PE and the computed earnings CAGR -- info's PEG is
    # None for most Indian names, and its enterpriseToEbitda is unreliable
    # (INFY.NS reports 982).
    g = m.get("earnings_cagr_%", np.nan)
    put("PEG", _safe_div(m.get("PE", np.nan), g) if np.isfinite(g) and g > 0 else np.nan,
        COMPUTED)

    div_paid = latest(cfs, "dividends_paid")
    put("div_yield_%",
        _safe_div(abs(div_paid), mcap_stmt) * 100 if np.isfinite(div_paid) else np.nan,
        COMPUTED, info_key="dividendYield")

    m["data_flags"] = "; ".join(flags)
    m["ticker"] = ticker
    m["sector"] = sector
    m["is_financial"] = is_financial
    m["yf_sector"] = info.get("sector", "")
    m["statement_asof"] = (str(max(inc.columns).date())
                           if isinstance(inc, pd.DataFrame) and not inc.empty else "")
    m["n_annual_periods"] = (len(inc.columns)
                             if isinstance(inc, pd.DataFrame) and not inc.empty else 0)
    return m, p


def provenance_summary(prov_by_ticker: dict[str, dict]) -> pd.DataFrame:
    """Count how each metric was sourced across the universe.

    Notebook 06 prints this: it is the visible proof that ratios came from the
    statements rather than from ``.info``.
    """
    rows = []
    metrics = sorted({k for d in prov_by_ticker.values() for k in d})
    for metric in metrics:
        vals = [d.get(metric, MISSING) for d in prov_by_ticker.values()]
        rows.append({
            "metric": metric,
            COMPUTED: vals.count(COMPUTED),
            INFO: vals.count(INFO),
            NA_SECTOR: vals.count(NA_SECTOR),
            MISSING: vals.count(MISSING),
        })
    out = pd.DataFrame(rows)
    out["coverage_%"] = ((out[COMPUTED] + out[INFO]) /
                         (len(prov_by_ticker) - out[NA_SECTOR]).replace(0, np.nan) * 100).round(1)
    return out.sort_values("coverage_%")


def assert_provenance(prov_by_ticker: dict[str, dict],
                      metrics: list[str]) -> pd.DataFrame:
    """Trap #2 regression guard.

    Every scorecard metric must carry a known provenance for every ticker.
    An untracked value -- one that appeared without us knowing where it came
    from -- fails the build.
    """
    valid = {COMPUTED, INFO, NA_SECTOR, MISSING}
    bad = []
    for tkr, d in prov_by_ticker.items():
        for metric in metrics:
            tag = d.get(metric)
            if tag is None:
                bad.append((tkr, metric, "NO PROVENANCE RECORDED"))
            elif tag not in valid:
                bad.append((tkr, metric, f"unknown tag {tag!r}"))
    if bad:
        raise AssertionError(
            f"{len(bad)} metric(s) without valid provenance, e.g. {bad[:5]}")
    return pd.DataFrame(
        [{"tickers_checked": len(prov_by_ticker), "metrics_checked": len(metrics),
          "result": "all metrics carry a known provenance"}])


def statement_trends(bundle: dict, keys=("revenue", "net_income", "ebit", "ocf")) -> pd.DataFrame:
    """Annual statement lines side by side, oldest first, for the deep dives."""
    inc = bundle.get("income", pd.DataFrame())
    cfs = bundle.get("cashflow", pd.DataFrame())
    out = {}
    for k in keys:
        s = series_for(inc, k)
        if s.empty:
            s = series_for(cfs, k)
        if not s.empty:
            out[k] = s
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df.index = [pd.Timestamp(i).date() for i in df.index]
    return df
