"""Central configuration: paths, tickers, universes, endpoints, parameters.

Everything a notebook might want to change lives here so the notebooks
themselves stay analysis-only.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
DASHBOARD_JSON = OUTPUT_DIR / "dashboard.json"

# Optional real book, hand-maintained: columns `ticker,weight` (or `ticker,shares`).
# Absent by default -- notebook 11 runs on the model book alone when it is missing.
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"

for _d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, CHART_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------
# Notebooks may set mkt.cache.REFRESH = True (or export MKT_REFRESH=1) to force
# a re-fetch of every source regardless of TTL.
REFRESH_DEFAULT = os.environ.get("MKT_REFRESH", "0") not in ("0", "", "false", "False")

CACHE_TTL_HOURS = {
    "prices": 12,
    "fundamentals": 168,      # 1 week -- statements change quarterly
    "nse_indices": 6,
    "nse_flows": 12,
    "nse_status": 1,
    "treasury": 12,
    "worldbank": 720,         # 30 days -- annual data
    # Option chains and OI move intraday, but the endpoints are fragile enough that
    # re-hitting them on every cell run is the bigger risk. 3h is a working compromise.
    "nse_derivatives": 3,
    "default": 24,
}

# --------------------------------------------------------------------------
# Network behaviour
# --------------------------------------------------------------------------
HTTP_TIMEOUT = 25
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.5        # seconds; exponential backoff 1.5, 3.0, 6.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
NSE_BASE = "https://www.nseindia.com"
NSE_ENDPOINTS = {
    "all_indices": "/api/allIndices",
    "fii_dii": "/api/fiidiiTradeReact",
    "market_status": "/api/marketStatus",
    # --- derivatives (notebook 09). Probed live on 2026-09-08; see the verdicts in
    # `mkt/derivatives.py`. The v3 chain endpoint REQUIRES an `expiry` argument taken
    # from `option_chain_expiries` for that same symbol -- omitting it returns `{}`
    # with HTTP 200, and the older /api/option-chain-equities path does the same.
    "option_chain": "/api/option-chain-v3?type={kind}&symbol={symbol}&expiry={expiry}",
    "option_chain_expiries": "/api/option-chain-contract-info?symbol={symbol}",
    "fno_symbols": "/api/master-quote",
    "derivatives_snapshot": "/api/snapshot-derivatives-equity?index=futures&limit=500",
    "live_stock_futures": "/api/liveEquity-derivatives?index=stock_fut",
    # Measured 403 on every attempt, with and without a get-quotes Referer. Kept so the
    # failure is visible in health_check() rather than being quietly dropped.
    "quote_trade_info": "/api/quote-equity?symbol={symbol}&section=trade_info",
}

# Lot sizes live in an archive CSV, not the JSON API.
NSE_LOTS_CSV = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

TREASURY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

WORLDBANK_BASE = "https://api.worldbank.org/v2"

# --------------------------------------------------------------------------
# Macro (notebook 01)
# --------------------------------------------------------------------------
WB_COUNTRIES = {
    "IND": "India",
    "USA": "United States",
    "CHN": "China",
    "EUU": "Euro Area",
    "JPN": "Japan",
}

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (%)",
    "FP.CPI.TOTL.ZG": "Inflation, CPI (%)",
    "NE.EXP.GNFS.ZS": "Exports (% GDP)",
    "NY.GDP.PCAP.CD": "GDP per capita (USD)",
}

WB_START, WB_END = 2005, 2026

# Transmission channel: dollar, energy, metals, currency.
MACRO_TICKERS = {
    "DX-Y.NYB": "US Dollar Index",
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "GC=F": "Gold",
    "HG=F": "Copper",
    "NG=F": "Natural Gas",
    "INR=X": "USDINR",
}

# Chronological crisis chain drawn on the macro charts.
CRISIS_CHAIN = [
    ("2008-09-15", "GFC / Lehman"),
    ("2013-05-22", "Taper tantrum"),
    ("2018-10-03", "2018 tightening"),
    ("2020-03-23", "COVID crash"),
    ("2021-11-01", "Inflation shock"),
    ("2023-03-10", "Hiking cycle / SVB"),
]

# --------------------------------------------------------------------------
# Risk regime (notebook 02)
# --------------------------------------------------------------------------
RISK_TICKERS = {
    "^VIX": "VIX (equity vol)",
    "^MOVE": "MOVE (bond vol)",
    "^OVX": "OVX (oil vol)",
    "^SKEW": "SKEW (tail risk)",
    "^VVIX": "VVIX (vol of vol)",
}

RISK_ASSETS = {
    "^GSPC": "S&P 500",
    "GC=F": "Gold",
    "HG=F": "Copper",
    "DX-Y.NYB": "US Dollar Index",
    "TLT": "US 20y+ Treasuries",
    "HYG": "US High Yield",
    "LQD": "US Investment Grade",
}

# BTC trades 7 days a week: fetched and aligned separately (trap #1).
CRYPTO_TICKERS = {"BTC-USD": "Bitcoin"}

RISK_LOOKBACK_YEARS = 10

# --------------------------------------------------------------------------
# Bonds and rates (notebook 03)
# --------------------------------------------------------------------------
TREASURY_TENORS = [
    "1 Mo", "1.5 Month", "2 Mo", "3 Mo", "4 Mo", "6 Mo",
    "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]

TENOR_YEARS = {
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25,
    "4 Mo": 4 / 12, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}

CURVE_SLOPES = {
    "2s10s": ("2 Yr", "10 Yr"),
    "3m10y": ("3 Mo", "10 Yr"),
    "10s30s": ("10 Yr", "30 Yr"),
    "5s30s": ("5 Yr", "30 Yr"),
}

# NSE fixed-income indices carried in /api/allIndices. Names verified against a
# live pull of the endpoint on 2026-09-08 -- they are not the short "GS" codes.
INDIA_BOND_INDICES = [
    "NIFTY 10 YR BENCHMARK G-SEC",
    "NIFTY 10 YR BENCHMARK G-SEC (CLEAN PRICE)",
    "NIFTY 4-8 YR G-SEC INDEX",
    "NIFTY 8-13 YR G-SEC",
    "NIFTY 11-15 YR G-SEC INDEX",
    "NIFTY 15 YR AND ABOVE G-SEC INDEX",
    "NIFTY COMPOSITE G-SEC INDEX",
]

# Approximate duration of each G-Sec index, for the India duration read.
INDIA_BOND_DURATION = {
    "NIFTY 4-8 YR G-SEC INDEX": 6.0,
    "NIFTY 8-13 YR G-SEC": 10.5,
    "NIFTY 10 YR BENCHMARK G-SEC": 10.0,
    "NIFTY 11-15 YR G-SEC INDEX": 13.0,
    "NIFTY 15 YR AND ABOVE G-SEC INDEX": 18.0,
    "NIFTY COMPOSITE G-SEC INDEX": 9.0,
}

TREASURY_YEARS_BACK = 6

# --------------------------------------------------------------------------
# Global equity (notebook 04)
# --------------------------------------------------------------------------
GLOBAL_EQUITY = {
    "^GSPC": ("S&P 500", "US"),
    "^NDX": ("Nasdaq 100", "US"),
    "^RUT": ("Russell 2000", "US"),
    "^GDAXI": ("DAX", "Europe"),
    "^FTSE": ("FTSE 100", "Europe"),
    "^STOXX50E": ("Euro Stoxx 50", "Europe"),
    "^N225": ("Nikkei 225", "Asia"),
    "^HSI": ("Hang Seng", "Asia"),
    "000001.SS": ("Shanghai Composite", "Asia"),
    "^KS11": ("KOSPI", "Asia"),
    "^TWII": ("Taiwan Weighted", "Asia"),
    "^BVSP": ("Bovespa", "LatAm"),
    "^NSEI": ("Nifty 50", "India"),
    "^BSESN": ("Sensex", "India"),
}

GLOBAL_LOOKBACK_YEARS = 10
BENCHMARK = "^NSEI"
WORLD_BENCHMARK = "^GSPC"

# --------------------------------------------------------------------------
# India market and sectors (notebook 05)
# --------------------------------------------------------------------------
# /api/allIndices tags every index with a `key` category. Selecting on the
# category is more robust than a hardcoded name list -- NSE adds indices often.
NSE_CATEGORIES = {
    "broad": "BROAD MARKET INDICES",
    "sector": "SECTORAL INDICES",
    "thematic": "THEMATIC INDICES",
    "strategy": "STRATEGY INDICES",
    "fixed_income": "FIXED INCOME INDICES",
    "derivatives": "INDICES ELIGIBLE IN DERIVATIVES",
}

# Market-cap curve. Names verified against the live endpoint.
INDIA_BROAD_INDICES = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY MIDCAP 100",
    "NIFTY SMALLCAP 100", "NIFTY 500", "NIFTY MICROCAP 250",
]

# Thematic indices worth carrying alongside the official sector list.
INDIA_THEMATIC_EXTRA = [
    "NIFTY INDIA DEFENCE", "NIFTY CPSE", "NIFTY PSE", "NIFTY COMMODITIES",
    "NIFTY INDIA CONSUMPTION", "NIFTY MNC", "NIFTY INDIA MANUFACTURING",
    "NIFTY INDIA DIGITAL", "NIFTY INFRASTRUCTURE", "NIFTY ENERGY",
    "NIFTY INDIA RAILWAYS PSU", "NIFTY HOUSING", "NIFTY TRANSPORTATION & LOGISTICS",
]

INDIA_VIX = "INDIA VIX"
NIFTY_INDEX_NAME = "NIFTY 50"

# The FII/DII endpoint returns ONLY the latest session -- there is no history
# API (verified: ?date= is ignored, /api/historical/ returns 503/404). Every run
# appends its snapshot here so a real flow history accumulates over time.
FII_DII_HISTORY = DATA_DIR / "fii_dii_history.csv"

# --------------------------------------------------------------------------
# Nifty 50 universe (notebook 06)
# --------------------------------------------------------------------------
# Snapshot of the index. Any ticker that fails to resolve on Yahoo is reported
# in the notebook's dropped-ticker list rather than silently skipped.
NIFTY_50 = {
    "ADANIENT.NS": "Metals & Mining",
    "ADANIPORTS.NS": "Services",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer Durables",
    "AXISBANK.NS": "Financial Services",
    "BAJAJ-AUTO.NS": "Automobile",
    "BAJFINANCE.NS": "Financial Services",
    "BAJAJFINSV.NS": "Financial Services",
    "BEL.NS": "Capital Goods",
    "BHARTIARTL.NS": "Telecom",
    "CIPLA.NS": "Healthcare",
    "COALINDIA.NS": "Oil Gas & Fuels",
    "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Automobile",
    "ETERNAL.NS": "Consumer Services",
    "GRASIM.NS": "Construction Materials",
    "HCLTECH.NS": "Information Technology",
    "HDFCBANK.NS": "Financial Services",
    "HDFCLIFE.NS": "Financial Services",
    "HEROMOTOCO.NS": "Automobile",
    "HINDALCO.NS": "Metals & Mining",
    "HINDUNILVR.NS": "FMCG",
    "ICICIBANK.NS": "Financial Services",
    "INDUSINDBK.NS": "Financial Services",
    "INFY.NS": "Information Technology",
    "ITC.NS": "FMCG",
    "JIOFIN.NS": "Financial Services",
    "JSWSTEEL.NS": "Metals & Mining",
    "KOTAKBANK.NS": "Financial Services",
    "LT.NS": "Construction",
    "M&M.NS": "Automobile",
    "MARUTI.NS": "Automobile",
    "NESTLEIND.NS": "FMCG",
    "NTPC.NS": "Power",
    "ONGC.NS": "Oil Gas & Fuels",
    "POWERGRID.NS": "Power",
    "RELIANCE.NS": "Oil Gas & Fuels",
    "SBILIFE.NS": "Financial Services",
    "SBIN.NS": "Financial Services",
    "SHRIRAMFIN.NS": "Financial Services",
    "SUNPHARMA.NS": "Healthcare",
    "TCS.NS": "Information Technology",
    "TATACONSUM.NS": "FMCG",
    # Tata Motors demerged; the old TATAMOTORS.NS ticker no longer resolves on
    # Yahoo (verified 404 on 2026-09-08). TMPV.NS is the passenger-vehicle entity.
    "TMPV.NS": "Automobile",
    "TATASTEEL.NS": "Metals & Mining",
    "TECHM.NS": "Information Technology",
    "TITAN.NS": "Consumer Durables",
    "TRENT.NS": "Consumer Services",
    "ULTRACEMCO.NS": "Construction Materials",
    "WIPRO.NS": "Information Technology",
}

# Financials use a different ratio vocabulary (no EBITDA, no classical D/E).
FINANCIAL_SECTORS = {"Financial Services"}

# --------------------------------------------------------------------------
# Indicator parameters (swing / positional horizon: days to weeks)
# --------------------------------------------------------------------------
DMA_WINDOWS = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
REL_STRENGTH_WINDOW = 63        # ~3 months of trading days
VOL_WINDOW = 21
PRICE_HISTORY_PERIOD = "3y"     # for the company screen
LONG_HISTORY_PERIOD = "20y"     # for macro / regime work

# Composite scorecard weighting (confirmed: 50/50).
SCORE_WEIGHTS = {"fundamental": 0.50, "technical": 0.50}
TOP_N_DEEP_DIVE = 8
MAX_PER_SECTOR = 2              # enforces sector diversification in the top 8

# --------------------------------------------------------------------------
# Signal validation (notebook 07)
# --------------------------------------------------------------------------
# Defaults to the Nifty 50 for consistency with the rest of the chain, but 50 names
# is thin for cross-sectional statistics: quintiles of 10, and a rank-IC standard
# error near 1/sqrt(49) = 0.14. Widening this is a one-line change and is worth
# doing before trusting any per-factor conclusion.
BACKTEST_UNIVERSE = NIFTY_50
BACKTEST_PERIOD = "10y"
REBALANCE_FREQ = "ME"           # month-end
IC_HORIZONS = (21, 63, 126)     # trading days: ~1m, ~3m, ~6m
N_QUANTILES = 5
COST_BPS = 25                   # round-trip, applied to turnover

# --------------------------------------------------------------------------
# Portfolio construction (notebook 10)
# --------------------------------------------------------------------------
TARGET_PORTFOLIO_VOL_PCT = 12.0
MAX_POSITION_WEIGHT = 0.12      # per name, gross
MAX_SECTOR_WEIGHT = 0.30        # per sector, gross

# Gross exposure conditioned on the top-down chain: notebook 02's risk label.
# This is the whole point of having built the macro chain -- risk-off cuts gross.
GROSS_BY_REGIME = {"Risk-On": 1.5, "Neutral": 1.0, "Risk-Off": 0.5}

MAX_PCT_OF_ADV = 0.10           # no position larger than 10% of 20d average volume
DEFAULT_CAPITAL = 10_000_000    # INR 1 crore

# Static lot-size fallback, snapshotted from NSE's fo_mktlots.csv on 2026-09-08.
# Notebook 09 replaces this with the live table; notebook 10 needs it so it can run
# standalone when the derivatives endpoints are unavailable. Lot sizes are revised
# periodically by SEBI, so a stale entry is a real risk -- `derivatives.lot_sizes()`
# reports how far it diverged when both are present.
LOT_SIZE_FALLBACK = {
    "ADANIENT.NS": 309, "ADANIPORTS.NS": 475, "APOLLOHOSP.NS": 125,
    "ASIANPAINT.NS": 250, "AXISBANK.NS": 625, "BAJAJ-AUTO.NS": 75,
    "BAJFINANCE.NS": 750, "BAJAJFINSV.NS": 300, "BEL.NS": 1425,
    "BHARTIARTL.NS": 475, "CIPLA.NS": 425, "COALINDIA.NS": 1350,
    "DRREDDY.NS": 625, "EICHERMOT.NS": 100, "ETERNAL.NS": 2425,
    "GRASIM.NS": 250, "HCLTECH.NS": 400, "HDFCBANK.NS": 650,
    "HDFCLIFE.NS": 1100, "HEROMOTOCO.NS": 150, "HINDALCO.NS": 700,
    "HINDUNILVR.NS": 300, "ICICIBANK.NS": 700, "INDUSINDBK.NS": 700,
    "INFY.NS": 400, "ITC.NS": 1725, "JIOFIN.NS": 2350, "JSWSTEEL.NS": 675,
    "KOTAKBANK.NS": 2000, "LT.NS": 175, "M&M.NS": 200, "MARUTI.NS": 50,
    "NESTLEIND.NS": 500, "NTPC.NS": 1500, "ONGC.NS": 2250,
    "POWERGRID.NS": 1900, "RELIANCE.NS": 500, "SBILIFE.NS": 375,
    "SBIN.NS": 750, "SHRIRAMFIN.NS": 825, "SUNPHARMA.NS": 350,
    "TCS.NS": 225, "TATACONSUM.NS": 550, "TMPV.NS": 1600,
    "TATASTEEL.NS": 2750, "TECHM.NS": 600, "TITAN.NS": 175,
    "TRENT.NS": 225, "ULTRACEMCO.NS": 50, "WIPRO.NS": 3000,
}
LOT_SIZE_FALLBACK_ASOF = "2026-09-08"

# Short book sizing. Cash-market shorts cannot be carried overnight in India, so the
# short leg is stock futures -- whole lots only (trap #5).
SHORT_LEG_INSTRUMENT = "stock futures (overnight cash shorts are not permitted)"
MIN_SHORT_NAMES = 3

# --------------------------------------------------------------------------
# Portfolio risk (notebook 11)
# --------------------------------------------------------------------------
VAR_CONFIDENCE = (0.95, 0.99)
VAR_HORIZON_DAYS = 1
COV_LOOKBACK_DAYS = 504         # ~2 years of daily returns

# Dated ranges around the CRISIS_CHAIN anchors, for replaying real episodes on the
# current book. A 10y panel does not reach the first two; `replay_episodes` reports
# coverage rather than silently returning fewer rows.
STRESS_EPISODES = {
    "GFC / Lehman":        ("2008-09-01", "2009-03-09"),
    "Taper tantrum":       ("2013-05-22", "2013-09-04"),
    "2018 tightening":     ("2018-10-01", "2018-12-24"),
    "COVID crash":         ("2020-02-19", "2020-03-23"),
    "Inflation shock":     ("2021-11-01", "2022-06-16"),
    "Hiking cycle / SVB":  ("2023-03-08", "2023-03-20"),
}

# Forward scenarios, expressed as shocks to observable factors. Betas are estimated
# by regressing each name's returns on these factors -- they are estimates, not
# certainties, and the notebook prints the R-squared next to every one.
SCENARIO_FACTORS = {
    "crude": "BZ=F",
    "usdinr": "INR=X",
    "us_duration": "TLT",
    "world_equity": "^GSPC",
    "gold": "GC=F",
    "dollar": "DX-Y.NYB",
}

SCENARIOS = {
    "Crude +30%":      {"crude": 0.30},
    "USDINR +5%":      {"usdinr": 0.05},
    # A +100bp move in the US 10y is roughly -17% on TLT at its ~17y duration; the
    # shock is expressed through the traded proxy because that is what a beta can be
    # estimated against.
    "US10Y +100bp":    {"us_duration": -0.17},
    "Global risk-off": {"world_equity": -0.10, "gold": 0.05, "dollar": 0.03,
                        "us_duration": 0.04},
}

# Freshness tolerance for the source health table, in calendar days.
STALENESS_TOLERANCE_DAYS = {
    "prices": 5,
    "nse": 5,
    "treasury": 7,
    "worldbank": 800,           # annual series publish with a long lag
    "nse_derivatives": 3,       # contracts expire; a week-old chain is not a chain
}


# ==========================================================================
# INSTITUTIONAL LAYER (notebooks 12-14)
# ==========================================================================
# Everything below was added when the chain was raised to an institutional
# standard. It is kept in one block so the original research configuration
# above stays readable, and so a reviewer can see exactly what the fund layer
# assumes.

# --------------------------------------------------------------------------
# Risk-free rate
# --------------------------------------------------------------------------
# Every Sharpe, Sortino, Calmar, alpha and information ratio in `mkt.perf` is
# an EXCESS-return statistic. Computing them against zero -- which the original
# `risk.drawdown_summary` and `backtest.curve_stats` did -- overstates a long
# book's Sharpe by roughly rf/vol, i.e. about 0.5 at a 6.25% rate and 13% vol.
# That is not a rounding error; it is the difference between "beats cash" and
# "does not".
#
# There is no keyless daily source for the Indian T-bill. This is therefore a
# STATED PRIOR with an as-of date, overridable per call and per notebook, and
# `perf.risk_free_annual()` prefers a live short rate from the dashboard when
# notebook 03 has published one.
RISK_FREE_ANNUAL_PCT = 6.25          # ~91-day T-bill / overnight MIBOR area
RISK_FREE_ASOF = "2026-09-09"
RISK_FREE_SOURCE = "stated prior (RBI policy corridor); override per call"

# A market-neutral book funded on collateral earns the rate on its cash, so its
# spread is already an excess return. Long-only and net-long books are not.
NEUTRAL_BOOK_RF_IS_ZERO = True

# --------------------------------------------------------------------------
# Transaction costs -- the India stack, per leg and per instrument
# --------------------------------------------------------------------------
# A flat round-trip number cannot be right for this book, because the two legs
# are different instruments carrying different statutory charges:
#
#   * the long leg is cash-market delivery: STT on BOTH sides at 0.10%, stamp
#     duty on the buy;
#   * the short leg is stock futures: STT on the SELL only at 0.05% (raised from
#     0.02% by Union Budget 2026, effective 1 April 2026), a lower exchange
#     charge, and stamp duty on the buy.
#
# Netting those into one 25bps figure hides that the short leg's statutory cost
# is roughly a third of the long leg's, which changes where turnover is worth
# spending. Rates below are statutory as of COST_ASOF and must be re-checked
# against the current SEBI/exchange circulars; brokerage is the only negotiable
# line and is set at an institutional level, not a retail one.
COST_ASOF = "2026-09-11"
COST_RATES = {
    # every rate is a FRACTION of the traded value unless stated otherwise
    "cash_delivery": {
        "brokerage":        0.00030,   # 3bps -- institutional; retail is far higher
        "stt_buy":          0.00100,   # 0.10% delivery, charged on both sides
        "stt_sell":         0.00100,
        "exchange_txn":     0.0000297, # NSE cash segment
        "sebi_fee":         0.000001,  # INR 10 per crore
        "stamp_buy":        0.00015,   # buy side only
        "stamp_sell":       0.0,
        "gst_rate":         0.18,      # on brokerage + exchange + SEBI fee
    },
    "stock_futures": {
        "brokerage":        0.00020,   # 2bps
        "stt_buy":          0.0,
        "stt_sell":         0.00050,   # 0.05%, sell side only (Budget 2026; was 0.02%)
        "exchange_txn":     0.0000173, # NSE F&O segment
        "sebi_fee":         0.000001,
        "stamp_buy":        0.00002,   # 0.002%, buy side only
        "stamp_sell":       0.0,
        "gst_rate":         0.18,
    },
}

# Market impact: the square-root law, impact = c * sigma_daily * sqrt(participation)
# where participation is order value / ADV. This is the Almgren/Kissell form used
# across the industry; IMPACT_COEF is the only free parameter and 0.8 is the
# conventional mid-range calibration for liquid single names.
IMPACT_COEF = 0.8
HALF_SPREAD_BPS = {"cash_delivery": 2.0, "stock_futures": 1.5}
# Past this participation rate the square-root model is extrapolating beyond what
# it was calibrated on; `costs.trade_cost` flags rather than silently scaling.
IMPACT_PARTICIPATION_WARN = 0.20

# --------------------------------------------------------------------------
# Style factor risk model (notebook 12)
# --------------------------------------------------------------------------
# A cross-sectional fundamental factor model in the Barra tradition, built from
# the universe itself rather than bought. Exposures are standardised factor
# scores; factor returns are estimated by weighted cross-sectional regression on
# each date; the factor covariance and the residual variances then decompose the
# book's risk into a systematic part and an idiosyncratic part.
STYLE_FACTORS = {
    # factor -> (source column, direction)
    "size":       ("log_mcap",        +1),
    "value":      ("earnings_yield",  +1),
    "momentum":   ("mom_12_1_%",      +1),
    "low_vol":    ("vol_ann_%",       -1),
    "quality":    ("ROE_%",           +1),
    "growth":     ("revenue_cagr_%",  +1),
}
FACTOR_HALFLIFE_DAYS = 126        # exponential weighting on the factor covariance
FACTOR_MIN_NAMES = 20             # below this a cross-sectional regression is noise
FACTOR_WINSOR_Z = 3.0

# --------------------------------------------------------------------------
# Statistical validation (notebook 07 hardening)
# --------------------------------------------------------------------------
FDR_ALPHA = 0.10                  # Benjamini-Hochberg false discovery rate
PURGE_EMBARGO_PCT = 0.01          # embargo as a fraction of the sample, post-purge
CSCV_SPLITS = 10                  # combinatorially symmetric CV blocks, for PBO
BOOTSTRAP_N = 1000
BOOTSTRAP_BLOCK = 6               # stationary-bootstrap mean block, in rebalances

# --------------------------------------------------------------------------
# Margin (notebook 13)
# --------------------------------------------------------------------------
# SPAN+ELM is computed by the exchange from a proprietary daily risk array and
# cannot be reproduced exactly without the SPAN file. What can be done honestly
# is a documented approximation with the right shape: SPAN is close to a
# worst-case scanned loss over a price grid, and ELM is a flat add-on. Both are
# labelled approximations everywhere they surface, and the regulatory floor is
# applied because that is what actually binds on a low-vol name.
#
# Read the result as a FLOOR. Broker-quoted initial margin on single-stock futures
# commonly runs well above this proxy's ~8-9% of notional; size cash against the
# broker's SPAN figure before trading.
SPAN_SCAN_RANGE_SIGMA = 3.5       # price scan range, in daily sigmas
SPAN_MIN_SCAN_PCT = 0.05          # floor on the scan range
ELM_PCT = 0.035                   # extreme loss margin, flat on notional (stock futures: 3.5%)
MARGIN_MTM_BUFFER_DAYS = 3        # cash held back against mark-to-market calls
MARGIN_UTILISATION_LIMIT = 0.65   # of deployable capital; above this a call bites

# --------------------------------------------------------------------------
# Risk limits (notebook 14) -- the register the book is monitored against
# --------------------------------------------------------------------------
# A limit nobody measures is a sentence in a document. Each entry is
# (metric, operator, threshold, severity); `limits.monitor` evaluates every one
# against the live book and appends any breach to a dated log.
RISK_LIMITS = [
    ("gross_exposure",        "<=", 2.00, "hard"),
    ("net_exposure_abs",      "<=", 0.50, "hard"),
    ("net_beta_abs",          "<=", 0.20, "soft"),
    ("max_position_weight",   "<=", 0.12, "hard"),
    ("max_sector_weight",     "<=", 0.30, "hard"),
    ("ex_ante_vol_%",         "<=", 15.0, "soft"),
    ("var_95_1d_%",           "<=", 2.50, "soft"),
    ("cvar_95_1d_%",          "<=", 3.50, "soft"),
    ("pct_risk_top_name",     "<=", 25.0, "soft"),
    ("effective_bets",        ">=", 4.00, "soft"),
    ("margin_utilisation",    "<=", 0.65, "hard"),
    ("worst_stress_loss_%",   ">=", -20.0, "hard"),
    ("days_to_liquidate_p95", "<=", 5.00, "soft"),
]
BREACH_LOG = OUTPUT_DIR / "breach_log.csv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RUN_MANIFEST = OUTPUT_DIR / "run_manifest.json"
IC_MEMO = OUTPUT_DIR / "ic_memo.md"
RANDOM_SEED = 20260909
