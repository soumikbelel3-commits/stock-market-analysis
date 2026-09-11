"""Retrying fetchers for every data source in the project.

Design rules, each of which exists because a probe failed:

* Every network call retries with exponential backoff (``^VIX`` failed once and
  worked on retry -- a single attempt would have silently produced an all-NaN
  volatility series).
* A failed live fetch falls back to cache and says so; it never returns an
  empty frame dressed up as data. Empty results raise.
* Every frame carries ``.attrs['as_of']`` -- markets close at different times,
  so a stale close must never be compared against a fresh one (trap #3).
* Price history is fetched one ticker at a time. Batch ``yf.download`` across
  markets forces a union calendar and injects spurious NaNs (trap #1); joining
  is the caller's job, via ``mkt.align``.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from . import cache, config

# --------------------------------------------------------------------------
# Status registry -- what health_check() reports on
# --------------------------------------------------------------------------
_STATUS: list[dict] = []


def _record(source: str, status: str, rows: int | None = None,
            as_of: Any = None, elapsed: float | None = None,
            detail: str = "") -> None:
    _STATUS.append({
        "source": source,
        "status": status,
        "rows": rows,
        "as_of": as_of,
        "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
        "detail": detail,
        "logged_at": datetime.now(timezone.utc),
    })


def status_log() -> pd.DataFrame:
    """Every fetch attempt this session, newest last."""
    if not _STATUS:
        return pd.DataFrame(columns=["source", "status", "rows", "as_of",
                                     "elapsed_s", "detail", "logged_at"])
    return pd.DataFrame(_STATUS)


def reset_status() -> None:
    _STATUS.clear()


class FetchError(RuntimeError):
    """Raised when a source cannot be served live or from cache."""


def _retry(fn: Callable[[], Any], label: str,
           attempts: int = config.RETRY_ATTEMPTS,
           base_delay: float = config.RETRY_BASE_DELAY) -> Any:
    """Call fn with exponential backoff. Raises FetchError after `attempts`."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001 - deliberate
            last = exc
            if i < attempts - 1:
                wait = base_delay * (2 ** i)
                print(f"  [retry {i + 1}/{attempts - 1}] {label}: "
                      f"{type(exc).__name__}: {str(exc)[:110]} -- waiting {wait:.1f}s")
                time.sleep(wait)
    raise FetchError(f"{label} failed after {attempts} attempts: "
                     f"{type(last).__name__}: {last}") from last


def _stamp(df: pd.DataFrame, source: str, as_of: Any) -> pd.DataFrame:
    df.attrs["source"] = source
    df.attrs["as_of"] = as_of
    df.attrs["fetched_at"] = datetime.now(timezone.utc)
    return df


# ==========================================================================
# NSE India
# ==========================================================================
_nse_sess = None
_nse_kind = ""


def nse_session(force_new: bool = False):
    """A cookie-warmed NSE session.

    Two things learned the hard way against the live endpoint:

    * ``https://www.nseindia.com/`` returns **403 while still setting the
      cookies the API needs**, and the API call then succeeds. Calling
      ``raise_for_status()`` on the warm-up turns a working flow into a hard
      failure, so the warm-up deliberately ignores the status code.
    * ``curl_cffi`` with a Chrome TLS fingerprint gets a clean 200 on both the
      root and the API, so it is preferred when installed; plain ``requests``
      is the fallback.
    """
    global _nse_sess, _nse_kind
    if _nse_sess is not None and not force_new:
        return _nse_sess

    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{config.NSE_BASE}/",
        "Connection": "keep-alive",
    }

    try:
        from curl_cffi import requests as cr
        s = cr.Session(impersonate="chrome")
        s.headers.update(headers)
        _nse_kind = "curl_cffi"
    except Exception:
        s = requests.Session()
        s.headers.update(headers)
        _nse_kind = "requests"

    def warm():
        # Status is intentionally not raised on: 403-with-cookies is the
        # normal path for the plain-requests fallback.
        s.get(config.NSE_BASE, timeout=config.HTTP_TIMEOUT)
        if len(s.cookies) == 0:
            raise RuntimeError("NSE warm-up set no cookies")
        return True

    _retry(warm, f"nse warmup ({_nse_kind})")
    _nse_sess = s
    return s


def _nse_json(path: str, label: str) -> Any:
    def go():
        s = nse_session()
        r = s.get(config.NSE_BASE + path, timeout=config.HTTP_TIMEOUT)
        if r.status_code in (401, 403):
            nse_session(force_new=True)               # cookies expired
            raise RuntimeError(f"HTTP {r.status_code}, re-warmed session")
        r.raise_for_status()
        return r.json()
    return _retry(go, label)


_NSE_NUMERIC = [
    "last", "variation", "percentChange", "open", "high", "low", "previousClose",
    "yearHigh", "yearLow", "indicativeClose", "pe", "pb", "dy", "declines",
    "advances", "unchanged", "perChange365d", "perChange30d",
    "oneWeekAgoVal", "oneMonthAgoVal", "oneYearAgoVal", "previousDayVal",
]


def nse_all_indices() -> pd.DataFrame:
    """All ~139 NSE indices with official PE/PB/DY, breadth and 30d/365d returns.

    This is the primary India source: Yahoo's Indian sector indices are ~55%
    missing, and NSE publishes the official valuation numbers here.
    """
    key = cache.make_key("nse_all_indices")

    def loader():
        t0 = time.time()
        payload = _nse_json(config.NSE_ENDPOINTS["all_indices"], "nse allIndices")
        df = pd.DataFrame(payload["data"])
        if df.empty:
            raise FetchError("nse allIndices returned no rows")
        for c in _NSE_NUMERIC:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = payload.get("timestamp")
        df.attrs["elapsed"] = time.time() - t0
        return df

    df, st = cache.cached(key, "nse_indices", loader)
    ts = df["timestamp"].iloc[0] if "timestamp" in df.columns and len(df) else None
    as_of = _parse_nse_ts(ts)
    _record("NSE /api/allIndices", st, len(df), as_of, df.attrs.get("elapsed"),
            f"{df['key'].nunique()} categories" if "key" in df.columns else "")
    return _stamp(df, "NSE /api/allIndices", as_of)


def _parse_nse_ts(ts: Any) -> pd.Timestamp | None:
    if not ts or not isinstance(ts, str):
        return None
    # The derivatives endpoints stamp seconds; allIndices and marketStatus do not.
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(ts, fmt))
        except ValueError:
            continue
    return None


def nse_index_row(name: str, indices: pd.DataFrame | None = None) -> pd.Series | None:
    """One index row by exact name, or None if NSE is not carrying it today."""
    idx = nse_all_indices() if indices is None else indices
    hit = idx[idx["index"] == name]
    return None if hit.empty else hit.iloc[0]


def nse_by_category(category: str, indices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Slice allIndices by its `key` category, e.g. 'SECTORAL INDICES'."""
    idx = nse_all_indices() if indices is None else indices
    return idx[idx["key"] == category].copy()


def nse_market_status() -> pd.DataFrame:
    """Market state and trade date per segment -- used for freshness stamping."""
    key = cache.make_key("nse_market_status")

    def loader():
        payload = _nse_json(config.NSE_ENDPOINTS["market_status"], "nse marketStatus")
        df = pd.DataFrame(payload["marketState"])
        if df.empty:
            raise FetchError("nse marketStatus returned no rows")
        return df

    df, st = cache.cached(key, "nse_status", loader)
    cap = df[df["market"] == "Capital Market"]
    as_of = _parse_nse_ts(cap["tradeDate"].iloc[0]) if not cap.empty else None
    _record("NSE /api/marketStatus", st, len(df), as_of, None,
            cap["marketStatus"].iloc[0] if not cap.empty else "")
    return _stamp(df, "NSE /api/marketStatus", as_of)


def nse_fii_dii() -> pd.DataFrame:
    """FII and DII cash-market activity for the latest completed session.

    NSE exposes only the most recent session here; there is no history API
    (``?date=`` is ignored, ``/api/historical/`` returns 503/404). Use
    ``fii_dii_history()`` for the accumulated series.
    """
    key = cache.make_key("nse_fii_dii")

    def loader():
        payload = _nse_json(config.NSE_ENDPOINTS["fii_dii"], "nse fiidii")
        df = pd.DataFrame(payload)
        if df.empty:
            raise FetchError("nse fiidii returned no rows")
        for c in ("buyValue", "sellValue", "netValue"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        return df

    df, st = cache.cached(key, "nse_flows", loader)
    as_of = df["date"].max() if "date" in df.columns else None
    _record("NSE /api/fiidiiTradeReact", st, len(df), as_of, None,
            "latest session only (no history API)")
    return _stamp(df, "NSE /api/fiidiiTradeReact", as_of)


def fii_dii_history(append_latest: bool = True) -> pd.DataFrame:
    """Accumulated FII/DII history, wide by category.

    Because NSE publishes only one session, this file grows by one row per
    trading day each time the project is run. It starts short and becomes a
    real series with use; notebook 05 reports how many days it holds.
    """
    path = config.FII_DII_HISTORY
    hist = pd.DataFrame()
    if path.exists():
        hist = pd.read_csv(path, parse_dates=["date"])

    if append_latest:
        try:
            latest = nse_fii_dii()
            new = latest[["date", "category", "buyValue", "sellValue", "netValue"]]
            hist = (pd.concat([hist, new], ignore_index=True)
                    .dropna(subset=["date"])
                    .drop_duplicates(subset=["date", "category"], keep="last")
                    .sort_values(["date", "category"]))
            hist.to_csv(path, index=False)
        except FetchError as exc:
            print(f"  [fii_dii_history] could not append latest session: {exc}")

    if hist.empty:
        return hist
    wide = hist.pivot_table(index="date", columns="category",
                            values="netValue", aggfunc="last").sort_index()
    wide.columns = [str(c) for c in wide.columns]
    return _stamp(wide, "fii_dii_history.csv",
                  wide.index.max() if len(wide) else None)


# ==========================================================================
# US Treasury -- full daily yield curve (replaces FRED, which is dead)
# ==========================================================================
def treasury_curve(years_back: int = config.TREASURY_YEARS_BACK) -> pd.DataFrame:
    """Daily par yield curve, 1M to 30Y, indexed by date and ascending in time."""
    this_year = datetime.now().year
    years = list(range(this_year - years_back + 1, this_year + 1))
    key = cache.make_key("treasury_curve", years=years)

    def loader():
        frames = []
        for y in years:
            url = config.TREASURY_CSV.format(year=y)

            def go(u=url, yr=y):
                r = requests.get(u, headers={"User-Agent": config.USER_AGENT},
                                 timeout=config.HTTP_TIMEOUT)
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text))
                if d.empty or "Date" not in d.columns:
                    raise FetchError(f"treasury {yr}: unexpected payload")
                return d

            frames.append(_retry(go, f"treasury {y}"))
        df = pd.concat(frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df.empty:
            raise FetchError("treasury curve is empty after parsing")
        return df

    df, st = cache.cached(key, "treasury", loader)
    df = df.sort_index()
    as_of = df.index.max()
    _record("US Treasury daily curve", st, len(df), as_of, None,
            f"{df.shape[1]} tenors, {years[0]}-{years[-1]}")
    return _stamp(df, "US Treasury daily curve", as_of)


# ==========================================================================
# World Bank -- macro fundamentals (replaces FRED)
# ==========================================================================
def worldbank(indicator: str,
              countries: Iterable[str] | None = None,
              start: int = config.WB_START,
              end: int = config.WB_END) -> pd.DataFrame:
    """One indicator, wide by country, indexed by year."""
    codes = list(countries or config.WB_COUNTRIES)
    key = cache.make_key("worldbank", indicator=indicator, countries=codes,
                         start=start, end=end)

    def loader():
        url = (f"{config.WORLDBANK_BASE}/country/{';'.join(codes)}"
               f"/indicator/{indicator}?format=json&per_page=2000"
               f"&date={start}:{end}")

        def go():
            r = requests.get(url, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise FetchError(f"worldbank {indicator}: no data block")
            return payload

        payload = _retry(go, f"worldbank {indicator}")
        raw = pd.DataFrame(payload[1])
        raw["year"] = pd.to_numeric(raw["date"], errors="coerce")
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw["iso3"] = raw["countryiso3code"].replace("", pd.NA)
        # Aggregates such as EUU come back with an empty iso3 field.
        raw["iso3"] = raw["iso3"].fillna(
            raw["country"].apply(lambda c: c.get("id") if isinstance(c, dict) else None))
        wide = raw.pivot_table(index="year", columns="iso3", values="value",
                               aggfunc="last").sort_index()
        wide.columns = [config.WB_COUNTRIES.get(c, c) for c in wide.columns]
        wide.attrs["last_updated"] = payload[0].get("lastupdated")
        if wide.dropna(how="all").empty:
            raise FetchError(f"worldbank {indicator}: all values null")
        return wide

    df, st = cache.cached(key, "worldbank", loader)
    non_null = df.dropna(how="all")
    as_of = int(non_null.index.max()) if len(non_null) else None
    _record(f"World Bank {indicator}", st, len(df), as_of, None,
            config.WB_INDICATORS.get(indicator, ""))
    return _stamp(df, f"World Bank {indicator}", as_of)


# ==========================================================================
# yfinance -- prices and statements
# ==========================================================================
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars from different exchanges arrive tz-aware in local time.

    Joining Asia/Kolkata against America/New_York is meaningless at the row
    level, so everything is reduced to a naive calendar date.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    return df[~df.index.duplicated(keep="last")]


def price_history(ticker: str,
                  period: str = config.PRICE_HISTORY_PERIOD,
                  interval: str = "1d",
                  quiet: bool = False) -> pd.DataFrame:
    """OHLCV for one ticker, indexed by naive calendar date.

    One ticker per call, deliberately: batching across markets is what causes
    calendar contamination.
    """
    import yfinance as yf

    key = cache.make_key("price", ticker=ticker, period=period, interval=interval)

    def loader():
        def go():
            h = yf.Ticker(ticker).history(period=period, interval=interval,
                                          auto_adjust=False)
            if h is None or h.empty:
                raise FetchError(f"{ticker}: no rows returned")
            return h
        h = _retry(go, f"price {ticker}")
        h = _normalize_index(h)
        keep = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
                if c in h.columns]
        return h[keep]

    t0 = time.time()
    df, st = cache.cached(key, "prices", loader)
    as_of = df.index.max() if len(df) else None
    if not quiet:
        _record(f"yfinance {ticker}", st, len(df), as_of, time.time() - t0, period)
    return _stamp(df, f"yfinance {ticker}", as_of)


def price_histories(tickers: Iterable[str],
                    period: str = config.PRICE_HISTORY_PERIOD,
                    interval: str = "1d",
                    verbose: bool = True) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    """Fetch many tickers one at a time.

    Returns ``(frames, failures)``. Failures are returned, never swallowed --
    a dropped ticker must show up in the notebook, not vanish.
    """
    frames: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for t in tickers:
        try:
            frames[t] = price_history(t, period=period, interval=interval, quiet=True)
        except Exception as exc:                      # noqa: BLE001
            failures.append((t, f"{type(exc).__name__}: {str(exc)[:90]}"))
    if verbose:
        print(f"  fetched {len(frames)}/{len(frames) + len(failures)} tickers "
              f"in {time.time() - t0:.0f}s"
              + (f" | dropped: {[f[0] for f in failures]}" if failures else ""))
    _record("yfinance batch", "live" if frames else "failed", len(frames),
            max((d.index.max() for d in frames.values()), default=None),
            time.time() - t0, f"{len(failures)} failed")
    return frames, failures


def close_frame(frames: dict[str, pd.DataFrame], field: str = "Close") -> pd.DataFrame:
    """Join one field across tickers on the union of dates -- UNALIGNED.

    The result still has real holiday gaps. Pass it through
    ``mkt.align.align_to_calendar`` before computing anything cross-sectional.
    """
    cols = {t: d[field] for t, d in frames.items() if field in d.columns and len(d)}
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.attrs["as_of_by_col"] = {t: s.last_valid_index() for t, s in cols.items()}
    return out


def fx_rate(base: str, quote: str) -> float:
    """Spot FX, e.g. fx_rate('USD', 'INR') -> INR per USD.

    Needed because Yahoo reports some Indian companies' statements in a
    different currency from their share price -- measured on 2026-09-08,
    INFY.NS and HCLTECH.NS carry USD statements against an INR quote. Mixing
    the two without converting inflates PE and PB by the USDINR rate (~95x).
    """
    if base == quote:
        return 1.0
    pair = f"{base}{quote}=X"
    df = price_history(pair, period="5d", quiet=True)
    s = df["Close"].dropna()
    if s.empty:
        raise FetchError(f"no FX rate for {pair}")
    return float(s.iloc[-1])


def fundamentals_bundle(ticker: str) -> dict[str, Any]:
    """Annual + quarterly statements and .info for one company.

    ``.info`` is included for cross-checking only. Ratios are computed from the
    statements (trap #2: ``returnOnEquity`` is silently None for RELIANCE.NS).
    """
    import yfinance as yf

    key = cache.make_key("fundamentals", ticker=ticker)

    def loader():
        def go():
            tk = yf.Ticker(ticker)
            bundle = {
                "income": tk.income_stmt,
                "balance": tk.balance_sheet,
                "cashflow": tk.cashflow,
                "q_income": tk.quarterly_income_stmt,
                "q_balance": tk.quarterly_balance_sheet,
                "q_cashflow": tk.quarterly_cashflow,
            }
            try:
                bundle["info"] = tk.info or {}
            except Exception:
                bundle["info"] = {}
            if all(isinstance(v, pd.DataFrame) and v.empty
                   for k, v in bundle.items() if k != "info"):
                raise FetchError(f"{ticker}: all statements empty")
            return bundle
        return _retry(go, f"fundamentals {ticker}")

    t0 = time.time()
    bundle, st = cache.cached(key, "fundamentals", loader)
    inc = bundle.get("income")
    as_of = (max(inc.columns).date()
             if isinstance(inc, pd.DataFrame) and not inc.empty else None)
    _record(f"yf fundamentals {ticker}", st, None, as_of, time.time() - t0)
    return bundle


def fundamentals_many(tickers: Iterable[str]) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    frames: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for t in tickers:
        try:
            frames[t] = fundamentals_bundle(t)
        except Exception as exc:                      # noqa: BLE001
            failures.append((t, f"{type(exc).__name__}: {str(exc)[:90]}"))
    print(f"  fundamentals: {len(frames)}/{len(frames) + len(failures)} in "
          f"{time.time() - t0:.0f}s"
          + (f" | dropped: {[f[0] for f in failures]}" if failures else ""))
    return frames, failures


# ==========================================================================
# Health check
# ==========================================================================
def _age_days(as_of: Any) -> float | None:
    if as_of is None:
        return None
    try:
        if isinstance(as_of, (int,)):               # World Bank year
            return (datetime.now().year - as_of) * 365.25
        ts = pd.Timestamp(as_of)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return (pd.Timestamp.now().normalize() - ts.normalize()).days
    except Exception:
        return None


def health_check(verbose: bool = True,
                 include_derivatives: bool = False) -> pd.DataFrame:
    """Probe every source and report freshness. Notebook 01 prints this first.

    A source is OK only if it returned data whose as-of date is inside the
    tolerance in ``config.STALENESS_TOLERANCE_DAYS``. Anything else is flagged
    so a quietly stale feed cannot masquerade as live.

    ``include_derivatives`` adds the NSE F&O endpoints. Off by default because
    they fail far more often than the rest of the project's sources, and a red
    row there should not imply the whole chain is broken -- notebook 09 calls
    ``derivatives.health_check()`` for the detailed version.
    """
    checks = [
        ("NSE allIndices", "nse", lambda: nse_all_indices()),
        ("NSE marketStatus", "nse", lambda: nse_market_status()),
        ("NSE FII/DII", "nse", lambda: nse_fii_dii()),
        ("US Treasury curve", "treasury", lambda: treasury_curve()),
        ("World Bank GDP", "worldbank",
         lambda: worldbank("NY.GDP.MKTP.KD.ZG")),
        ("yfinance ^NSEI", "prices", lambda: price_history("^NSEI", period="1mo")),
        ("yfinance ^GSPC", "prices", lambda: price_history("^GSPC", period="1mo")),
        ("yfinance ^VIX", "prices", lambda: price_history("^VIX", period="1mo")),
    ]

    if include_derivatives:
        from . import derivatives as _dv
        checks += [
            ("NSE fo_mktlots.csv", "nse_derivatives",
             lambda: _dv.lot_sizes(["RELIANCE.NS"]).to_frame()),
            ("NSE master-quote", "nse_derivatives",
             lambda: _dv.fno_eligible(["RELIANCE.NS"]).to_frame()),
            ("NSE option-chain-v3", "nse_derivatives",
             lambda: _dv.option_chain("NIFTY", kind="index")),
            ("NSE futures snapshot", "nse_derivatives",
             lambda: _dv.futures_snapshot()),
        ]

    rows = []
    for name, kind, fn in checks:
        t0 = time.time()
        try:
            df = fn()
            as_of = df.attrs.get("as_of")
            age = _age_days(as_of)
            tol = config.STALENESS_TOLERANCE_DAYS[kind]
            ok = age is not None and age <= tol
            rows.append({
                "source": name,
                "status": "OK" if ok else ("STALE" if age is not None else "NO AS-OF"),
                "rows": len(df),
                "as_of": as_of,
                "age_days": age,
                "tolerance_days": tol,
                "secs": round(time.time() - t0, 2),
                "note": "",
            })
        except Exception as exc:                      # noqa: BLE001
            rows.append({
                "source": name, "status": "FAILED", "rows": 0, "as_of": None,
                "age_days": None, "tolerance_days": config.STALENESS_TOLERANCE_DAYS[kind],
                "secs": round(time.time() - t0, 2),
                "note": f"{type(exc).__name__}: {str(exc)[:80]}",
            })

    out = pd.DataFrame(rows)
    if verbose:
        n_ok = (out["status"] == "OK").sum()
        print(f"Source health: {n_ok}/{len(out)} OK  "
              f"(checked {datetime.now():%Y-%m-%d %H:%M})")
    return out
