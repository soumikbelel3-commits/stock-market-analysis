"""NSE F&O: option chains, PCR, futures basis, rollover, lot sizes.

The most fragile module in the project, and deliberately the last one built.
Every endpoint here is undocumented and NSE changes them without notice, so the
verdicts below are dated measurements rather than assumptions. Probed live on
2026-09-08 against a cookie-warmed ``fetch.nse_session()``:

**Working**

* ``/api/option-chain-v3?type=Equity|Indices&symbol=X&expiry=DD-Mon-YYYY`` -- the
  full chain. ``expiry`` is **mandatory** and must be one of that symbol's own
  expiries; passing an index expiry to an equity symbol returns an empty
  ``data`` list with HTTP 200.
* ``/api/option-chain-contract-info?symbol=X`` -- the expiry list feeding it.
* ``nsearchives.nseindia.com/content/fo/fo_mktlots.csv`` -- lot sizes, 217 rows,
  one column per expiry month. The only working lot-size source.
* ``/api/master-quote`` -- a flat list of ~210 F&O-eligible symbols.
* ``/api/snapshot-derivatives-equity?index=futures&limit=N`` -- futures with open
  interest, but only ~25 underlyings regardless of ``limit``.

**Dead**

* ``/api/option-chain-equities`` returns ``{}`` with HTTP 200 -- the exact failure
  mode this project refuses to tolerate elsewhere, an empty result dressed as
  data. ``/api/option-chain-indices`` and ``/api/quote-derivative`` return 404.
* ``/api/quote-equity?...&section=trade_info`` returns **403** on every attempt,
  with and without a ``/get-quotes/equity`` Referer. **Delivery percentage is
  therefore not obtainable**, and ``delivery_pct`` says so rather than guessing.

Two consequences worth carrying into notebook 09:

1. **Futures coverage is partial.** Basis and rollover exist only for the ~25
   most active underlyings, not the whole Nifty 50. Coverage is reported.
2. **Crowding has no history.** One snapshot gives a level, not a percentile.
   Following the precedent set by ``fii_dii_history.csv`` -- NSE publishes only
   the latest session and there is no history API -- every run appends its
   snapshot to ``data/derivatives_history.csv`` so percentiles become available
   with use. On a fresh clone that file has one row.
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import cache, config, fetch

DERIVATIVES_HISTORY = config.DATA_DIR / "derivatives_history.csv"


def _symbol(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE'. Yahoo suffix off, NSE symbol out."""
    return str(ticker).replace(".NS", "").replace(".BO", "").strip()


# ==========================================================================
# Lot sizes and F&O eligibility -- what notebook 10 actually needs
# ==========================================================================
def _lots_csv() -> pd.DataFrame:
    key = cache.make_key("nse_fo_lots")

    def loader():
        def go():
            s = fetch.nse_session()
            r = s.get(config.NSE_LOTS_CSV, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                raise fetch.FetchError("fo_mktlots.csv returned no rows")
            df.columns = [str(c).strip() for c in df.columns]
            # Every field in this CSV is space-padded to a fixed width
            # ('NIFTY     ', '65         '). Strip unconditionally rather than
            # gating on `dtype == object`: pandas 3.0 types text columns as the
            # new `str` dtype, so an object check silently skips them and every
            # symbol lookup then misses.
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].astype("string").str.strip()
            # The file repeats its header partway through, between the index
            # block and the stock block.
            df = df[df["SYMBOL"].str.upper() != "SYMBOL"]
            return df.reset_index(drop=True)
        return fetch._retry(go, "nse fo_mktlots.csv")

    df, st = cache.cached(key, "nse_derivatives", loader)
    fetch._record("NSE fo_mktlots.csv", st, len(df), None, None,
                  f"{df.shape[1] - 2} expiry columns")
    return df


def lot_sizes(symbols: Iterable[str] | None = None,
              month: str | None = None) -> pd.Series:
    """Lot size per ticker for the near expiry month.

    Feeds the trap #5 guard in ``portfolio.assert_implementable``. Returns a
    Series indexed by whatever was passed in (Yahoo tickers stay Yahoo tickers),
    with NaN for anything not F&O-eligible.
    """
    df = _lots_csv()
    if "SYMBOL" not in df.columns:
        raise fetch.FetchError(f"fo_mktlots.csv: no SYMBOL column, got {list(df.columns)}")

    expiry_cols = [c for c in df.columns if c.upper() not in ("UNDERLYING", "SYMBOL")]
    col = month if month in df.columns else (expiry_cols[0] if expiry_cols else None)
    if col is None:
        raise fetch.FetchError("fo_mktlots.csv: no expiry columns found")

    table = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                      index=df["SYMBOL"].values)
    table = table[~table.index.duplicated(keep="first")]

    if symbols is None:
        out = table.dropna()
        out.name = "lot"
        out.attrs["expiry_column"] = col
        return out

    symbols = list(symbols)
    out = pd.Series({t: table.get(_symbol(t), np.nan) for t in symbols}, name="lot")
    out.attrs["expiry_column"] = col
    out.attrs["n_resolved"] = int(out.notna().sum())
    out.attrs["as_of"] = datetime.now().date().isoformat()
    return out


def fno_eligible(symbols: Iterable[str] | None = None) -> pd.Series:
    """Whether each ticker is F&O-eligible, from ``/api/master-quote``.

    Matters more than it looks: every Nifty 50 constituent qualifies, so the
    short leg is unconstrained today -- but widen the universe to the Nifty 500
    and most names become un-shortable overnight, which is where trap #5 stops
    being about lot arithmetic and starts removing names entirely.
    """
    key = cache.make_key("nse_fno_symbols")

    def loader():
        payload = fetch._nse_json(config.NSE_ENDPOINTS["fno_symbols"], "nse master-quote")
        if not isinstance(payload, list) or not payload:
            raise fetch.FetchError("master-quote returned no symbols")
        return pd.DataFrame({"symbol": payload})

    df, st = cache.cached(key, "nse_derivatives", loader)
    fetch._record("NSE /api/master-quote", st, len(df), None, None,
                  f"{len(df)} F&O-eligible symbols")
    universe = set(df["symbol"].astype(str))
    if symbols is None:
        return pd.Series(True, index=sorted(universe), name="fno_eligible")
    symbols = list(symbols)
    out = pd.Series({t: _symbol(t) in universe for t in symbols}, name="fno_eligible")
    out.attrs["n_eligible"] = int(out.sum())
    out.attrs["universe_size"] = len(universe)
    return out


# ==========================================================================
# Option chain
# ==========================================================================
def option_expiries(symbol: str) -> list[str]:
    """Valid expiry strings for one symbol. The chain endpoint requires one."""
    sym = _symbol(symbol)
    key = cache.make_key("nse_oc_expiries", symbol=sym)

    def loader():
        path = config.NSE_ENDPOINTS["option_chain_expiries"].format(symbol=quote(sym))
        payload = fetch._nse_json(path, f"nse contract-info {sym}")
        dates = (payload or {}).get("expiryDates") or []
        if not dates:
            raise fetch.FetchError(f"contract-info {sym}: no expiry dates")
        return pd.DataFrame({"expiry": dates})

    df, st = cache.cached(key, "nse_derivatives", loader)
    fetch._record(f"NSE contract-info {sym}", st, len(df), None, None,
                  f"{len(df)} expiries")
    return list(df["expiry"])


def option_chain(symbol: str, kind: str = "equity",
                 expiry: str | None = None) -> pd.DataFrame:
    """One symbol's option chain for one expiry, CE and PE side by side.

    ``expiry`` defaults to the nearest. Passing an expiry that does not belong
    to this symbol returns an empty chain with HTTP 200 rather than an error,
    which is why the expiry list is fetched per symbol instead of shared.
    """
    sym = _symbol(symbol)
    typ = "Indices" if kind.lower().startswith(("ind", "idx")) else "Equity"
    exp = expiry or option_expiries(sym)[0]
    key = cache.make_key("nse_option_chain", symbol=sym, kind=typ, expiry=exp)

    def loader():
        path = config.NSE_ENDPOINTS["option_chain"].format(
            kind=typ, symbol=quote(sym), expiry=quote(exp))
        payload = fetch._nse_json(path, f"nse option-chain {sym} {exp}")
        rec = (payload or {}).get("records", {})
        rows = rec.get("data") or []
        if not rows:
            raise fetch.FetchError(
                f"option chain {sym} {exp}: empty data block "
                f"(HTTP 200 with no rows -- check the expiry belongs to this symbol)")
        flat = []
        for r in rows:
            base = {"strike": r.get("strikePrice"), "expiry": r.get("expiryDates")}
            for side in ("CE", "PE"):
                leg = r.get(side) or {}
                for k_src, k_dst in (("openInterest", "oi"),
                                     ("changeinOpenInterest", "oi_chg"),
                                     ("totalTradedVolume", "volume"),
                                     ("impliedVolatility", "iv"),
                                     ("lastPrice", "ltp"),
                                     ("change", "chg")):
                    base[f"{side}_{k_dst}"] = leg.get(k_src)
            flat.append(base)
        df = pd.DataFrame(flat)
        for c in df.columns:
            if c != "expiry":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
        # Carried as columns, not `.attrs`: the parquet cache does not round-trip
        # attrs, so on a cache hit an attrs-only stamp comes back as None. This is
        # the same pattern `fetch.nse_all_indices` uses for its timestamp.
        df["underlying_value"] = pd.to_numeric(rec.get("underlyingValue"),
                                               errors="coerce")
        df["timestamp"] = rec.get("timestamp")
        return df

    df, st = cache.cached(key, "nse_derivatives", loader)
    ts = df["timestamp"].iloc[0] if "timestamp" in df.columns and len(df) else None
    as_of = fetch._parse_nse_ts(ts)
    fetch._record(f"NSE option-chain {sym}", st, len(df), as_of, None, f"expiry {exp}")
    df.attrs["symbol"] = sym
    df.attrs["expiry"] = exp
    df.attrs["underlying_value"] = (float(df["underlying_value"].iloc[0])
                                    if "underlying_value" in df.columns and len(df)
                                    else None)
    df.attrs["timestamp"] = ts
    return fetch._stamp(df, f"NSE option-chain {sym}", as_of)


def pcr(chain: pd.DataFrame) -> dict:
    """Put/call ratio by open interest and by volume.

    High PCR means puts are being bought or written heavily -- conventionally
    read as fear, and at extremes as a contrarian bullish signal. The two
    measures answer different questions: OI is positioning that has accumulated,
    volume is what happened today.
    """
    ce_oi, pe_oi = chain["CE_oi"].sum(), chain["PE_oi"].sum()
    ce_v, pe_v = chain["CE_volume"].sum(), chain["PE_volume"].sum()
    return {
        "pcr_oi": float(pe_oi / ce_oi) if ce_oi else np.nan,
        "pcr_volume": float(pe_v / ce_v) if ce_v else np.nan,
        "total_ce_oi": float(ce_oi), "total_pe_oi": float(pe_oi),
        "total_ce_volume": float(ce_v), "total_pe_volume": float(pe_v),
    }


def max_pain(chain: pd.DataFrame) -> dict:
    """The strike at which the most option value expires worthless.

    Computed properly -- total intrinsic payout to holders at each candidate
    settlement price, minimised -- not the common shortcut of taking the
    highest-OI strike.
    """
    strikes = chain["strike"].dropna().unique()
    ce_oi = chain.set_index("strike")["CE_oi"].fillna(0)
    pe_oi = chain.set_index("strike")["PE_oi"].fillna(0)
    pain = {}
    for s in strikes:
        call_pay = float((np.maximum(s - ce_oi.index.to_numpy(), 0) * ce_oi.to_numpy()).sum())
        put_pay = float((np.maximum(pe_oi.index.to_numpy() - s, 0) * pe_oi.to_numpy()).sum())
        pain[float(s)] = call_pay + put_pay
    if not pain:
        return {"max_pain": np.nan}
    series = pd.Series(pain).sort_index()
    mp = float(series.idxmin())
    spot = chain.attrs.get("underlying_value")
    return {
        "max_pain": mp,
        "spot": spot,
        "distance_%": (float((mp / spot - 1) * 100)
                       if spot and np.isfinite(spot) and spot > 0 else np.nan),
        "pain_curve": series,
    }


def iv_skew(chain: pd.DataFrame, width_pct: float = 5.0) -> dict:
    """Put IV minus call IV at roughly equidistant out-of-the-money strikes.

    Positive skew means downside protection costs more than upside -- the normal
    state for equities, and its *size* is the read on how nervous positioning is.
    """
    spot = chain.attrs.get("underlying_value")
    if not spot or not np.isfinite(spot):
        return {"skew": np.nan}
    lo, hi = spot * (1 - width_pct / 100), spot * (1 + width_pct / 100)
    put_leg = chain.iloc[(chain["strike"] - lo).abs().argsort()[:1]]
    call_leg = chain.iloc[(chain["strike"] - hi).abs().argsort()[:1]]
    put_iv = float(put_leg["PE_iv"].iloc[0]) if len(put_leg) else np.nan
    call_iv = float(call_leg["CE_iv"].iloc[0]) if len(call_leg) else np.nan
    atm = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]]
    atm_iv = float(np.nanmean([atm["CE_iv"].iloc[0], atm["PE_iv"].iloc[0]])) if len(atm) else np.nan
    return {
        "spot": float(spot),
        "put_strike": float(put_leg["strike"].iloc[0]) if len(put_leg) else np.nan,
        "call_strike": float(call_leg["strike"].iloc[0]) if len(call_leg) else np.nan,
        "put_iv": put_iv, "call_iv": call_iv, "atm_iv": atm_iv,
        "skew": (put_iv - call_iv) if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan,
    }


def oi_buildup(chain: pd.DataFrame, price_change_pct: float) -> dict:
    """Classify positioning from the joint move in price and open interest.

    The standard four-way read. Rising OI means new positions; falling OI means
    existing ones closing. Combined with the price direction that distinguishes
    fresh conviction from a squeeze:

    ===================  ==================  ====================
    price                open interest       reading
    ===================  ==================  ====================
    up                   up                  long buildup
    down                 up                  short buildup
    up                   down                short covering
    down                 down                long unwinding
    ===================  ==================  ====================
    """
    ce_chg, pe_chg = chain["CE_oi_chg"].sum(), chain["PE_oi_chg"].sum()
    net = float(ce_chg + pe_chg)
    up = price_change_pct > 0
    rising = net > 0
    label = ("long buildup" if up and rising else
             "short buildup" if (not up) and rising else
             "short covering" if up and (not rising) else "long unwinding")
    return {"ce_oi_change": float(ce_chg), "pe_oi_change": float(pe_chg),
            "net_oi_change": net, "price_change_%": float(price_change_pct),
            "buildup": label}


# ==========================================================================
# Futures
# ==========================================================================
def futures_snapshot() -> pd.DataFrame:
    """All stock and index futures NSE is currently publishing.

    Coverage is roughly 25 underlyings regardless of the ``limit`` parameter --
    measured, not assumed. Basis and rollover are available only for these.
    """
    key = cache.make_key("nse_futures_snapshot")

    def loader():
        payload = fetch._nse_json(config.NSE_ENDPOINTS["derivatives_snapshot"],
                                  "nse futures snapshot")
        block = (payload or {}).get("volume") or {}
        rows = block.get("data") or []
        if not rows:
            raise fetch.FetchError("futures snapshot returned no rows")
        df = pd.DataFrame(rows)
        for c in ("lastPrice", "openInterest", "numberOfContractsTraded",
                  "totalTurnover", "underlyingValue", "pChange"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["expiry_dt"] = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y",
                                         errors="coerce")
        df["timestamp"] = block.get("timestamp")
        return df

    df, st = cache.cached(key, "nse_derivatives", loader)
    as_of = fetch._parse_nse_ts(df["timestamp"].iloc[0] if len(df) else None)
    fetch._record("NSE futures snapshot", st, len(df), as_of, None,
                  f"{df['underlying'].nunique()} underlyings")
    return fetch._stamp(df, "NSE futures snapshot", as_of)


def futures_basis(symbol: str, snapshot: pd.DataFrame | None = None) -> dict:
    """Futures premium over spot, and the carry it annualises to.

    A rich basis means longs are paying up to hold the position -- crowded, or
    at least expensive. A discount means the opposite. Annualising makes names
    with different days-to-expiry comparable.
    """
    snap = futures_snapshot() if snapshot is None else snapshot
    sym = _symbol(symbol)
    rows = snap[(snap["underlying"] == sym) & (snap["instrumentType"] == "FUTSTK")]
    if rows.empty:
        return {"symbol": sym, "available": False,
                "note": "not among the underlyings NSE publishes futures for"}
    rows = rows.sort_values("expiry_dt")
    near = rows.iloc[0]
    spot = float(near.get("underlyingValue", np.nan))
    fut = float(near.get("lastPrice", np.nan))
    days = max((near["expiry_dt"] - pd.Timestamp.now().normalize()).days, 1)
    basis = fut - spot if np.isfinite(fut) and np.isfinite(spot) else np.nan
    basis_pct = basis / spot * 100 if np.isfinite(basis) and spot else np.nan
    out = {
        "symbol": sym, "available": True, "spot": spot, "near_future": fut,
        "expiry": near["expiryDate"], "days_to_expiry": int(days),
        "basis": basis, "basis_%": basis_pct,
        "annualised_carry_%": (basis_pct * 365 / days
                               if np.isfinite(basis_pct) else np.nan),
        "near_oi": float(near.get("openInterest", np.nan)),
    }
    if len(rows) > 1:
        nxt = rows.iloc[1]
        out["next_future"] = float(nxt.get("lastPrice", np.nan))
        out["next_oi"] = float(nxt.get("openInterest", np.nan))
        out["next_expiry"] = nxt["expiryDate"]
    return out


def rollover(symbol: str, snapshot: pd.DataFrame | None = None) -> dict:
    """Share of open interest that has moved to the next expiry.

    **An approximation, not the exchange's figure.** NSE's official rollover
    percentage is computed over the expiry-day session and is not exposed in any
    public JSON endpoint. This is next-month OI as a share of total OI across
    listed expiries -- directionally the same quantity, and it drifts upward
    through the month by construction, so it is only comparable against the same
    point in a previous cycle.
    """
    snap = futures_snapshot() if snapshot is None else snapshot
    sym = _symbol(symbol)
    rows = snap[(snap["underlying"] == sym) & (snap["instrumentType"] == "FUTSTK")]
    if rows.empty or rows["openInterest"].notna().sum() < 2:
        return {"symbol": sym, "available": False,
                "note": "fewer than two expiries published for this underlying"}
    rows = rows.sort_values("expiry_dt")
    oi = rows["openInterest"].fillna(0.0)
    total = float(oi.sum())
    near_days = max((rows.iloc[0]["expiry_dt"] - pd.Timestamp.now().normalize()).days, 0)
    return {
        "symbol": sym, "available": True,
        "near_oi": float(oi.iloc[0]), "far_oi": float(oi.iloc[1:].sum()),
        "total_oi": total,
        "rollover_%": float(oi.iloc[1:].sum() / total * 100) if total else np.nan,
        "days_to_near_expiry": int(near_days),
        "method": "next-expiry OI share -- approximation, not NSE's official figure",
    }


def delivery_pct(symbol: str) -> dict:
    """Delivery percentage. **Not obtainable** -- the endpoint returns 403.

    Kept as a function so the failure is explicit and appears in
    ``fetch.status_log()``, rather than the metric quietly vanishing from
    notebook 09 with no explanation of why.
    """
    sym = _symbol(symbol)
    path = config.NSE_ENDPOINTS["quote_trade_info"].format(symbol=quote(sym))
    try:
        payload = fetch._nse_json(path, f"nse trade_info {sym}")
        block = (payload or {}).get("securityWiseDP") or {}
        val = pd.to_numeric(block.get("deliveryToTradedQuantity"), errors="coerce")
        fetch._record(f"NSE trade_info {sym}", "live", 1, None, None, "delivery %")
        return {"symbol": sym, "available": bool(np.isfinite(val)),
                "delivery_%": float(val) if np.isfinite(val) else np.nan}
    except Exception as exc:                              # noqa: BLE001
        fetch._record(f"NSE trade_info {sym}", "failed", 0, None, None,
                      f"{type(exc).__name__}: {str(exc)[:60]}")
        return {"symbol": sym, "available": False, "delivery_%": np.nan,
                "note": ("quote-equity?section=trade_info returns 403 -- measured "
                         "2026-09-08 with and without a get-quotes Referer")}


# ==========================================================================
# Crowding
# ==========================================================================
def crowding_row(symbol: str, chain: pd.DataFrame | None = None,
                 snapshot: pd.DataFrame | None = None,
                 price_change_pct: float = 0.0) -> dict:
    """Everything crowding-related for one name, in one row."""
    sym = _symbol(symbol)
    row: dict = {"symbol": sym}
    if chain is not None and len(chain):
        row.update(pcr(chain))
        mp = max_pain(chain)
        row["max_pain"] = mp.get("max_pain")
        row["max_pain_distance_%"] = mp.get("distance_%")
        row.update({k: v for k, v in iv_skew(chain).items()
                    if k in ("atm_iv", "skew", "put_iv", "call_iv")})
        row.update({k: v for k, v in oi_buildup(chain, price_change_pct).items()
                    if k in ("net_oi_change", "buildup")})
    fb = futures_basis(sym, snapshot)
    if fb.get("available"):
        row["basis_%"] = fb.get("basis_%")
        row["annualised_carry_%"] = fb.get("annualised_carry_%")
        row["futures_oi"] = fb.get("near_oi")
    ro = rollover(sym, snapshot)
    if ro.get("available"):
        row["rollover_%"] = ro.get("rollover_%")
    return row


def crowding_score(table: pd.DataFrame) -> pd.Series:
    """A relative 0-100 crowding read across whatever names resolved.

    Deliberately **relative, not absolute**. With a single snapshot and no OI
    history there is no percentile to place a name in, so this ranks the names
    against each other on the day and nothing more. ``append_history`` exists to
    remove that limitation over time.

    Higher = more crowded long: a rich futures basis (longs paying up to hold the
    position) and a low PCR (little put protection held against it).

    ``net_oi_change`` is deliberately **not** an input. It measures how much new
    option open interest accumulated, which is a magnitude and not a direction --
    a large build can be new longs or new shorts. It is reported as a column
    because it is informative next to the buildup label, but scoring it as if
    'more positioning' meant 'more crowded long' would be wrong.
    """
    parts, weights = [], []
    for col, direction in (("basis_%", 1), ("annualised_carry_%", 1),
                           ("pcr_oi", -1)):
        if col not in table.columns:
            continue
        s = pd.to_numeric(table[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        parts.append(s.rank(pct=True, ascending=(direction > 0)) * 100)
        weights.append(s.notna().astype(float))
    if not parts:
        return pd.Series(np.nan, index=table.index, name="crowding_score")
    stacked = pd.concat(parts, axis=1)
    avail = pd.concat(weights, axis=1)
    out = (stacked.fillna(0) * avail).sum(axis=1) / avail.sum(axis=1).replace(0, np.nan)
    return out.rename("crowding_score")


def append_history(table: pd.DataFrame, path=None) -> pd.DataFrame:
    """Accumulate daily snapshots so crowding gains a history.

    Same reasoning as ``fii_dii_history.csv``: NSE publishes only the current
    state and there is no history API, so a percentile can only be built by
    saving what each run sees. On a fresh clone this file has one row; it becomes
    a real series with use.

    Deliberately outside ``data/cache/`` -- clearing the cache is safe, deleting
    this loses history that cannot be reconstructed.
    """
    path = DERIVATIVES_HISTORY if path is None else path
    snap = table.copy()
    snap.insert(0, "date", pd.Timestamp.now().normalize())
    hist = pd.DataFrame()
    if path.exists():
        try:
            hist = pd.read_csv(path, parse_dates=["date"])
        except Exception as exc:                          # noqa: BLE001
            print(f"  [derivatives] could not read {path}: {exc}")
    out = (pd.concat([hist, snap], ignore_index=True)
           .drop_duplicates(subset=["date", "symbol"], keep="last")
           .sort_values(["date", "symbol"]))
    out.to_csv(path, index=False)
    out.attrs["sessions"] = int(out["date"].nunique())
    out.attrs["path"] = str(path)
    return out


# ==========================================================================
# Health
# ==========================================================================
def health_check(sample_symbol: str = "RELIANCE", verbose: bool = True) -> pd.DataFrame:
    """Probe every derivatives source. Notebook 09 prints this and degrades visibly.

    Separate from ``fetch.health_check`` because these endpoints fail far more
    often than the rest of the project's sources, and a red row here should not
    imply the whole chain is broken.
    """
    def _delivery_probe():
        got = delivery_pct(sample_symbol)
        if not got["available"]:
            raise RuntimeError("HTTP 403 -- measured 2026-09-08, not obtainable")
        return pd.Series([got["delivery_%"]])

    checks = [
        ("NSE fo_mktlots.csv (lot sizes)", lambda: lot_sizes(["RELIANCE.NS"])),
        ("NSE master-quote (F&O eligibility)", lambda: fno_eligible(["RELIANCE.NS"])),
        ("NSE contract-info (expiries)", lambda: pd.Series(option_expiries(sample_symbol))),
        ("NSE option-chain-v3 (equity)", lambda: option_chain(sample_symbol)),
        ("NSE option-chain-v3 (index)", lambda: option_chain("NIFTY", kind="index")),
        ("NSE futures snapshot", lambda: futures_snapshot()),
        ("NSE quote-equity trade_info (delivery %)", _delivery_probe),
    ]
    rows = []
    for name, fn in checks:
        t0 = time.time()
        try:
            obj = fn()
            n = len(obj) if hasattr(obj, "__len__") else 1
            ok = n > 0
            rows.append({"source": name, "status": "OK" if ok else "EMPTY", "rows": n,
                         "secs": round(time.time() - t0, 2), "note": ""})
        except Exception as exc:                          # noqa: BLE001
            rows.append({"source": name, "status": "FAILED", "rows": 0,
                         "secs": round(time.time() - t0, 2),
                         "note": f"{type(exc).__name__}: {str(exc)[:70]}"})
    out = pd.DataFrame(rows)
    if verbose:
        n_ok = int((out["status"] == "OK").sum())
        print(f"Derivatives source health: {n_ok}/{len(out)} OK  "
              f"(checked {datetime.now():%Y-%m-%d %H:%M})")
    return out
