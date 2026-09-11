"""Calendar alignment for cross-market frames, plus coverage reporting.

This module exists because of a measured failure. Downloading ``BTC-USD``
(7-day week) alongside equities (5-day week) in a single ``yf.download`` call
forced a calendar-day index and injected ~32% spurious NaNs into every equity
series -- verified again on 2026-09-08: 31.3% on ^GSPC, 32.4% on ^NSEI.

Every cross-market join in this project goes through ``align_to_calendar``, and
returns are computed *after* alignment, never before.

A second, subtler rule is enforced here: forward-filling must never extend a
series past its own last real observation. US equities settle hours after
India closes, so on any given evening ^GSPC is a day behind ^NSEI. Filling that
gap would silently compare a stale close to a fresh one (trap #3).
"""
from __future__ import annotations

from typing import Iterable, Literal

import pandas as pd

from . import config, fetch

CalendarSpec = Literal["B", "NSE", "NYSE", "union", "intersection"] | pd.DatetimeIndex

_CAL_ANCHOR = {"NSE": "^NSEI", "NYSE": "^GSPC"}
_cal_cache: dict[str, pd.DatetimeIndex] = {}


def trading_calendar(kind: CalendarSpec,
                     start: pd.Timestamp | None = None,
                     end: pd.Timestamp | None = None,
                     frame: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """Return the reference calendar to align onto.

    ``NSE`` and ``NYSE`` take their calendar from the index's own trading days,
    which handles holidays correctly without hardcoding a holiday table.
    """
    if isinstance(kind, pd.DatetimeIndex):
        return kind.normalize()

    if kind in _CAL_ANCHOR:
        if kind not in _cal_cache:
            anchor = fetch.price_history(_CAL_ANCHOR[kind],
                                         period=config.LONG_HISTORY_PERIOD,
                                         quiet=True)
            _cal_cache[kind] = pd.DatetimeIndex(anchor.index)
        cal = _cal_cache[kind]
    elif kind == "B":
        lo = start if start is not None else (frame.index.min() if frame is not None else None)
        hi = end if end is not None else (frame.index.max() if frame is not None else None)
        if lo is None or hi is None:
            raise ValueError("calendar 'B' needs start/end or a frame")
        cal = pd.bdate_range(lo, hi)
    elif kind == "union":
        if frame is None:
            raise ValueError("calendar 'union' needs a frame")
        cal = pd.DatetimeIndex(frame.index)
    elif kind == "intersection":
        if frame is None:
            raise ValueError("calendar 'intersection' needs a frame")
        cal = pd.DatetimeIndex(frame.dropna(how="any").index)
    else:
        raise ValueError(f"unknown calendar {kind!r}")

    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    return cal.normalize().unique().sort_values()


def _mask_beyond_last_obs(filled: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Undo any fill that runs past a column's own last real observation."""
    for col in filled.columns:
        if col not in raw.columns:
            continue
        last = raw[col].last_valid_index()
        first = raw[col].first_valid_index()
        if last is not None:
            filled.loc[filled.index > last, col] = pd.NA
        if first is not None:
            filled.loc[filled.index < first, col] = pd.NA
    return filled.astype(float)


def align_to_calendar(df: pd.DataFrame,
                      calendar: CalendarSpec = "B",
                      method: str = "ffill",
                      max_gap: int = 3,
                      start: pd.Timestamp | None = None,
                      end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Reindex a cross-market frame onto one calendar and fill short gaps only.

    Parameters
    ----------
    calendar : 'B', 'NSE', 'NYSE', 'union', 'intersection', or a DatetimeIndex
    method   : 'ffill' or 'none'
    max_gap  : maximum consecutive sessions to carry a stale value forward.
               Longer gaps stay NaN -- a two-week hole is missing data, not a
               flat price.
    """
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    cal = trading_calendar(calendar, start=start, end=end, frame=out)
    cal = cal[(cal >= out.index.min()) & (cal <= out.index.max())]
    if len(cal) == 0:
        raise ValueError("calendar and frame do not overlap")

    # Union first so no real observation is thrown away by an anchor holiday,
    # then reduce to the reference calendar after filling.
    union = out.index.union(cal)
    wide = out.reindex(union)
    if method == "ffill":
        wide = wide.ffill(limit=max_gap)
    elif method != "none":
        raise ValueError(f"unknown method {method!r}")

    wide = _mask_beyond_last_obs(wide, out)
    aligned = wide.reindex(cal)
    aligned.attrs.update(df.attrs)
    aligned.attrs["calendar"] = calendar if isinstance(calendar, str) else "custom"
    aligned.attrs["as_of_by_col"] = {c: out[c].last_valid_index() for c in out.columns}
    return aligned


def coverage_report(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Per-column coverage: window, observations, gaps, and staleness.

    ``lag_days`` is how far each column's last observation sits behind the
    freshest column in the frame -- the direct read on trap #3.
    """
    if df.empty:
        return pd.DataFrame()
    rows = []
    last_overall = max((df[c].last_valid_index() for c in df.columns
                        if df[c].last_valid_index() is not None), default=None)
    for c in df.columns:
        s = df[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            rows.append({"series": c, "first_obs": None, "last_obs": None,
                         "n_obs": 0, "n_slots": len(s), "nan_pct_in_window": 100.0,
                         "max_gap_days": None, "lag_days": None})
            continue
        window = s.loc[first:last]
        obs = window.dropna()
        gap = (obs.index.to_series().diff().dt.days.max()
               if len(obs) > 1 else None)
        rows.append({
            "series": c,
            "first_obs": first.date(),
            "last_obs": last.date(),
            "n_obs": int(window.notna().sum()),
            "n_slots": int(len(window)),
            "nan_pct_in_window": round(float(window.isna().mean() * 100), 2),
            "max_gap_days": None if gap is None else int(gap),
            "lag_days": None if last_overall is None else int((last_overall - last).days),
        })
    out = pd.DataFrame(rows).sort_values("nan_pct_in_window", ascending=False)
    out.attrs["label"] = label
    return out.reset_index(drop=True)


def nan_rate(df: pd.DataFrame) -> float:
    """Overall NaN percentage inside each column's own observed window."""
    if df.empty:
        return 100.0
    total = miss = 0
    for c in df.columns:
        s = df[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            continue
        w = s.loc[first:last]
        total += len(w)
        miss += int(w.isna().sum())
    return 100.0 if total == 0 else round(miss / total * 100, 2)


def common_window(df: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """Trim to the span where enough columns actually have data."""
    if df.empty:
        return df
    ok = df.notna().mean(axis=1) >= min_coverage
    if not ok.any():
        return df
    return df.loc[ok[ok].index.min():ok[ok].index.max()]


def to_returns(df: pd.DataFrame, periods: int = 1, log: bool = False) -> pd.DataFrame:
    """Returns computed *after* alignment, as the module docstring requires."""
    if log:
        import numpy as np
        return np.log(df / df.shift(periods))
    return df.pct_change(periods)


def rebase(df: pd.DataFrame, base: float = 100.0,
           start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rebase every column to `base` at the first date all of them are present."""
    if df.empty:
        return df
    if start is None:
        valid = df.dropna(how="any")
        start = valid.index.min() if len(valid) else df.index.min()
    first = df.loc[start:].ffill().bfill().iloc[0]
    return df.div(first) * base


def assert_alignment(df: pd.DataFrame, max_nan_pct: float = 8.0,
                     label: str = "frame") -> float:
    """Regression guard for trap #1. Raises if the NaN rate blows out.

    Tolerance defaults to 8%: real holiday divergence between the NSE, NYSE and
    European calendars runs ~3-6%. The contaminated frame measured 31-32%.
    """
    rate = nan_rate(df)
    if rate > max_nan_pct:
        raise AssertionError(
            f"{label}: post-alignment NaN rate {rate}% exceeds {max_nan_pct}% "
            f"-- calendar contamination (trap #1) has reappeared.")
    return rate


def align_multi(frames: dict[str, pd.DataFrame],
                field: str = "Close",
                calendar: CalendarSpec = "B",
                max_gap: int = 3) -> pd.DataFrame:
    """Convenience: raw ticker frames -> one aligned field frame."""
    raw = fetch.close_frame(frames, field=field)
    if raw.empty:
        return raw
    return align_to_calendar(raw, calendar=calendar, max_gap=max_gap)


def describe_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column as-of stamps, for printing next to any cross-market table."""
    rows = [{"series": c,
             "as_of": df[c].last_valid_index(),
             "value": df[c].dropna().iloc[-1] if df[c].notna().any() else None}
            for c in df.columns]
    out = pd.DataFrame(rows)
    if len(out) and out["as_of"].notna().any():
        newest = out["as_of"].max()
        out["lag_days"] = (newest - out["as_of"]).dt.days
    return out


def latest_row(df: pd.DataFrame, min_coverage: float = 0.5,
               max_lookback: int = 10) -> pd.Series:
    """The most recent row that actually carries data. Not ``df.iloc[-1]``.

    A trap this module creates and should therefore close. ``align_to_calendar``
    deliberately never fills a series past its own last real observation, so the
    aligned panel legitimately ends with an all-NaN row whenever the calendar has
    opened a session that has not printed yet -- which is most of any trading day,
    and all of a run made before the close.

    ``df.iloc[-1]`` is then a vector of NaN that looks exactly like "the latest
    prices". Everything downstream inherits it silently: a period return comes back
    empty, a scorecard scores nothing, an attribution table renders blank. The
    failure is invisible because nothing raised.

    Returns the last row with at least ``min_coverage`` of columns present, and
    stamps ``.attrs['as_of']`` with the date it came from so a stale row is
    reported rather than assumed.
    """
    if df.empty:
        return pd.Series(dtype=float)
    frac = df.notna().mean(axis=1)
    ok = frac[frac >= min_coverage]
    if ok.empty:
        raise ValueError(
            f"latest_row: no session in the last {len(df)} has {min_coverage:.0%} "
            f"coverage; best was {frac.max():.1%} on {frac.idxmax().date()}")
    stamp = ok.index[-1]
    skipped = int((df.index > stamp).sum())
    if skipped > max_lookback:
        raise ValueError(
            f"latest_row: the most recent usable session is {stamp.date()}, "
            f"{skipped} sessions before the panel's end -- the panel is stale.")
    row = df.loc[stamp].copy()
    row.attrs["as_of"] = stamp
    row.attrs["coverage_%"] = float(frac.loc[stamp] * 100)
    row.attrs["sessions_skipped"] = skipped
    return row


def period_return(df: pd.DataFrame, min_coverage: float = 0.5) -> pd.Series:
    """Total return per column over the panel, using real endpoints at both ends.

    Uses ``latest_row`` at the end and the first adequately-covered session at the
    start, so a partially-printed current session cannot turn a whole attribution
    table into NaN.
    """
    if df.empty:
        return pd.Series(dtype=float)
    last = latest_row(df, min_coverage)
    frac = df.notna().mean(axis=1)
    first_idx = frac[frac >= min_coverage].index
    if first_idx.empty:
        return pd.Series(np.nan, index=df.columns)
    first = df.loc[first_idx[0]]
    out = (last / first - 1).rename("period_return")
    out.attrs["start"] = first_idx[0]
    out.attrs["end"] = last.attrs.get("as_of")
    return out
