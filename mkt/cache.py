"""Parquet cache with TTL, keyed by source + params.

Why this exists: the Nifty 50 fundamentals pull costs ~75 seconds of network
time. Paying that on every notebook re-run makes the project unusable. Every
fetcher in `mkt.fetch` goes through `cached()`.

Set ``mkt.cache.REFRESH = True`` at the top of a notebook (or export
``MKT_REFRESH=1``) to force a re-fetch of every source regardless of TTL.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import config

# Module-level switch. Notebooks flip this; fetchers read it every call.
REFRESH: bool = config.REFRESH_DEFAULT

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def set_refresh(value: bool) -> None:
    """Force (or stop forcing) a re-fetch of every cached source."""
    global REFRESH
    REFRESH = bool(value)


def make_key(source: str, **params: Any) -> str:
    """Build a filesystem-safe cache key from a source name and its params."""
    if params:
        blob = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha1(blob.encode()).hexdigest()[:10]
        raw = f"{source}__{digest}"
    else:
        raw = source
    return _SAFE.sub("_", raw)[:120]


def _paths(key: str) -> tuple[Path, Path, Path]:
    base = config.CACHE_DIR / key
    return (
        base.with_suffix(".parquet"),
        base.with_suffix(".pkl"),
        base.with_suffix(".meta.json"),
    )


def ttl_for(kind: str) -> float:
    return config.CACHE_TTL_HOURS.get(kind, config.CACHE_TTL_HOURS["default"])


def read_meta(key: str) -> dict | None:
    _, _, meta_p = _paths(key)
    if not meta_p.exists():
        return None
    try:
        return json.loads(meta_p.read_text())
    except Exception:
        return None


def age_hours(key: str) -> float | None:
    """Hours since this key was last written, or None if never."""
    meta = read_meta(key)
    if not meta or "fetched_at" not in meta:
        return None
    try:
        fetched = datetime.fromisoformat(meta["fetched_at"])
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0


def save(key: str, obj: Any, **meta_extra: Any) -> None:
    """Persist an object. DataFrames go to parquet; anything else to pickle."""
    pq_p, pkl_p, meta_p = _paths(key)
    fmt = "pickle"
    if isinstance(obj, pd.DataFrame):
        try:
            df = obj.copy()
            # Parquet needs flat string column names.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["|".join(str(x) for x in c) for c in df.columns]
            else:
                df.columns = [str(c) for c in df.columns]
            df.to_parquet(pq_p)
            fmt = "parquet"
            pkl_p.unlink(missing_ok=True)
        except Exception:
            # Object columns with mixed types, exotic index types, etc.
            with open(pkl_p, "wb") as fh:
                pickle.dump(obj, fh)
            pq_p.unlink(missing_ok=True)
    else:
        with open(pkl_p, "wb") as fh:
            pickle.dump(obj, fh)
        pq_p.unlink(missing_ok=True)

    meta = {
        "key": key,
        "format": fmt,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        **meta_extra,
    }
    meta_p.write_text(json.dumps(meta, indent=2, default=str))


def load(key: str, max_age_hours: float | None = None) -> Any | None:
    """Return the cached object, or None if missing or older than max_age."""
    pq_p, pkl_p, _ = _paths(key)
    if max_age_hours is not None:
        age = age_hours(key)
        if age is None or age > max_age_hours:
            return None
    try:
        if pq_p.exists():
            return pd.read_parquet(pq_p)
        if pkl_p.exists():
            with open(pkl_p, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        return None
    return None


def cached(key: str, kind: str, loader: Callable[[], Any],
           allow_stale_on_failure: bool = True) -> tuple[Any, str]:
    """Fetch through the cache.

    Returns ``(object, status)`` where status is one of ``cache``, ``live``,
    ``stale-fallback`` or ``failed``. The status is what lets the source health
    table tell the difference between "fresh" and "quietly serving last week".
    """
    ttl = ttl_for(kind)
    if not REFRESH:
        hit = load(key, max_age_hours=ttl)
        if hit is not None:
            return hit, "cache"

    try:
        obj = loader()
    except Exception as exc:
        if allow_stale_on_failure:
            stale = load(key, max_age_hours=None)
            if stale is not None:
                print(f"  [cache] {key}: live fetch failed ({type(exc).__name__}: {exc}); "
                      f"serving cache aged {age_hours(key):.1f}h")
                return stale, "stale-fallback"
        raise

    save(key, obj, kind=kind)
    return obj, "live"


def clear(pattern: str = "*") -> int:
    """Delete cache files matching a glob. Returns the number removed."""
    n = 0
    for p in config.CACHE_DIR.glob(pattern):
        if p.is_file():
            p.unlink()
            n += 1
    return n


def inventory() -> pd.DataFrame:
    """Table of everything currently in the cache, newest first."""
    rows = []
    for meta_p in sorted(config.CACHE_DIR.glob("*.meta.json")):
        meta = json.loads(meta_p.read_text())
        key = meta.get("key", meta_p.stem)
        data_p = meta_p.with_name(meta_p.name.replace(".meta.json", ""))
        size = 0
        for suffix in (".parquet", ".pkl"):
            cand = data_p.with_suffix(suffix)
            if cand.exists():
                size = cand.stat().st_size
        rows.append({
            "key": key,
            "kind": meta.get("kind", ""),
            "format": meta.get("format", ""),
            "age_hours": round(age_hours(key) or float("nan"), 2),
            "size_kb": round(size / 1024, 1),
        })
    if not rows:
        return pd.DataFrame(columns=["key", "kind", "format", "age_hours", "size_kb"])
    return pd.DataFrame(rows).sort_values("age_hours").reset_index(drop=True)


def timed(label: str):
    """Tiny context manager for printing how long a fetch actually took."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            print(f"  [{label}] {time.time() - self.t0:.1f}s")
            return False
    return _T()
