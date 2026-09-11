"""Run manifest: what was run, on what data, with which assumptions.

Why this module exists
----------------------
Every number in this project is derived from live data, which is the chain's best
feature and its one reproducibility problem. Re-run it tomorrow and the scores
move -- that is the design. But it means that a table printed last month cannot be
reconstructed, defended, or debugged unless something recorded the conditions it
was produced under.

The manifest is that record. It captures, on every run:

* **Environment** -- Python and library versions. A pandas minor release has
  changed groupby semantics before and will again.
* **Data lineage** -- every cache file the run could have used, with its size,
  age and a content hash. The hash is what makes "the numbers changed" separable
  into "the data changed" and "the code changed", which are very different
  conversations.
* **Configuration fingerprint** -- a hash over the parameters that actually move
  results: weights, caps, universe, cost rates, limits. Two runs with the same
  fingerprint were asked the same question.
* **Chain state** -- which notebooks have published to ``dashboard.json`` and how
  stale each one is. A portfolio built on a risk regime computed three weeks ago
  is a different object from one built this morning, and only the manifest shows
  the difference.

This is the cheapest institutional control there is, and its absence is what
makes most research code unauditable six months later.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import config, dashboard

# Parameters that change results. Anything not on this list is presentation.
FINGERPRINT_KEYS = [
    "SCORE_WEIGHTS", "FUNDAMENTAL_SPEC_HASH", "TECHNICAL_SPEC_HASH",
    "TOP_N_DEEP_DIVE", "MAX_PER_SECTOR", "BACKTEST_PERIOD", "REBALANCE_FREQ",
    "IC_HORIZONS", "N_QUANTILES", "COST_BPS", "TARGET_PORTFOLIO_VOL_PCT",
    "MAX_POSITION_WEIGHT", "MAX_SECTOR_WEIGHT", "GROSS_BY_REGIME",
    "MAX_PCT_OF_ADV", "DEFAULT_CAPITAL", "VAR_CONFIDENCE", "COV_LOOKBACK_DAYS",
    "RISK_FREE_ANNUAL_PCT", "IMPACT_COEF", "FDR_ALPHA", "SPAN_SCAN_RANGE_SIGMA",
    "ELM_PCT", "MARGIN_UTILISATION_LIMIT", "FACTOR_HALFLIFE_DAYS", "RANDOM_SEED",
]


def _stable_hash(obj) -> str:
    """Content hash that does not depend on dict ordering or numpy types."""
    def norm(o):
        if isinstance(o, dict):
            return {str(k): norm(o[k]) for k in sorted(o, key=str)}
        if isinstance(o, (list, tuple, set)):
            return [norm(v) for v in (sorted(o, key=str) if isinstance(o, set) else o)]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return round(float(o), 10)
        if isinstance(o, Path):
            return str(o)
        return o
    blob = json.dumps(norm(obj), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def environment() -> dict:
    """Interpreter and library versions."""
    mods = {}
    for name in ("pandas", "numpy", "scipy", "matplotlib", "yfinance",
                 "pyarrow", "requests"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:                                    # noqa: BLE001
            mods[name] = "not installed"
    from . import __version__
    return {
        "mkt_version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "packages": mods,
    }


def config_fingerprint() -> dict:
    """Hash of the parameters that move results, plus the values themselves.

    Both, deliberately. The hash makes two runs comparable at a glance; the values
    make a difference explicable without checking out the old code.
    """
    from . import score
    values: dict = {}
    for key in FINGERPRINT_KEYS:
        if key == "FUNDAMENTAL_SPEC_HASH":
            values[key] = _stable_hash(score.FUNDAMENTAL_SPEC)
        elif key == "TECHNICAL_SPEC_HASH":
            values[key] = _stable_hash(score.TECHNICAL_SPEC)
        elif hasattr(config, key):
            v = getattr(config, key)
            values[key] = list(v) if isinstance(v, tuple) else v
    return {
        "fingerprint": _stable_hash(values),
        "universe_size": len(config.NIFTY_50),
        "universe_hash": _stable_hash(sorted(config.NIFTY_50)),
        "values": values,
    }


def data_lineage(cache_dir=None, max_files: int = 500) -> dict:
    """Every cache artefact, with age and content hash.

    The age column is the one that gets read. A fundamentals file inside its
    one-week TTL is fine; the same file at nineteen days means a fetch failed and
    the fallback served stale data -- which ``fetch`` reports at the time and
    nobody remembers by the time the output is questioned.
    """
    cache_dir = config.CACHE_DIR if cache_dir is None else Path(cache_dir)
    now = datetime.now(timezone.utc)
    rows = []
    if cache_dir.exists():
        for p in sorted(cache_dir.glob("*"))[:max_files]:
            if p.suffix not in (".parquet", ".json"):
                continue
            st = p.stat()
            age_h = (now.timestamp() - st.st_mtime) / 3600.0
            rows.append({
                "file": p.name,
                "bytes": st.st_size,
                "age_hours": round(age_h, 2),
                "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                if st.st_size < 20_000_000 else "skipped (large)",
            })
    df = pd.DataFrame(rows)
    # The content hash covers name, size and content ONLY -- never age. Age ticks
    # every second, so including it would give two runs on identical data
    # different hashes, which is precisely the comparison this exists to support.
    stable = (df[["file", "bytes", "sha256_16"]].to_dict(orient="records")
              if len(df) else [])
    return {
        "cache_dir": str(cache_dir),
        "n_files": len(df),
        "total_bytes": int(df["bytes"].sum()) if len(df) else 0,
        "oldest_hours": float(df["age_hours"].max()) if len(df) else None,
        "newest_hours": float(df["age_hours"].min()) if len(df) else None,
        "content_hash": _stable_hash(stable) if len(df) else "",
        "files": df.to_dict(orient="records"),
    }


def chain_state() -> dict:
    """Which notebooks have published, and how stale each one's signal is.

    A book sized on notebook 02's risk regime is only as current as that regime.
    ``max_stale_hours`` is the number to look at: it is the age of the oldest link
    the current output depends on.
    """
    data = dashboard.read()
    now = datetime.now(timezone.utc)
    rows = []
    for key in dashboard.ORDER:
        blk = data.get(key)
        if not isinstance(blk, dict):
            rows.append({"notebook": key, "present": False, "age_hours": None,
                         "signal": ""})
            continue
        ts = blk.get("updated_at", "")
        age = None
        try:
            age = round((now - datetime.fromisoformat(ts)).total_seconds() / 3600.0, 2)
        except Exception:                                    # noqa: BLE001
            pass
        rows.append({"notebook": key, "present": True, "age_hours": age,
                     "signal": str(blk.get("signal", ""))[:60]})
    df = pd.DataFrame(rows)
    ages = pd.to_numeric(df["age_hours"], errors="coerce").dropna()
    return {
        "notebooks_present": int(df["present"].sum()),
        "notebooks_expected": len(dashboard.ORDER),
        "complete": bool(df["present"].all()),
        "max_stale_hours": float(ages.max()) if len(ages) else None,
        "min_stale_hours": float(ages.min()) if len(ages) else None,
        "links": df.to_dict(orient="records"),
    }


def build(extra: Mapping | None = None, path=None, write: bool = True) -> dict:
    """Assemble and write the manifest. Call once at the end of a chain run.

    ``run_id`` is a hash over the environment, the configuration and the data --
    not over the clock. Two runs on the same data with the same code get the same
    run id, which is what makes it useful for deduplication; the wall-clock time is
    recorded separately.
    """
    env = environment()
    cfg = config_fingerprint()
    lineage = data_lineage()
    chain = chain_state()

    run_id = _stable_hash({
        "env": env["packages"], "python": env["python"],
        "config": cfg["fingerprint"], "data": lineage["content_hash"],
    })

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
        "config": cfg,
        "data_lineage": lineage,
        "chain_state": chain,
        "seed": config.RANDOM_SEED,
        "rates_as_of": {
            "risk_free": config.RISK_FREE_ASOF,
            "transaction_costs": config.COST_ASOF,
            "lot_sizes": config.LOT_SIZE_FALLBACK_ASOF,
        },
    }
    if extra:
        manifest["run"] = dict(extra)

    if write:
        path = config.RUN_MANIFEST if path is None else Path(path)
        path.write_text(json.dumps(manifest, indent=2, default=str))
        manifest["path"] = str(path)
    return manifest


def read(path=None) -> dict:
    path = config.RUN_MANIFEST if path is None else Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def summary(m: Mapping | None = None) -> pd.DataFrame:
    """One-screen view of a manifest. What a reviewer reads first."""
    m = m or read()
    if not m:
        return pd.DataFrame()
    env, cfg = m.get("environment", {}), m.get("config", {})
    lin, ch = m.get("data_lineage", {}), m.get("chain_state", {})
    rows = [
        ("run_id", m.get("run_id", "")),
        ("generated_at", m.get("generated_at", "")),
        ("mkt version", env.get("mkt_version", "")),
        ("python", env.get("python", "")),
        ("pandas / numpy / scipy",
         f"{env.get('packages', {}).get('pandas', '')} / "
         f"{env.get('packages', {}).get('numpy', '')} / "
         f"{env.get('packages', {}).get('scipy', '')}"),
        ("config fingerprint", cfg.get("fingerprint", "")),
        ("universe", f"{cfg.get('universe_size', '')} names "
                     f"(hash {cfg.get('universe_hash', '')})"),
        ("cache files", f"{lin.get('n_files', 0)} files, "
                        f"{(lin.get('total_bytes', 0) or 0) / 1e6:.1f} MB"),
        ("oldest cache entry", f"{lin.get('oldest_hours', 0) or 0:.1f} h"),
        ("chain links published",
         f"{ch.get('notebooks_present', 0)}/{ch.get('notebooks_expected', 0)}"),
        ("oldest chain signal",
         f"{ch.get('max_stale_hours') or 0:.1f} h" if ch.get("max_stale_hours") is not None else "n/a"),
        ("risk-free as of", m.get("rates_as_of", {}).get("risk_free", "")),
        ("cost rates as of", m.get("rates_as_of", {}).get("transaction_costs", "")),
        ("lot sizes as of", m.get("rates_as_of", {}).get("lot_sizes", "")),
    ]
    return pd.DataFrame(rows, columns=["item", "value"]).set_index("item")


def compare(a: Mapping, b: Mapping) -> pd.DataFrame:
    """Diff two manifests. Answers: why did the numbers change?

    The first row is the one that matters. When ``config`` matches and ``data``
    does not, the input moved and the code did not -- which is the normal case and
    needs no investigation. When ``config`` differs, someone changed an assumption,
    and the ``values`` block says which.
    """
    rows = []
    for label, path in (("run_id", ("run_id",)),
                        ("config fingerprint", ("config", "fingerprint")),
                        ("data content hash", ("data_lineage", "content_hash")),
                        ("universe hash", ("config", "universe_hash")),
                        ("mkt version", ("environment", "mkt_version")),
                        ("python", ("environment", "python"))):
        va, vb = a, b
        for k in path:
            va = (va or {}).get(k) if isinstance(va, dict) else None
            vb = (vb or {}).get(k) if isinstance(vb, dict) else None
        rows.append({"item": label, "a": va, "b": vb, "same": va == vb})

    va = (a.get("config", {}) or {}).get("values", {})
    vb = (b.get("config", {}) or {}).get("values", {})
    for key in sorted(set(va) | set(vb)):
        if va.get(key) != vb.get(key):
            rows.append({"item": f"config.{key}", "a": va.get(key),
                         "b": vb.get(key), "same": False})

    out = pd.DataFrame(rows).set_index("item")
    out.attrs["identical"] = bool(out["same"].all())
    return out
