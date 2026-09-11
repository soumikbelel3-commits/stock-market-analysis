"""The chain's shared output: outputs/dashboard.json.

Each notebook's final cell appends its signal here, so the six links of the
chain -- macro, risk, rates, global equity, India, stocks -- can be read in one
place and the later notebooks can consult the earlier ones' conclusions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config

# Run order. `summary()` iterates this, so a section missing from ORDER never
# renders -- and a section in ORDER but absent from the file is skipped, which is
# what lets a partial build (say, 09 unavailable) degrade cleanly.
ORDER = ["01_macro", "02_risk", "03_rates", "04_global_equity",
         "05_india_market", "06_india_companies",
         "07_signal_validation", "08_earnings_quality",
         "09_positioning", "10_portfolio", "11_portfolio_risk",
         # institutional layer
         "12_factor_risk", "13_execution_capacity", "14_governance"]


def _clean(obj: Any) -> Any:
    """Make numpy / pandas values JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 4)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def read() -> dict:
    if not config.DASHBOARD_JSON.exists():
        return {}
    try:
        return json.loads(config.DASHBOARD_JSON.read_text())
    except json.JSONDecodeError:
        return {}


def append(section: str, payload: dict) -> dict:
    """Write one notebook's signal into the dashboard and return the whole file."""
    data = read()
    data[section] = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_clean(payload),
    }
    data["_chain"] = [s for s in ORDER if s in data]
    config.DASHBOARD_JSON.write_text(json.dumps(data, indent=2))
    return data


def summary() -> pd.DataFrame:
    """One row per notebook: its headline signal and when it was written."""
    data = read()
    rows = []
    for key in ORDER:
        if key not in data:
            continue
        blk = data[key]
        rows.append({
            "notebook": key,
            "signal": blk.get("signal", ""),
            "score": blk.get("score", None),
            "as_of": blk.get("as_of", ""),
            "updated_at": blk.get("updated_at", "")[:19].replace("T", " "),
        })
    return pd.DataFrame(rows)


def show(section: str | None = None) -> None:
    data = read()
    blob = data if section is None else data.get(section, {})
    print(json.dumps(blob, indent=2)[:4000])
