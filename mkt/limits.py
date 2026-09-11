"""Risk limits: the register, the monitor, and the breach log.

Why this module exists
----------------------
The chain enforced constraints in four places -- ``apply_caps`` clipped weights,
``vol_target`` refused to lever, ``assert_implementable`` refused an untradeable
short, ``assert_var_agreement`` refused an inconsistent covariance. All four are
good, and all four are *scattered*, *implicit*, and *invisible until they fire*.

An institutional book works the other way round. There is one register of limits,
agreed before the book is built rather than discovered while building it; every
limit is measured on every run whether it binds or not; and every breach is
written to a dated log that outlives the session. The difference is not
bureaucracy -- it is that the second arrangement can answer "were we inside our
limits on 14 March" and the first cannot.

Hard versus soft
----------------
A **hard** limit stops the book: ``assert_within_limits`` raises. A **soft** limit
is a conversation: it is recorded, reported and escalated, but a book that is
concentrated because the opportunity set is concentrated should not be
mechanically forbidden. Which is which lives in ``config.RISK_LIMITS``, in one
place, where it can be argued about.

The register is deliberately data, not code. Changing a limit should be a diff in
a config list that a reviewer can read in ten seconds, not an edit inside a
function.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config

OPERATORS = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: v == t,
}


# ==========================================================================
# Gathering the metrics
# ==========================================================================
def book_metrics(book: pd.DataFrame,
                 sectors: Mapping[str, str] | pd.Series | None = None,
                 cov: pd.DataFrame | None = None,
                 var_table: pd.DataFrame | None = None,
                 risk_contrib: pd.DataFrame | None = None,
                 concentration: Mapping | None = None,
                 adequacy: Mapping | None = None,
                 stress: pd.DataFrame | None = None,
                 liquidity: pd.DataFrame | None = None) -> dict:
    """Collect every metric the register refers to, from whatever is available.

    Deliberately tolerant: each input is optional and a missing one leaves its
    metrics absent rather than zero. ``monitor`` then reports those limits as
    ``not measured``, which is honest -- a limit that could not be evaluated is
    not a limit that passed, and conflating the two is how a monitoring system
    comes to report all-green while measuring nothing.
    """
    w = book["weight"] if "weight" in book.columns else book.iloc[:, 0]
    w = pd.to_numeric(w, errors="coerce").dropna()
    m: dict[str, float] = {
        "gross_exposure": float(w.abs().sum()),
        "net_exposure": float(w.sum()),
        "net_exposure_abs": float(abs(w.sum())),
        "n_positions": float((w != 0).sum()),
        "n_long": float((w > 0).sum()),
        "n_short": float((w < 0).sum()),
        "max_position_weight": float(w.abs().max()) if len(w) else np.nan,
    }
    tot = w.abs().sum()
    m["effective_positions"] = (float(1.0 / ((w.abs() / tot) ** 2).sum())
                                if tot > 0 else np.nan)

    if "beta_contrib" in book.columns:
        m["net_beta"] = float(pd.to_numeric(book["beta_contrib"], errors="coerce").sum())
        m["net_beta_abs"] = abs(m["net_beta"])

    if sectors is not None:
        sec = pd.Series(sectors) if not isinstance(sectors, pd.Series) else sectors
        by_sec = w.abs().groupby(sec.reindex(w.index).fillna("?")).sum()
        m["max_sector_weight"] = float(by_sec.max()) if len(by_sec) else np.nan
        m["n_sectors"] = float(len(by_sec))

    if cov is not None:
        from . import portfolio as pf
        m["ex_ante_vol_%"] = float(pf.portfolio_vol(w, cov))

    if var_table is not None and len(var_table):
        for c, key in (("historical_%", "var"), ("cvar_%", "cvar")):
            if c in var_table.columns:
                for conf in var_table.index:
                    tag = f"{key}_{int(round(float(conf) * 100))}_1d_%"
                    m[tag] = float(var_table.loc[conf, c])

    if risk_contrib is not None and len(risk_contrib) and "pct_of_risk" in risk_contrib.columns:
        m["pct_risk_top_name"] = float(risk_contrib["pct_of_risk"].max())
        m["pct_risk_top3"] = float(risk_contrib["pct_of_risk"].nlargest(3).sum())

    if concentration:
        for k in ("effective_bets_entropy", "pc1_variance_%", "avg_correlation"):
            if k in concentration:
                m[k] = float(concentration[k])
        if "effective_bets_entropy" in concentration:
            m["effective_bets"] = float(concentration["effective_bets_entropy"])

    if adequacy:
        for k in ("margin_utilisation", "free_cash_%", "leverage_on_capital"):
            if k in adequacy and adequacy[k] is not None:
                m[k] = float(adequacy[k])

    if stress is not None and len(stress) and "book_return_%" in stress.columns:
        s = pd.to_numeric(stress["book_return_%"], errors="coerce").dropna()
        if len(s):
            m["worst_stress_loss_%"] = float(s.min())

    if liquidity is not None and len(liquidity):
        col = ("days_to_liquidate" if "days_to_liquidate" in liquidity.columns
               else "days_to_trade" if "days_to_trade" in liquidity.columns else None)
        if col:
            d = pd.to_numeric(liquidity[col], errors="coerce").dropna()
            if len(d):
                m["days_to_liquidate_p95"] = float(np.percentile(d, 95))
                m["days_to_liquidate_max"] = float(d.max())

    return m


# ==========================================================================
# The monitor
# ==========================================================================
def monitor(metrics: Mapping[str, float],
            register: Sequence[tuple] | None = None) -> pd.DataFrame:
    """Evaluate every limit in the register. Passes are reported, not just failures.

    Reporting the passes is the point. A monitor that prints only breaches gives no
    way to tell "inside every limit" from "the monitor did not run", and the
    ``headroom`` column turns a binary into a gauge -- a limit at 96% of its
    threshold has not breached and is not comfortable either.
    """
    reg = list(register or config.RISK_LIMITS)
    rows = []
    for metric, op, threshold, severity in reg:
        value = metrics.get(metric)
        if value is None or not np.isfinite(float(value)):
            rows.append({"metric": metric, "operator": op, "threshold": threshold,
                         "severity": severity, "value": np.nan, "status": "not measured",
                         "headroom": np.nan, "headroom_%": np.nan})
            continue
        v = float(value)
        ok = OPERATORS[op](v, float(threshold))
        # Headroom is signed so that positive always means "room left", whichever
        # direction the limit points.
        room = (float(threshold) - v) if op in ("<=", "<") else (v - float(threshold))
        denom = abs(float(threshold)) if float(threshold) != 0 else np.nan
        rows.append({
            "metric": metric, "operator": op, "threshold": float(threshold),
            "severity": severity, "value": v,
            "status": "ok" if ok else "BREACH",
            "headroom": room,
            "headroom_%": room / denom * 100 if np.isfinite(denom) else np.nan,
        })

    out = pd.DataFrame(rows).set_index("metric")
    breaches = out[out["status"] == "BREACH"]
    out.attrs["n_limits"] = len(out)
    out.attrs["n_measured"] = int((out["status"] != "not measured").sum())
    out.attrs["n_breach"] = len(breaches)
    out.attrs["n_hard_breach"] = int((breaches["severity"] == "hard").sum())
    out.attrs["n_soft_breach"] = int((breaches["severity"] == "soft").sum())
    out.attrs["n_not_measured"] = int((out["status"] == "not measured").sum())
    # A limit inside 10% of its threshold is worth naming before it breaches.
    near = out[(out["status"] == "ok") & (out["headroom_%"] < 10)]
    out.attrs["near_limit"] = list(near.index)
    return out.sort_values(["status", "severity", "headroom_%"])


def log_breaches(mon: pd.DataFrame, as_of=None, run_id: str = "",
                 path=None) -> pd.DataFrame:
    """Append today's breaches to a dated CSV that outlives the session.

    Deliberately outside ``data/cache/`` for the same reason
    ``fii_dii_history.csv`` is: deleting the cache is safe, and deleting the
    record of when the book was outside its limits is not. The log is the only
    artefact here that answers a question about the past.

    Nothing is written when there is no breach -- an empty log means a clean
    history, and padding it with "all clear" rows would bury the entries that
    matter.
    """
    path = config.BREACH_LOG if path is None else path
    stamp = pd.Timestamp(as_of or pd.Timestamp.utcnow()).tz_localize(None)
    breaches = mon[mon["status"] == "BREACH"]
    if breaches.empty:
        return pd.DataFrame()

    rows = breaches.reset_index().assign(
        as_of=stamp.date(),
        logged_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_id=run_id,
    )
    cols = ["as_of", "run_id", "metric", "severity", "value", "operator",
            "threshold", "headroom", "logged_at"]
    rows = rows[[c for c in cols if c in rows.columns]]
    header = not path.exists()
    rows.to_csv(path, mode="a", header=header, index=False)
    rows.attrs["path"] = str(path)
    return rows


def breach_history(path=None, days: int = 365) -> pd.DataFrame:
    """Read the breach log back. What has actually been going wrong, and how often.

    A single breach is an incident; the same limit breaching every month is a
    limit that is set wrong, or a strategy that does not fit inside it. Only the
    history distinguishes the two, which is the whole argument for keeping one.
    """
    path = config.BREACH_LOG if path is None else path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["as_of"])
    if days:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
        df = df[df["as_of"] >= cutoff]
    if df.empty:
        return df
    summary = (df.groupby(["metric", "severity"])
               .agg(n_breaches=("value", "size"),
                    first=("as_of", "min"), last=("as_of", "max"),
                    worst_value=("value", "max"))
               .sort_values("n_breaches", ascending=False))
    summary.attrs["n_rows"] = len(df)
    summary.attrs["window_days"] = days
    return summary


# ==========================================================================
# Guard
# ==========================================================================
def assert_within_limits(mon: pd.DataFrame, label: str = "book",
                         allow_soft: bool = True) -> pd.DataFrame:
    """Regression guard: stop on a hard breach.

    Soft breaches pass by default and are returned in the report so they can be
    escalated by a human, which is what "soft" means. Set ``allow_soft=False``
    for a production run where nothing should be outside any limit without an
    explicit decision.

    ``not measured`` is treated as a failure for hard limits. A hard limit that
    could not be evaluated has not been satisfied -- if the covariance matrix is
    missing, the volatility limit is unknown, and an unknown hard limit is
    exactly the situation the register exists to prevent.
    """
    hard_breach = mon[(mon["status"] == "BREACH") & (mon["severity"] == "hard")]
    hard_unmeasured = mon[(mon["status"] == "not measured") & (mon["severity"] == "hard")]
    soft_breach = mon[(mon["status"] == "BREACH") & (mon["severity"] == "soft")]

    problems = []
    if len(hard_breach):
        detail = "; ".join(
            f"{m} = {r['value']:.4g} (limit {r['operator']} {r['threshold']:.4g})"
            for m, r in hard_breach.iterrows())
        problems.append(f"{len(hard_breach)} hard limit breach(es): {detail}")
    if len(hard_unmeasured):
        problems.append(
            f"{len(hard_unmeasured)} hard limit(s) could not be measured "
            f"({', '.join(hard_unmeasured.index)}) -- an unmeasured hard limit is "
            f"not a satisfied one")
    if not allow_soft and len(soft_breach):
        problems.append(f"{len(soft_breach)} soft limit breach(es) with "
                        f"allow_soft=False: {', '.join(soft_breach.index)}")

    if problems:
        raise AssertionError(f"{label}: " + "; ".join(problems))

    return pd.DataFrame([{
        "limits_in_register": mon.attrs.get("n_limits", len(mon)),
        "measured": mon.attrs.get("n_measured", np.nan),
        "hard_breaches": 0,
        "soft_breaches": len(soft_breach),
        "near_limit": ", ".join(mon.attrs.get("near_limit", [])) or "none",
        "result": ("book is inside every hard limit"
                   + (f"; {len(soft_breach)} soft breach(es) to escalate"
                      if len(soft_breach) else "")),
    }])
