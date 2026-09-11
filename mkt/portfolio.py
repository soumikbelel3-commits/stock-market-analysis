"""Position sizing: weighting schemes, vol targeting, caps, and the lot-size trap.

Everything above this module produces a *ranking*. A ranking is not a portfolio.
This module turns one into the other, and then refuses to pretend the result is
tradeable when it is not.

Trap #5 -- the shortability trap
--------------------------------
A market-neutral book with continuous weights is **unimplementable in India**.
Cash-market shorts cannot be carried overnight; they are intraday only. An
overnight short must be a stock future, which imposes three constraints that
continuous weights quietly ignore:

* only F&O-eligible names (all 50 Nifty constituents qualify -- but that stops
  being true the moment the universe widens);
* **whole lots**, so short weights quantise. Measured on 2026-09-08 against live
  ``fo_mktlots.csv``, one lot of a Nifty 50 name costs between INR 4.3 lakh
  (INFY) and 11.0 lakh (APOLLOHOSP), median 6.4 lakh. At the default INR 1 crore
  book that is **4.3%-11.0% of capital per single lot** -- so the short leg
  cannot express any target weight below about 4.3%, and a 5% target in
  APOLLOHOSP rounds to either 0% or 11%;
* SEBI's F&O ban list makes names un-shortable on any given day, and margin is
  SPAN+ELM on the notional rather than on the net.

The long leg has no such problem: cash equity buys in single shares.

So sizing runs **after** lot sizes are known -- which is why notebook 09 runs
before notebook 10 -- and ``assert_implementable`` raises when a short leg
cannot be expressed in whole lots at the stated capital, reporting the minimum
capital that would work. ``quantize_to_lots`` then reports realised weights next
to target weights, so the tracking error lot rounding introduces is visible
rather than assumed away.

Zero new dependencies
---------------------
Risk parity is ~20 lines of fixed-point numpy rather than an optimiser.
Constrained minimum variance uses ``scipy.optimize`` SLSQP -- scipy is already
in ``requirements.txt``. This follows the precedent set when notebook 03
hand-rolled PCA rather than adding sklearn.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config, dashboard, indicators as ind

TRADING_DAYS = ind.TRADING_DAYS


# ==========================================================================
# Weighting schemes
# ==========================================================================
def equal_weight(names: Iterable[str]) -> pd.Series:
    names = list(names)
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def inverse_vol(rets: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Weight inversely to realised volatility -- the cheapest risk control there is.

    Equal *weight* is not equal *risk*: a name at 45% vol contributes roughly
    three times the risk of one at 15% for the same rupee. This is the one-line
    correction, and it needs no covariance matrix.
    """
    r = rets.tail(window) if window else rets
    vol = r.std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.dropna()
    if inv.empty:
        return equal_weight(rets.columns)
    return (inv / inv.sum()).reindex(rets.columns).fillna(0.0)


def risk_parity(cov: pd.DataFrame, tol: float = 1e-8,
                max_iter: int = 10_000, damping: float = 0.5) -> pd.Series:
    """Equal *risk contribution* weights, by fixed-point iteration.

    Inverse-vol equalises standalone risk; risk parity equalises risk after
    correlation, so two names that move together stop being counted as two
    independent bets. The iteration is ``w <- 1 / (Sigma w)`` renormalised, with
    damping for stability -- about twenty lines and no optimiser, on the
    precedent that notebook 03 hand-rolled PCA rather than adding sklearn.
    """
    S = cov.to_numpy(dtype=float)
    n = S.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov.index)

    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        mrc = S @ w
        mrc = np.where(np.abs(mrc) < 1e-18, 1e-18, mrc)
        w_new = 1.0 / mrc
        w_new = np.clip(w_new, 0.0, None)
        total = w_new.sum()
        if not np.isfinite(total) or total <= 0:
            break
        w_new /= total
        step = np.max(np.abs(w_new - w))
        w = damping * w + (1 - damping) * w_new
        w /= w.sum()
        if step < tol:
            break
    return pd.Series(w, index=cov.index)


def min_variance(cov: pd.DataFrame, cap: float = config.MAX_POSITION_WEIGHT,
                 floor: float = 0.0) -> pd.Series:
    """Long-only minimum-variance weights with a per-name cap.

    Uncapped minimum variance concentrates viciously -- it will happily put 60%
    in the single lowest-vol name, which is a covariance-estimation artefact
    rather than a view. The cap is what makes the output a portfolio.
    """
    from scipy.optimize import minimize

    S = cov.to_numpy(dtype=float)
    n = S.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    cap = max(cap, 1.0 / n)                 # an infeasible cap is a caller error

    res = minimize(
        lambda w: float(w @ S @ w),
        x0=np.full(n, 1.0 / n),
        jac=lambda w: 2 * (S @ w),
        method="SLSQP",
        bounds=[(floor, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x if res.success else np.full(n, 1.0 / n)
    w = np.clip(w, 0.0, None)
    out = pd.Series(w / w.sum(), index=cov.index)
    out.attrs["converged"] = bool(res.success)
    out.attrs["message"] = str(res.message)
    return out


SCHEMES = {
    "equal": lambda rets, cov, cap: equal_weight(cov.index),
    "inverse_vol": lambda rets, cov, cap: inverse_vol(rets),
    "risk_parity": lambda rets, cov, cap: risk_parity(cov),
    "min_variance": lambda rets, cov, cap: min_variance(cov, cap),
}


def compare_schemes(rets: pd.DataFrame, cov: pd.DataFrame,
                    cap: float = config.MAX_POSITION_WEIGHT) -> pd.DataFrame:
    """Every scheme side by side, with the diagnostics that separate them."""
    cols = {}
    for name, fn in SCHEMES.items():
        try:
            cols[name] = fn(rets, cov, cap)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [portfolio] {name} failed: {type(exc).__name__}: {exc}")
    out = pd.DataFrame(cols)
    stats = {}
    for name in out.columns:
        w = out[name].fillna(0.0)
        stats[name] = {
            "ann_vol_%": portfolio_vol(w, cov),
            "max_weight_%": float(w.max() * 100),
            "effective_n": effective_n(w),
            "top3_share_%": float(w.nlargest(3).sum() * 100),
        }
    out.attrs["stats"] = pd.DataFrame(stats).T
    return out


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame,
                  annualize: bool = True) -> float:
    """Annualised portfolio volatility in percent."""
    idx = cov.index
    w = weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    if var < 0:
        return float("nan")
    v = np.sqrt(var)
    return float(v * (np.sqrt(TRADING_DAYS) if annualize else 1.0) * 100)


def effective_n(weights: pd.Series) -> float:
    """Inverse Herfindahl: how many positions the book *really* holds."""
    w = weights.abs()
    total = w.sum()
    if total <= 0:
        return float("nan")
    p = w / total
    return float(1.0 / (p ** 2).sum())


# ==========================================================================
# Vol targeting, caps and regime-conditional gross
# ==========================================================================
def vol_target(weights: pd.Series, cov: pd.DataFrame,
               target_vol: float = config.TARGET_PORTFOLIO_VOL_PCT,
               max_leverage: float = 1.0) -> tuple[pd.Series, float]:
    """Scale the book so its ex-ante vol hits the target. Returns (weights, scale).

    ``max_leverage`` defaults to **1.0**, i.e. vol targeting may de-risk but not
    lever up. That default exists because of an ordering bug this function is
    easy to create: caps are applied at the gross the macro regime authorised,
    and a scale above 1 then multiplies every weight straight back through both
    the position cap and that gross budget. A 12% position cap silently becomes
    a 21% position at 1.7x, and the book runs leverage the risk regime never
    approved.

    Scaling *down* is always safe -- it cannot breach a cap -- so the asymmetric
    default is the conservative one. Raise it deliberately, and re-apply
    ``apply_caps`` afterwards if you do.

    The scale actually required to hit the target is recorded in
    ``.attrs['required_scale']`` so an under-risked book reports the shortfall
    rather than hiding it.
    """
    current = portfolio_vol(weights, cov)
    if not np.isfinite(current) or current <= 0:
        out = weights.copy()
        out.attrs.update({"required_scale": np.nan, "applied_scale": 1.0,
                          "vol_before_%": current, "vol_after_%": current,
                          "constrained": False})
        return out, 1.0
    required = float(target_vol / current)
    scale = float(np.clip(required, 0.0, max_leverage))
    out = weights * scale
    out.attrs.update({
        "required_scale": required,
        "applied_scale": scale,
        "vol_before_%": current,
        "vol_after_%": current * scale,
        "target_vol_%": target_vol,
        "max_leverage": max_leverage,
        "constrained": bool(required > max_leverage + 1e-12),
    })
    return out, scale


def _water_fill(base: pd.Series, caps: pd.Series, total: float,
                tol: float = 1e-12) -> pd.Series:
    """Allocate ``total`` proportionally to ``base``, respecting per-item caps.

    Items that would exceed their cap are pinned at it and the remainder is
    re-shared among the rest, repeatedly. This terminates in at most one pass
    per item and lands exactly on ``total`` whenever the caps admit it, which an
    iterative clip-and-renormalise does not -- renormalising after a clip is
    precisely what pushes a capped name back over its cap.
    """
    out = pd.Series(0.0, index=base.index, dtype=float)
    free = list(base.index)
    remaining = float(total)

    for _ in range(len(base) + 1):
        if not free or remaining <= tol:
            break
        b = base.reindex(free)
        c = caps.reindex(free)
        if b.sum() <= tol:                       # nothing to weight by: share evenly
            share = pd.Series(remaining / len(free), index=free)
        else:
            share = remaining * b / b.sum()
        over = share > c + tol
        if not over.any():
            out.loc[free] = share
            remaining = 0.0
            break
        hit = [t for t in free if bool(over.get(t, False))]
        out.loc[hit] = c.reindex(hit)
        remaining -= float(c.reindex(hit).sum())
        free = [t for t in free if t not in set(hit)]

    return out


def apply_caps(weights: pd.Series, sectors: Mapping[str, str] | pd.Series,
               max_pos: float = config.MAX_POSITION_WEIGHT,
               max_sector: float = config.MAX_SECTOR_WEIGHT) -> pd.Series:
    """Clip per-name and per-sector exposure, preserving gross and sign.

    Allocation is two-stage water-filling -- gross across sectors first, then
    within each sector across its names. That respects both caps *exactly* and
    terminates, where clipping and renormalising in a loop does not: every
    renormalisation after a clip can push a capped name back over its cap, which
    is a real bug rather than a rounding artefact.

    Raises when the caps cannot admit the requested gross, because silently
    returning a smaller book would misreport the exposure actually taken.
    """
    w = weights.dropna().astype(float)
    if w.empty:
        return w
    sec = pd.Series(sectors) if not isinstance(sectors, pd.Series) else sectors
    sec = sec.reindex(w.index).fillna("?")
    sign = np.sign(w).replace(0, 1.0)
    base = w.abs()
    gross = float(base.sum())

    members = {s: list(g.index) for s, g in base.groupby(sec)}
    # A sector can hold no more than its own cap, and no more than its names allow.
    sec_caps = pd.Series({s: min(max_sector, len(m) * max_pos)
                          for s, m in members.items()})
    if float(sec_caps.sum()) < gross - 1e-9:
        raise ValueError(
            f"apply_caps: gross {gross:.3f} exceeds total capacity "
            f"{sec_caps.sum():.3f} under max_pos={max_pos} and "
            f"max_sector={max_sector} across {len(members)} sector(s). "
            f"Add names, widen a cap, or lower gross.")

    sec_base = base.groupby(sec).sum()
    sec_alloc = _water_fill(sec_base, sec_caps, gross)

    a = pd.Series(0.0, index=base.index, dtype=float)
    for s, m in members.items():
        caps = pd.Series(max_pos, index=m)
        a.loc[m] = _water_fill(base.reindex(m), caps, float(sec_alloc[s]))

    out = a * sign
    out.attrs["max_position"] = float(a.max()) if len(a) else np.nan
    out.attrs["max_sector"] = float(a.groupby(sec).sum().max()) if len(a) else np.nan
    out.attrs["gross"] = float(a.sum())
    return out


def regime_gross(data: dict | None = None,
                 table: Mapping[str, float] | None = None) -> tuple[float, str]:
    """Gross exposure conditioned on the top-down chain.

    Reads notebook 02's risk label and notebook 01's macro nowcast from
    ``dashboard.json``. This is the whole reason the macro chain was built: a
    risk-off tape should shrink the book, not merely be described in a
    paragraph above it.

    Returns ``(gross, rationale)``.
    """
    data = dashboard.read() if data is None else data
    table = dict(table or config.GROSS_BY_REGIME)

    risk = str(data.get("02_risk", {}).get("signal", "")).strip()
    gross = table.get(risk)
    if gross is None:
        gross = table.get("Neutral", 1.0)
        why = f"risk label {risk or 'unavailable'!r} not in GROSS_BY_REGIME; using Neutral"
    else:
        why = f"notebook 02 risk regime = {risk}"

    nowcast = str(data.get("01_macro", {}).get("nowcast", {}).get("label", "")).strip()
    if nowcast == "Contractionary" and gross > min(table.values()):
        step = sorted(table.values())
        gross = step[max(0, step.index(gross) - 1)]
        why += f"; stepped down one notch on a contractionary macro nowcast"
    elif nowcast:
        why += f"; macro nowcast {nowcast} (no adjustment)"

    return float(gross), why


# ==========================================================================
# Long/short construction
# ==========================================================================
def leg_betas(weights: pd.Series, betas: pd.Series) -> float:
    w = weights.abs()
    b = betas.reindex(w.index)
    j = pd.concat([w, b], axis=1, keys=["w", "b"]).dropna()
    if j.empty or j["w"].sum() == 0:
        return np.nan
    return float((j["w"] * j["b"]).sum() / j["w"].sum())


def build_long_short(long_w: pd.Series, short_w: pd.Series,
                     betas: pd.Series | None = None,
                     gross: float = 1.0,
                     neutrality: str = "beta") -> pd.DataFrame:
    """Combine two legs into one book at a target gross exposure.

    ``neutrality``:

    * ``"beta"`` -- size the legs so their beta contributions cancel. Dollar
      neutrality is not market neutrality: a long book of high-beta names
      against an equal-rupee short book of low-beta names is still a long
      position in the market wearing a neutral label.
    * ``"dollar"`` -- equal rupees each side, reported for comparison.

    Each input leg is expected to sum to 1 in absolute terms; the result carries
    signed weights summing to ``gross`` in absolute terms.
    """
    long_w = long_w[long_w > 0].abs()
    short_w = short_w[short_w > 0].abs()
    if long_w.empty:
        raise ValueError("build_long_short: long leg is empty")
    long_w = long_w / long_w.sum()
    if not short_w.empty:
        short_w = short_w / short_w.sum()

    if short_w.empty:
        l_scale, s_scale = gross, 0.0
        method = "long-only (no short candidates)"
    elif neutrality == "dollar" or betas is None:
        l_scale = s_scale = gross / 2.0
        method = "dollar-neutral"
    else:
        bl, bs = leg_betas(long_w, betas), leg_betas(short_w, betas)
        # Both leg betas must be positive. A negative one would make the
        # solution below hand back a negative scale -- silently flipping a leg
        # to the other side of the book, which is worse than not matching.
        if not (np.isfinite(bl) and np.isfinite(bs)) or bl <= 0 or bs <= 0:
            l_scale = s_scale = gross / 2.0
            method = (f"dollar-neutral (leg betas unusable: long {bl:.2f}, short {bs:.2f})"
                      if np.isfinite(bl) and np.isfinite(bs)
                      else "dollar-neutral (betas unavailable)")
        else:
            # L*bl = S*bs and L+S = gross
            l_scale = gross * bs / (bl + bs)
            s_scale = gross * bl / (bl + bs)
            method = "beta-neutral"

    rows = []
    for t, w in long_w.items():
        rows.append({"ticker": t, "side": "long", "weight": float(w * l_scale)})
    for t, w in short_w.items():
        rows.append({"ticker": t, "side": "short", "weight": float(-w * s_scale)})

    book = pd.DataFrame(rows).set_index("ticker")
    if betas is not None:
        book["beta"] = betas.reindex(book.index)
        book["beta_contrib"] = book["weight"] * book["beta"]
    book.attrs["method"] = method
    book.attrs["gross"] = float(book["weight"].abs().sum())
    book.attrs["net"] = float(book["weight"].sum())
    book.attrs["net_beta"] = (float(book["beta_contrib"].sum())
                              if "beta_contrib" in book.columns else np.nan)
    book.attrs["long_scale"] = l_scale
    book.attrs["short_scale"] = s_scale
    return book


# ==========================================================================
# Lot sizes -- trap #5
# ==========================================================================
def lot_table(tickers: Iterable[str], live: pd.Series | None = None) -> pd.DataFrame:
    """Lot size per name, live where available and from the static table otherwise.

    ``config.LOT_SIZE_FALLBACK`` is a dated snapshot so notebook 10 can run when
    the derivatives endpoints are down -- which is what makes notebook 09 safely
    cuttable. When both are present the divergence is reported rather than
    silently preferred one way or the other: SEBI revises lot sizes, and a stale
    entry sizes a real trade wrongly.
    """
    tickers = list(tickers)
    fb = pd.Series({t: config.LOT_SIZE_FALLBACK.get(t) for t in tickers}, dtype="float64")
    rows = {"lot_fallback": fb}
    if live is not None and len(live):
        lv = pd.to_numeric(live.reindex(tickers), errors="coerce")
        rows["lot_live"] = lv
        rows["lot"] = lv.fillna(fb)
        rows["source"] = pd.Series(
            np.where(lv.notna(), "live", np.where(fb.notna(), "fallback", "missing")),
            index=tickers)
        rows["diverged"] = (lv.notna() & fb.notna() & (lv != fb))
    else:
        rows["lot"] = fb
        rows["source"] = pd.Series(
            np.where(fb.notna(), f"fallback ({config.LOT_SIZE_FALLBACK_ASOF})", "missing"),
            index=tickers)
        rows["diverged"] = pd.Series(False, index=tickers)
    out = pd.DataFrame(rows)
    out.attrs["n_missing"] = int(out["lot"].isna().sum())
    out.attrs["n_diverged"] = int(out["diverged"].sum())
    return out


def quantize_to_lots(book: pd.DataFrame, prices: pd.Series, lots: pd.Series,
                     capital: float = config.DEFAULT_CAPITAL) -> pd.DataFrame:
    """Round target weights to what can actually be traded.

    The two sides quantise differently, and conflating them overstates the
    problem by a factor of several hundred:

    * **longs** are cash equity and buy in *single shares*, so the rounding is
      negligible;
    * **shorts** must be stock futures and trade in *whole lots*, so the
      rounding is first-order at a retail-sized book.

    Returns target vs realised weight per name, so the tracking error rounding
    introduces is a number in a table rather than an assumption.
    """
    px = pd.to_numeric(prices.reindex(book.index), errors="coerce")
    lt = pd.to_numeric(lots.reindex(book.index), errors="coerce")

    rows = []
    for t, r in book.iterrows():
        side = r["side"]
        w = float(r["weight"])
        price, lot = float(px.get(t, np.nan)), float(lt.get(t, np.nan))
        # Long: one share is the unit. Short: one F&O lot is the unit.
        unit = 1.0 if side == "long" else lot
        unit_value = price * unit if np.isfinite(price) and np.isfinite(unit) else np.nan
        target_value = w * capital

        if not np.isfinite(unit_value) or unit_value <= 0:
            rows.append({"ticker": t, "side": side, "target_weight": w,
                         "price": price, "lot": lot, "unit_value": np.nan,
                         "target_units": np.nan, "units": np.nan,
                         "realised_value": np.nan, "realised_weight": np.nan,
                         "weight_error_pp": np.nan, "note": "no price or lot size"})
            continue

        target_units = target_value / unit_value
        units = float(np.round(target_units))
        # Never round a wanted position all the way to nothing without saying so.
        note = ""
        if units == 0 and abs(target_units) > 0:
            note = (f"rounds to zero -- one unit is "
                    f"{unit_value / capital * 100:.2f}% of capital")
        realised_value = units * unit_value
        realised_w = realised_value / capital
        rows.append({
            "ticker": t, "side": side, "target_weight": w, "price": price,
            "lot": lot if side == "short" else 1.0, "unit_value": unit_value,
            "target_units": target_units, "units": units,
            "realised_value": realised_value, "realised_weight": realised_w,
            "weight_error_pp": (realised_w - w) * 100, "note": note,
        })

    out = pd.DataFrame(rows).set_index("ticker")
    out.attrs["capital"] = capital
    out.attrs["target_gross"] = float(out["target_weight"].abs().sum())
    out.attrs["realised_gross"] = float(out["realised_weight"].abs().sum())
    out.attrs["target_net"] = float(out["target_weight"].sum())
    out.attrs["realised_net"] = float(out["realised_weight"].sum())
    out.attrs["tracking_error_pp"] = float(
        np.sqrt((out["weight_error_pp"].dropna() ** 2).sum()))
    out.attrs["dropped_to_zero"] = [t for t, r in out.iterrows()
                                    if r["units"] == 0 and abs(r["target_weight"]) > 0]
    return out


def assert_implementable(book: pd.DataFrame, prices: pd.Series, lots: pd.Series,
                         capital: float = config.DEFAULT_CAPITAL,
                         label: str = "book") -> pd.DataFrame:
    """Trap #5 regression guard. Raises when the short leg cannot be traded.

    A short position is expressible only if its target notional is at least one
    whole futures lot. When it is not, the failure is reported with the
    **minimum capital that would work**, because "increase the book to INR 2.1
    crore or drop this name" is an actionable answer and "weights look fine" is
    not.
    """
    px = pd.to_numeric(prices.reindex(book.index), errors="coerce")
    lt = pd.to_numeric(lots.reindex(book.index), errors="coerce")

    rows, bad = [], []
    for t, r in book.iterrows():
        if r["side"] != "short":
            continue
        w, price, lot = abs(float(r["weight"])), float(px.get(t, np.nan)), float(lt.get(t, np.nan))
        if not (np.isfinite(price) and np.isfinite(lot) and lot > 0):
            bad.append((t, "no lot size or price", np.nan))
            rows.append({"ticker": t, "target_weight": -w, "lot": lot,
                         "lot_value": np.nan, "lots_affordable": np.nan,
                         "min_capital": np.nan, "ok": False})
            continue
        lot_value = price * lot
        affordable = w * capital / lot_value
        min_capital = lot_value / w if w > 0 else np.inf
        ok = affordable >= 1.0
        rows.append({"ticker": t, "target_weight": -w, "lot": lot,
                     "lot_value": lot_value, "lots_affordable": affordable,
                     "lot_as_%_of_capital": lot_value / capital * 100,
                     "min_capital": min_capital, "ok": ok})
        if not ok:
            bad.append((t, f"{affordable:.2f} lots affordable", min_capital))

    report = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    if bad:
        need = max((b[2] for b in bad if np.isfinite(b[2])), default=np.nan)
        detail = "; ".join(f"{t} ({why})" for t, why, _ in bad[:5])
        raise AssertionError(
            f"{label}: {len(bad)} short leg(s) cannot be expressed in whole lots at "
            f"capital {capital:,.0f} -- trap #5. {detail}. "
            f"Minimum capital that would clear every short: {need:,.0f}."
            if np.isfinite(need) else
            f"{label}: {len(bad)} short leg(s) not implementable -- {detail}")

    report.attrs["capital"] = capital
    report.attrs["result"] = f"all {len(report)} short leg(s) expressible in whole lots"
    return report


def liquidity_check(book: pd.DataFrame, adv_value: pd.Series,
                    capital: float = config.DEFAULT_CAPITAL,
                    max_pct_adv: float = config.MAX_PCT_OF_ADV) -> pd.DataFrame:
    """Position size against average daily traded value.

    ``adv_value`` is rupees a day, not shares. A position that is 40% of a day's
    volume is not a position, it is a week of slippage.
    """
    adv = pd.to_numeric(adv_value.reindex(book.index), errors="coerce")
    value = book["weight"].abs() * capital
    pct = value / adv.replace(0, np.nan)
    out = pd.DataFrame({
        "side": book["side"],
        "position_value": value,
        "adv_value": adv,
        "pct_of_adv": pct * 100,
        "days_to_trade": pct / max_pct_adv,
        "breach": pct > max_pct_adv,
    })
    out.attrs["n_breach"] = int(out["breach"].fillna(False).sum())
    out.attrs["max_pct_adv"] = max_pct_adv * 100
    return out


# ==========================================================================
# The real book
# ==========================================================================
def load_book(path=None) -> pd.Series | None:
    """Read an optional hand-maintained book from ``data/portfolio.csv``.

    Accepts ``ticker,weight`` or ``ticker,shares`` (+ optional ``price``).
    Returns None when the file is absent, which is the normal case -- notebook
    11 then runs on the model book alone.
    """
    path = config.PORTFOLIO_CSV if path is None else path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError(f"{path}: needs a 'ticker' column, found {list(df.columns)}")
    df = df.set_index(cols["ticker"])

    if "weight" in cols:
        w = pd.to_numeric(df[cols["weight"]], errors="coerce").dropna()
        # Accept either fractions or percentages, decided by magnitude.
        if w.abs().sum() > 3:
            w = w / 100.0
    elif "shares" in cols and "price" in cols:
        value = (pd.to_numeric(df[cols["shares"]], errors="coerce")
                 * pd.to_numeric(df[cols["price"]], errors="coerce"))
        w = value / value.abs().sum()
    else:
        raise ValueError(f"{path}: needs 'weight', or 'shares' and 'price'")

    w.name = "weight"
    w.attrs["source"] = str(path)
    return w.dropna()


def gap_table(model: pd.Series, actual: pd.Series) -> pd.DataFrame:
    """Model book against real book: what you should hold vs what you do."""
    idx = model.index.union(actual.index)
    m = model.reindex(idx).fillna(0.0)
    a = actual.reindex(idx).fillna(0.0)
    out = pd.DataFrame({"model_weight": m, "actual_weight": a, "gap": a - m})
    out["action"] = np.where(out["gap"] > 0.005, "reduce",
                             np.where(out["gap"] < -0.005, "add", "hold"))
    out = out.sort_values("gap", key=abs, ascending=False)
    out.attrs["turnover_to_align"] = float(out["gap"].abs().sum() / 2)
    return out
