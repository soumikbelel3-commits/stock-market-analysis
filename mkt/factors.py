"""Cross-sectional style factor risk model, in the Barra tradition.

Why this module exists
----------------------
Before it, the chain measured risk two ways: a single beta to the Nifty
(``risk.portfolio_beta``) and betas to six macro proxies (``risk.factor_betas``).
Both are useful and neither answers the question a risk committee actually asks:

    *When this book loses money, what will it have been a bet on?*

A long/short book of Indian large caps is almost never a bet on "the market". It
is a bet on momentum against value, or on quality against size, and those bets
are invisible in a covariance matrix of fifty tickers. ``risk.concentration``
already reports that the leading eigenvalue eats most of the variance -- this
module puts a **name** on that eigenvalue.

How it works
------------
The standard cross-sectional construction:

1. **Exposures** ``X`` -- each name's standardised score on each style. Winsorised
   at ``config.FACTOR_WINSOR_Z`` and z-scored within the date, so an exposure of
   +1 always means "one cross-sectional standard deviation" and factors are
   comparable with one another.
2. **Factor returns** ``f`` -- estimated per date by weighted least squares of
   name returns on exposures, weights proportional to sqrt(market cap). This is a
   *regression*, not a portfolio sort: it gives the return to a unit exposure of
   each factor holding the others fixed, which is what a decomposition needs.
3. **Factor covariance** ``F`` -- exponentially weighted, so a covariance
   estimated over two years still reflects last quarter more heavily than the
   quarter before the pandemic.
4. **Specific variance** ``D`` -- the EWMA variance of the regression residuals,
   per name. What is left when every style is accounted for.

Total risk is then ``w'(XFX' + D)w``, and the split between the two terms is the
answer: how much of this book is a style bet, and how much is stock picking.

Point-in-time honesty (trap #4 again)
-------------------------------------
Momentum, low-volatility and size are computed from prices and are point-in-time
by construction. Value, quality and growth come from ``yfinance`` statements,
which have no point-in-time history -- so their exposures are **held at today's
values** across the estimation window.

For an *alpha* model that would be fatal, and notebook 07 refuses to do it. For a
*risk* model it is a much smaller sin -- the object being estimated is a
covariance, not an expected return, and a name's quality decile is far more
persistent than its return. It is still an assumption, so ``exposure_report``
labels every factor ``pit`` or ``static`` and ``assert_factor_pit`` makes the
split impossible to lose track of.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config, indicators as ind

TRADING_DAYS = ind.TRADING_DAYS

# Which styles can be derived from prices alone, and are therefore point-in-time.
PIT_FACTORS = {"momentum", "low_vol", "size_price", "reversal"}
STATIC_FACTORS = {"value", "quality", "growth", "size"}


# ==========================================================================
# Exposures
# ==========================================================================
def winsorised_z(s: pd.Series, clip: float | None = None) -> pd.Series:
    """Z-score within the cross-section, winsorised rather than clipped-then-scaled.

    Order matters. Clipping raw values *before* standardising lets one outlier set
    the scale and then be pulled back in, which compresses everyone else towards
    zero; winsorising the z-score keeps the scale honest and only limits the
    extreme name's influence. ``score.robust_z`` uses median/MAD for the same
    reason -- this is the mean/sd sibling, which is what a factor regression
    expects because the WLS fit assumes a centred design matrix.
    """
    c = config.FACTOR_WINSOR_Z if clip is None else float(clip)
    x = pd.to_numeric(s, errors="coerce")
    mu, sd = x.mean(), x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((x - mu) / sd).clip(-c, c).fillna(0.0)


def build_exposures(scorecard: pd.DataFrame,
                    price_panel: pd.DataFrame | None = None,
                    as_of: pd.Timestamp | None = None,
                    spec: Mapping[str, tuple[str, int]] | None = None) -> pd.DataFrame:
    """Today's exposure matrix: names x styles, standardised.

    ``scorecard`` is the notebook 06 screen (``outputs/nifty50_screen.csv``);
    ``price_panel`` supplies the point-in-time price factors when present. Any
    style whose source column is missing is dropped with a note rather than
    filled with zeros -- a factor that is silently all-zero contributes nothing to
    risk and looks like a hedge that is not there.
    """
    spec = dict(spec or config.STYLE_FACTORS)
    df = scorecard.copy()

    # Derived columns the spec refers to but the screen does not carry directly.
    if "market_cap" in df.columns and "log_mcap" not in df.columns:
        mc = pd.to_numeric(df["market_cap"], errors="coerce")
        df["log_mcap"] = np.log(mc.where(mc > 0))
    if "PE" in df.columns and "earnings_yield" not in df.columns:
        pe = pd.to_numeric(df["PE"], errors="coerce")
        df["earnings_yield"] = 1.0 / pe.where(pe > 0)
    if "mom_12_1_%" not in df.columns:
        if price_panel is not None and not price_panel.empty:
            df["mom_12_1_%"] = momentum_12_1(price_panel, as_of).reindex(df.index)
        elif {"ret_12m_%", "ret_3m_%"} <= set(df.columns):
            # 12-month return excluding the most recent month is the convention;
            # the screen carries 12m and 3m, so this is the closest available
            # proxy and is labelled as such in exposure_report.
            df["mom_12_1_%"] = (pd.to_numeric(df["ret_12m_%"], errors="coerce")
                                - pd.to_numeric(df["ret_3m_%"], errors="coerce") / 3.0)

    cols, dropped = {}, []
    for factor, (source, direction) in spec.items():
        if source not in df.columns or df[source].notna().sum() < 3:
            dropped.append((factor, source))
            continue
        cols[factor] = winsorised_z(df[source]) * direction

    if not cols:
        raise ValueError(f"build_exposures: no usable style columns; "
                         f"looked for {[s for s, _ in spec.values()]}")

    X = pd.DataFrame(cols, index=df.index)
    X.attrs["dropped"] = dropped
    X.attrs["as_of"] = as_of
    X.attrs["pit"] = {f: (f in PIT_FACTORS) for f in X.columns}
    return X


def momentum_12_1(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.Series:
    """12-month return skipping the most recent month.

    The skip is not cosmetic. One-month reversal is a well-documented and
    *opposite-signed* effect, so a raw 12-month momentum factor is a blend of two
    signals fighting each other -- which is one plausible reading of why notebook
    07's short-horizon momentum factors came back with the wrong sign.
    """
    px = panel if as_of is None else panel.loc[:pd.Timestamp(as_of)]
    if len(px) < 273:
        return pd.Series(np.nan, index=panel.columns)
    return ((px.iloc[-21] / px.iloc[-273] - 1) * 100).rename("mom_12_1_%")


def exposure_history(panel: pd.DataFrame, scorecard: pd.DataFrame,
                     dates: pd.DatetimeIndex,
                     spec: Mapping[str, tuple[str, int]] | None = None
                     ) -> dict[str, pd.DataFrame]:
    """Exposures on every estimation date: ``{factor: date x name}``.

    Price-based styles are recomputed on each date and are genuinely
    point-in-time. Statement-based styles are broadcast from today's values, which
    is the compromise trap #4 forces and which ``assert_factor_pit`` reports.
    """
    spec = dict(spec or config.STYLE_FACTORS)
    dates = pd.DatetimeIndex(dates)
    names = [c for c in panel.columns if c in scorecard.index]
    close = panel[names]

    out: dict[str, pd.DataFrame] = {}

    # --- point-in-time, from prices -------------------------------------
    if "momentum" in spec:
        mom = (close.shift(21) / close.shift(273) - 1) * 100
        out["momentum"] = mom.reindex(dates).apply(winsorised_z, axis=1)
    if "low_vol" in spec:
        vol = close.pct_change().rolling(63, min_periods=30).std() * np.sqrt(TRADING_DAYS) * 100
        out["low_vol"] = vol.reindex(dates).apply(winsorised_z, axis=1) * -1
    if "reversal" in spec:
        out["reversal"] = (close.pct_change(21) * -100).reindex(dates).apply(
            winsorised_z, axis=1)

    # --- static, from today's statements --------------------------------
    static_today = build_exposures(scorecard.loc[names], price_panel=close)
    for factor in spec:
        if factor in out:
            continue
        if factor not in static_today.columns:
            continue
        row = static_today[factor].reindex(names)
        out[factor] = pd.DataFrame(
            np.tile(row.to_numpy(dtype=float), (len(dates), 1)),
            index=dates, columns=names)

    for f, frame in out.items():
        frame.attrs["pit"] = f in PIT_FACTORS
    return out


def exposure_report(X: pd.DataFrame) -> pd.DataFrame:
    """One row per style: dispersion, coverage, and whether it is point-in-time.

    Printed before any decomposition is trusted. A style whose cross-sectional
    standard deviation has collapsed is not measuring anything, and a static style
    should be read with the trap #4 caveat attached.
    """
    rows = []
    pit = X.attrs.get("pit", {})
    for f in X.columns:
        s = pd.to_numeric(X[f], errors="coerce")
        rows.append({
            "factor": f,
            "basis": "point-in-time" if pit.get(f, f in PIT_FACTORS) else "static (trap #4)",
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
            "coverage_%": float(s.notna().mean() * 100),
            "n_at_winsor_bound": int((s.abs() >= config.FACTOR_WINSOR_Z - 1e-9).sum()),
        })
    out = pd.DataFrame(rows).set_index("factor")
    out.attrs["dropped"] = X.attrs.get("dropped", [])
    out.attrs["n_names"] = len(X)
    return out


# ==========================================================================
# Factor returns
# ==========================================================================
def estimate_factor_returns(exposures: Mapping[str, pd.DataFrame],
                            returns: pd.DataFrame,
                            weights: pd.Series | None = None,
                            min_names: int | None = None
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Weighted cross-sectional regression on every date.

    ``r_t = alpha_t * 1 + X_t f_t + u_t``, solved by WLS with weights proportional
    to sqrt(market cap). The intercept is the **market** factor: the return common
    to every name once styles are held flat, which is what makes the style returns
    interpretable as *relative* bets rather than as disguised market exposure.

    Square-root-of-cap weighting is the Barra convention. It sits between equal
    weighting -- where fifty small names outvote the index -- and cap weighting,
    where three names determine every factor return.

    Returns ``(factor_returns, residuals, r_squared)``.
    """
    min_n = config.FACTOR_MIN_NAMES if min_names is None else int(min_names)
    factors = list(exposures)
    if not factors:
        raise ValueError("estimate_factor_returns: no exposures supplied")

    dates = returns.index
    for f in factors:
        dates = dates.intersection(exposures[f].index)
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) == 0:
        raise ValueError("estimate_factor_returns: exposures and returns share no dates")

    f_rows, u_rows, r2_rows = {}, {}, {}
    for d in dates:
        y_full = pd.to_numeric(returns.loc[d], errors="coerce")
        Xd = pd.DataFrame({f: exposures[f].loc[d] for f in factors})
        j = pd.concat([y_full.rename("_y"), Xd], axis=1).dropna()
        if len(j) < min_n:
            continue
        y = j["_y"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(j)), j[factors].to_numpy(dtype=float)])

        if weights is not None:
            w = pd.to_numeric(weights.reindex(j.index), errors="coerce")
            w = np.sqrt(w.clip(lower=0).fillna(w.median())).to_numpy(dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w))
        else:
            w = np.ones(len(j))
        w = w / w.sum() * len(w)
        sw = np.sqrt(w)

        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        fitted = X @ coef
        resid = y - fitted
        ss_tot = float((w * (y - np.average(y, weights=w)) ** 2).sum())
        r2 = 1.0 - float((w * resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

        f_rows[d] = pd.Series(coef, index=["market"] + factors)
        u_rows[d] = pd.Series(resid, index=j.index)
        r2_rows[d] = r2

    if not f_rows:
        raise ValueError(f"estimate_factor_returns: no date had {min_n}+ usable names")

    F = pd.DataFrame(f_rows).T.sort_index()
    U = pd.DataFrame(u_rows).T.sort_index().reindex(columns=returns.columns)
    R2 = pd.Series(r2_rows).sort_index().rename("r2")
    F.attrs["n_dates"] = len(F)
    F.attrs["mean_r2"] = float(R2.mean())
    F.attrs["factors"] = factors
    return F, U, R2


def factor_return_summary(F: pd.DataFrame,
                          periods_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Annualised mean, volatility and t-statistic of each factor's return.

    The t-statistic here says whether the factor earned a premium **in this
    sample** -- it is a descriptive statement about the estimation window, not a
    forecast, and it is deliberately not fed back into ``score.TECHNICAL_SPEC``.
    Fitting weights to the same sample that produced them is the loop notebook 07
    exists to refuse.
    """
    rows = []
    for f in F.columns:
        s = pd.to_numeric(F[f], errors="coerce").dropna()
        if len(s) < 3:
            continue
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        rows.append({
            "factor": f,
            "ann_return_%": mu * periods_per_year * 100,
            "ann_vol_%": sd * np.sqrt(periods_per_year) * 100,
            "t_stat": mu / (sd / np.sqrt(len(s))) if sd > 0 else np.nan,
            "sharpe": (mu / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
            "hit_rate_%": float((s > 0).mean() * 100),
            "n_obs": int(len(s)),
        })
    return pd.DataFrame(rows).set_index("factor")


# ==========================================================================
# Covariance
# ==========================================================================
def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / max(float(halflife), 1.0))
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def factor_covariance(F: pd.DataFrame, halflife: int | None = None,
                      annualise: bool = True,
                      periods_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Exponentially weighted covariance of the factor returns.

    A flat two-year window says the market of eighteen months ago matters exactly
    as much as last week's. Exponential weighting says it does not, which is both
    obviously true and the reason a risk model reacts to a volatility regime
    change in weeks rather than quarters.
    """
    hl = config.FACTOR_HALFLIFE_DAYS if halflife is None else int(halflife)
    X = F.dropna(how="any")
    if len(X) < 10:
        cov = F.cov()
        cov.attrs["method"] = "sample (too few observations for EWMA)"
        return cov * (periods_per_year if annualise else 1.0)

    w = _ewma_weights(len(X), hl)
    x = X.to_numpy(dtype=float)
    mu = np.average(x, axis=0, weights=w)
    d = x - mu
    cov = (d * w[:, None]).T @ d / (1.0 - (w ** 2).sum())
    out = pd.DataFrame(cov, index=X.columns, columns=X.columns)
    if annualise:
        out = out * periods_per_year
    out.attrs["method"] = f"EWMA (halflife {hl} periods)"
    out.attrs["halflife"] = hl
    out.attrs["n_obs"] = int(len(X))
    out.attrs["annualised"] = annualise
    return out


def specific_variance(U: pd.DataFrame, halflife: int | None = None,
                      annualise: bool = True,
                      periods_per_year: float = TRADING_DAYS,
                      min_obs: int = 30) -> pd.Series:
    """EWMA variance of each name's residual: the risk no style explains.

    Names with too little history fall back to the cross-sectional median rather
    than to zero. A zero specific variance would tell the optimiser a name is
    riskless once hedged, which is the kind of error that builds an enormous
    position in the least-observed stock in the universe.
    """
    hl = config.FACTOR_HALFLIFE_DAYS if halflife is None else int(halflife)
    out = {}
    for t in U.columns:
        s = pd.to_numeric(U[t], errors="coerce").dropna()
        if len(s) < min_obs:
            out[t] = np.nan
            continue
        w = _ewma_weights(len(s), hl)
        v = s.to_numpy(dtype=float)
        mu = float(np.average(v, weights=w))
        out[t] = float(np.average((v - mu) ** 2, weights=w))
    ser = pd.Series(out, name="specific_var")
    med = ser.median()
    n_filled = int(ser.isna().sum())
    ser = ser.fillna(med)
    if annualise:
        ser = ser * periods_per_year
    ser.attrs["halflife"] = hl
    ser.attrs["n_backfilled"] = n_filled
    ser.attrs["annualised"] = annualise
    return ser


# ==========================================================================
# Risk decomposition
# ==========================================================================
def decompose_risk(weights: pd.Series, X: pd.DataFrame, F_cov: pd.DataFrame,
                   spec_var: pd.Series) -> dict:
    """Split the book's volatility into systematic and idiosyncratic.

    ``sigma^2 = (X'w)' F (X'w) + w' D w``

    The first term is what the book is betting on; the second is what it is
    betting *with*. A market-neutral book that turns out to be 85% systematic has
    not hedged its risk, it has changed which risk it holds -- and that is a
    sentence a risk committee can act on in a way that "annualised volatility
    11.4%" is not.
    """
    idx = X.index
    w = weights.reindex(idx).fillna(0.0)
    Xa = X.reindex(idx).fillna(0.0)

    common = [f for f in Xa.columns if f in F_cov.index]
    if not common:
        raise ValueError("decompose_risk: exposures and factor covariance share "
                         "no factors")
    Xa = Xa[common]
    Fm = F_cov.loc[common, common].to_numpy(dtype=float)

    b = Xa.T.to_numpy(dtype=float) @ w.to_numpy(dtype=float)      # factor exposure
    sys_var = float(b @ Fm @ b)
    d = pd.to_numeric(spec_var.reindex(idx), errors="coerce").fillna(
        pd.to_numeric(spec_var, errors="coerce").median()).to_numpy(dtype=float)
    spec_v = float((w.to_numpy(dtype=float) ** 2 * d).sum())
    total = sys_var + spec_v

    return {
        "total_vol_%": float(np.sqrt(max(total, 0)) * 100),
        "systematic_vol_%": float(np.sqrt(max(sys_var, 0)) * 100),
        "specific_vol_%": float(np.sqrt(max(spec_v, 0)) * 100),
        "systematic_share_%": float(sys_var / total * 100) if total > 0 else np.nan,
        "specific_share_%": float(spec_v / total * 100) if total > 0 else np.nan,
        "n_names": int((w != 0).sum()),
        "n_factors": len(common),
        "factor_cov_method": F_cov.attrs.get("method", ""),
    }


def factor_exposures_of_book(weights: pd.Series, X: pd.DataFrame,
                             F_cov: pd.DataFrame | None = None) -> pd.DataFrame:
    """The book's net exposure to each style, and what each contributes to risk.

    ``net_exposure`` is ``X'w``: a long/short book that is beta-neutral can still
    carry +0.8 of momentum, which is the bet nobody wrote down. When ``F_cov`` is
    supplied, the variance contribution of each factor is added -- and those sum
    to the systematic variance exactly, which is what makes the percentage column
    trustworthy.
    """
    idx = X.index
    w = weights.reindex(idx).fillna(0.0)
    b = X.reindex(idx).fillna(0.0).T @ w
    out = pd.DataFrame({"net_exposure": b})

    if F_cov is not None:
        common = [f for f in out.index if f in F_cov.index]
        Fm = F_cov.loc[common, common].to_numpy(dtype=float)
        bv = b.reindex(common).to_numpy(dtype=float)
        sys_var = float(bv @ Fm @ bv)
        contrib = bv * (Fm @ bv)                       # sums to sys_var exactly
        out.loc[common, "factor_vol_%"] = np.sqrt(np.diag(Fm)) * 100
        out.loc[common, "variance_contrib"] = contrib
        out.loc[common, "pct_of_systematic"] = (contrib / sys_var * 100
                                                if sys_var > 0 else np.nan)
        out.attrs["systematic_vol_%"] = float(np.sqrt(max(sys_var, 0)) * 100)
        out.attrs["sum_check"] = float(contrib.sum())
    out["abs_exposure"] = out["net_exposure"].abs()
    return out.sort_values("abs_exposure", ascending=False).drop(columns="abs_exposure")


def factor_model_cov(X: pd.DataFrame, F_cov: pd.DataFrame,
                     spec_var: pd.Series) -> pd.DataFrame:
    """Reconstruct the full name-by-name covariance implied by the model.

    ``Sigma = X F X' + D``. Useful because it is a *structured* estimate: fifty
    names on two years of data give a sample covariance with more parameters than
    observations, while this one is built from six factor variances and fifty
    residual variances. That is the same argument ``risk.shrunk_cov`` makes, taken
    a step further -- shrinkage regularises towards a target with no economic
    content, and this regularises towards one that does.
    """
    common = [f for f in X.columns if f in F_cov.index]
    Xa = X[common].fillna(0.0)
    Fm = F_cov.loc[common, common].to_numpy(dtype=float)
    Xn = Xa.to_numpy(dtype=float)
    d = pd.to_numeric(spec_var.reindex(X.index), errors="coerce")
    d = d.fillna(d.median()).to_numpy(dtype=float)
    S = Xn @ Fm @ Xn.T + np.diag(d)
    out = pd.DataFrame(S, index=X.index, columns=X.index)
    out.attrs["method"] = "factor model (X F X' + D)"
    out.attrs["n_factors"] = len(common)
    out.attrs["n_parameters"] = len(common) * (len(common) + 1) // 2 + len(X)
    out.attrs["n_sample_parameters"] = len(X) * (len(X) + 1) // 2
    return out


# ==========================================================================
# Guards
# ==========================================================================
def assert_factor_pit(X: pd.DataFrame, label: str = "exposures") -> pd.DataFrame:
    """Report which styles are point-in-time and which are not. Never silent.

    This does not raise on static exposures -- trap #4 makes them unavoidable for
    value, quality and growth, and refusing them would mean having no risk model.
    It raises when the split has not been recorded at all, because an exposure
    matrix that has lost track of its own provenance is exactly how a static
    fundamental quietly ends up inside a backtest.
    """
    pit = X.attrs.get("pit")
    if not pit:
        raise AssertionError(
            f"{label}: exposure matrix carries no 'pit' provenance map. Build it "
            f"with factors.build_exposures so each style is labelled "
            f"point-in-time or static (trap #4).")
    rows = [{"factor": f, "point_in_time": bool(v),
             "basis": "prices" if v else "today's statements, held constant"}
            for f, v in pit.items() if f in X.columns]
    out = pd.DataFrame(rows).set_index("factor")
    out.attrs["n_pit"] = int(out["point_in_time"].sum())
    out.attrs["n_static"] = int((~out["point_in_time"]).sum())
    out.attrs["result"] = (f"{out.attrs['n_pit']} point-in-time, "
                           f"{out.attrs['n_static']} static styles")
    return out


def assert_model_vs_sample(weights: pd.Series, model_cov: pd.DataFrame,
                           sample_rets: pd.DataFrame,
                           tolerance_ratio: float = 2.0,
                           label: str = "factor model") -> pd.DataFrame:
    """Cross-check the factor model's risk against the realised return panel.

    The same discipline ``risk.assert_var_agreement`` applies to VaR. A structured
    covariance is meant to be *better conditioned* than the sample, not to describe
    a different portfolio: if the model says 8% and the realised book did 20%, the
    exposures, the factor returns or the alignment are wrong.

    The tolerance is a ratio rather than a difference because the two disagree
    multiplicatively, and it is deliberately loose -- a structured estimate that
    matched the sample exactly would not be adding anything.
    """
    w = weights.reindex(model_cov.index).fillna(0.0)
    model_var = float(w.to_numpy(dtype=float) @ model_cov.to_numpy(dtype=float)
                      @ w.to_numpy(dtype=float))
    model_vol = float(np.sqrt(max(model_var, 0)) * 100)

    cols = [c for c in sample_rets.columns if c in w.index]
    pr = (sample_rets[cols] * w.reindex(cols)).sum(axis=1, min_count=1).dropna()
    realised = float(pr.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)

    if realised <= 0 or model_vol <= 0:
        raise AssertionError(f"{label}: a volatility came back non-positive "
                             f"(model {model_vol:.2f}%, realised {realised:.2f}%)")
    ratio = model_vol / realised
    if ratio > tolerance_ratio or ratio < 1.0 / tolerance_ratio:
        raise AssertionError(
            f"{label}: model volatility {model_vol:.2f}% and realised volatility "
            f"{realised:.2f}% differ by {ratio:.2f}x (tolerance {tolerance_ratio}x). "
            f"The factor model and the return panel describe different books -- "
            f"check exposure alignment and the annualisation of the factor "
            f"covariance.")
    return pd.DataFrame([{
        "model_vol_%": round(model_vol, 3),
        "realised_vol_%": round(realised, 3),
        "ratio": round(ratio, 3),
        "tolerance": tolerance_ratio,
        "n_obs": int(len(pr)),
        "result": "factor model agrees with the realised return panel",
    }])
