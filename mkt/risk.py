"""Portfolio risk: VaR, CVaR, beta, risk contribution, crisis replay, scenarios.

Runs on a **weights vector**, so the model book and a real book from
``data/portfolio.csv`` are the same code path with two inputs. Nothing here
knows or cares which one it was handed.

Two things this module is built to make visible:

* **Where the variance actually comes from.** The largest weight is rarely the
  largest risk contributor. ``risk_contribution`` decomposes total volatility
  into per-name marginal and component terms so the difference is a column in a
  table rather than a surprise in a drawdown.
* **What has already happened to a book like this.** ``replay_episodes`` puts
  today's weights through real dated crises from ``config.STRESS_EPISODES``
  rather than through a simulated shock, and reports which episodes the price
  history actually reaches.

Zero new dependencies: covariance shrinkage is the closed-form Ledoit-Wolf
constant-correlation estimator, and the Cornish-Fisher expansion is arithmetic.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import align, config, fetch, indicators as ind

TRADING_DAYS = ind.TRADING_DAYS


# ==========================================================================
# Covariance
# ==========================================================================
def shrunk_cov(rets: pd.DataFrame, shrinkage: float | None = None) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage towards a constant-correlation target.

    The sample covariance of 50 names on 500 days is badly conditioned: the
    smallest eigenvalues are mostly estimation error, and an optimiser will
    cheerfully pile into whatever direction they describe. Shrinking towards a
    structured target -- every pair assigned the average correlation -- fixes
    that, and the optimal intensity is closed form, so no optimiser and no new
    dependency is needed.

    Pass ``shrinkage`` to override the analytic intensity; the value used is
    recorded in ``.attrs['shrinkage']``.
    """
    X = rets.dropna(how="any")
    t, n = X.shape
    if t < 10 or n < 2:
        cov = rets.cov()
        cov.attrs["shrinkage"] = 0.0
        cov.attrs["method"] = "sample (too few observations to shrink)"
        return cov

    x = X.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    sample = (x.T @ x) / t

    var = np.diag(sample).copy()
    sqrtvar = np.sqrt(var)
    outer = np.outer(sqrtvar, sqrtvar)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, sample / outer, 0.0)
    r_bar = (corr.sum() - n) / (n * (n - 1))

    prior = r_bar * outer
    np.fill_diagonal(prior, var)

    if shrinkage is None:
        # pi-hat: asymptotic variance of the sample covariance entries
        y = x ** 2
        phi_mat = (y.T @ y) / t - sample ** 2
        phi = phi_mat.sum()

        # rho-hat: covariance between the sample estimator and the target
        term1 = ((x ** 3).T @ x) / t
        helpm = (x.T @ x) / t
        help_diag = np.diag(helpm)
        term2 = np.tile(help_diag.reshape(-1, 1), (1, n)) * helpm
        term3 = helpm * np.tile(var.reshape(1, -1), (n, 1))
        term4 = np.tile(var.reshape(-1, 1), (1, n)) * helpm
        theta = term1 - term2 - term3 + term4
        np.fill_diagonal(theta, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(outer > 0, np.outer(1.0 / sqrtvar, sqrtvar), 0.0)
        rho = np.trace(phi_mat) + r_bar * (scale * theta).sum()

        # gamma-hat: squared Frobenius distance from sample to target
        gamma = float(((sample - prior) ** 2).sum())
        kappa = (phi - rho) / gamma if gamma > 0 else 0.0
        shrinkage = float(np.clip(kappa / t, 0.0, 1.0))

    sigma = shrinkage * prior + (1 - shrinkage) * sample
    out = pd.DataFrame(sigma, index=X.columns, columns=X.columns)
    out.attrs["shrinkage"] = float(shrinkage)
    out.attrs["avg_correlation"] = float(r_bar)
    out.attrs["n_obs"] = int(t)
    out.attrs["method"] = "Ledoit-Wolf constant-correlation"
    return out


# ==========================================================================
# Portfolio returns
# ==========================================================================
def portfolio_returns(weights: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """Fixed-weight portfolio return series.

    Weights are held constant, i.e. the book is rebalanced every period. That is
    the right convention for a risk measurement -- it isolates the risk of the
    *stated* book rather than mixing in the drift of an unrebalanced one.
    """
    w = weights.reindex(rets.columns).fillna(0.0)
    return (rets * w).sum(axis=1, min_count=1).dropna().rename("portfolio")


# ==========================================================================
# Value at risk
# ==========================================================================
def hist_var(port_rets: pd.Series, conf: float = 0.95) -> float:
    """Empirical quantile loss. Makes no distributional assumption at all."""
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if len(r) < 20:
        return float("nan")
    return float(-np.percentile(r, (1 - conf) * 100))


def param_var(weights: pd.Series, cov: pd.DataFrame, conf: float = 0.95) -> float:
    """Gaussian VaR from the covariance matrix."""
    from scipy.stats import norm
    w = weights.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    if var < 0 or not np.isfinite(var):
        return float("nan")
    return float(-norm.ppf(1 - conf) * np.sqrt(var))


def cornish_fisher_var(port_rets: pd.Series, conf: float = 0.95) -> float:
    """Gaussian VaR corrected for skew and fat tails.

    Equity books are left-skewed and leptokurtic, so a Gaussian VaR
    systematically understates the loss that matters. The Cornish-Fisher
    expansion adjusts the quantile using the sample's own third and fourth
    moments -- the middle ground between assuming normality and needing enough
    history for a clean empirical tail.
    """
    from scipy.stats import norm
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if len(r) < 30:
        return float("nan")
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    s = float(r.skew())
    k = float(r.kurtosis())                  # pandas returns *excess* kurtosis
    z = float(norm.ppf(1 - conf))
    z_cf = (z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
    return float(-(mu + z_cf * sd))


def cvar(port_rets: pd.Series, conf: float = 0.95) -> float:
    """Expected shortfall: the average loss *given* the VaR level is breached."""
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if len(r) < 20:
        return float("nan")
    cut = np.percentile(r, (1 - conf) * 100)
    tail = r[r <= cut]
    return float("nan") if tail.empty else float(-tail.mean())


def var_table(weights: pd.Series, rets: pd.DataFrame, cov: pd.DataFrame,
              confidences: Sequence[float] = config.VAR_CONFIDENCE,
              capital: float | None = None) -> pd.DataFrame:
    """Every VaR method at every confidence, side by side.

    Three methods disagreeing is information: a large historical-vs-parametric
    gap means the return distribution is not Gaussian, and a large gap in the
    *other* direction usually means the covariance or the return panel is wrong.
    """
    pr = portfolio_returns(weights, rets)
    rows = []
    for c in confidences:
        h, p, cf, es = (hist_var(pr, c), param_var(weights, cov, c),
                        cornish_fisher_var(pr, c), cvar(pr, c))
        row = {"confidence": c, "historical_%": h * 100, "parametric_%": p * 100,
               "cornish_fisher_%": cf * 100, "cvar_%": es * 100,
               "hist_vs_param_pp": (h - p) * 100}
        if capital:
            row["historical_INR"] = h * capital
            row["cvar_INR"] = es * capital
        rows.append(row)
    out = pd.DataFrame(rows).set_index("confidence")
    out.attrs["n_obs"] = int(len(pr))
    out.attrs["horizon"] = "1 trading day"
    return out


def assert_var_agreement(vt: pd.DataFrame, tolerance_pp: float = 1.0,
                         label: str = "book") -> pd.DataFrame:
    """Cross-check: historical and parametric VaR must agree within tolerance.

    They measure the same thing two ways. A large divergence does not mean one
    is 'more conservative' -- it means the covariance matrix and the realised
    return panel describe different portfolios, which is a bug, not a view.
    """
    gap = vt["hist_vs_param_pp"].abs()
    bad = gap[gap > tolerance_pp]
    if len(bad):
        raise AssertionError(
            f"{label}: historical and parametric VaR differ by "
            f"{bad.max():.2f}pp at confidence {list(bad.index)} "
            f"(tolerance {tolerance_pp}pp) -- the covariance matrix and the return "
            f"panel disagree about this portfolio.")
    return pd.DataFrame([{"max_gap_pp": round(float(gap.max()), 3),
                          "tolerance_pp": tolerance_pp,
                          "result": "historical and parametric VaR agree"}])


# ==========================================================================
# Beta and risk decomposition
# ==========================================================================
def portfolio_beta(port_rets: pd.Series, bench_rets: pd.Series) -> dict:
    """Beta, alpha and R-squared against a benchmark."""
    j = pd.concat([port_rets, bench_rets], axis=1, keys=["p", "b"]).dropna()
    if len(j) < 30:
        return {"beta": np.nan, "alpha_ann_%": np.nan, "r2": np.nan, "n": len(j)}
    var_b = float(j["b"].var(ddof=1))
    if var_b <= 0:
        return {"beta": np.nan, "alpha_ann_%": np.nan, "r2": np.nan, "n": len(j)}
    beta = float(j["p"].cov(j["b"]) / var_b)
    alpha = float(j["p"].mean() - beta * j["b"].mean())
    r = float(j["p"].corr(j["b"]))
    return {"beta": beta, "alpha_ann_%": alpha * TRADING_DAYS * 100,
            "r2": r ** 2, "n": int(len(j))}


def name_betas(rets: pd.DataFrame, bench_rets: pd.Series) -> pd.Series:
    """Per-name beta to the benchmark -- the input to beta-matched sizing."""
    b = bench_rets.reindex(rets.index)
    var_b = b.var(ddof=1)
    if not np.isfinite(var_b) or var_b <= 0:
        return pd.Series(np.nan, index=rets.columns)
    return (rets.apply(lambda s: s.cov(b)) / var_b).rename("beta")


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Marginal and component contribution to portfolio volatility.

    Component contributions sum exactly to total volatility, which is what makes
    the percentage column meaningful. The name that drives the variance is
    rarely the largest weight -- it is the one whose covariance with everything
    else is highest.
    """
    idx = cov.index
    w = weights.reindex(idx).fillna(0.0)
    S = cov.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    var = float(wv @ S @ wv)
    if var <= 0 or not np.isfinite(var):
        return pd.DataFrame(index=idx)
    sd = np.sqrt(var)

    mcr = (S @ wv) / sd                       # d sigma / d w_i
    ccr = wv * mcr                            # sums to sigma
    ann = np.sqrt(TRADING_DAYS) * 100
    out = pd.DataFrame({
        "weight": w,
        "weight_%_of_gross": w.abs() / w.abs().sum() * 100 if w.abs().sum() else np.nan,
        "marginal_%": mcr * ann,
        "component_%": ccr * ann,
        "pct_of_risk": ccr / sd * 100,
    }, index=idx)
    out.attrs["portfolio_vol_%"] = float(sd * ann)
    out.attrs["sum_check"] = float(ccr.sum() * ann)
    return out.sort_values("pct_of_risk", ascending=False)


def concentration(cov: pd.DataFrame, weights: pd.Series | None = None) -> dict:
    """Eigenvalue concentration and the effective number of independent bets.

    A book of twenty names driven by one factor is one bet. The correlation
    matrix's leading eigenvalue says how much of the cross-sectional variance is
    that single factor; the entropy-based effective count says how many genuinely
    distinct bets remain.
    """
    sd = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    outer = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, cov.to_numpy(dtype=float) / outer, 0.0)
    vals = np.linalg.eigvalsh(corr)[::-1]
    vals = np.clip(vals, 0, None)
    total = vals.sum()
    p = vals / total if total > 0 else vals
    nz = p[p > 1e-12]
    out = {
        "n_assets": int(cov.shape[0]),
        "pc1_variance_%": float(p[0] * 100) if len(p) else np.nan,
        "pc3_variance_%": float(p[:3].sum() * 100) if len(p) >= 3 else np.nan,
        "effective_bets_entropy": float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else np.nan,
        "effective_bets_herfindahl": float(1.0 / (p ** 2).sum()) if total > 0 else np.nan,
        "avg_correlation": float((corr.sum() - cov.shape[0]) /
                                 (cov.shape[0] * (cov.shape[0] - 1)))
        if cov.shape[0] > 1 else np.nan,
    }
    if weights is not None:
        w = weights.reindex(cov.index).fillna(0.0).abs()
        out["effective_positions"] = (float(1.0 / ((w / w.sum()) ** 2).sum())
                                      if w.sum() > 0 else np.nan)
    out["eigenvalues"] = vals
    return out


# ==========================================================================
# Crisis replay
# ==========================================================================
def replay_episodes(weights: pd.Series, panel: pd.DataFrame,
                    episodes: Mapping[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Put today's book through real dated crises.

    Fixed weights, actual prices. Coverage is reported per episode rather than
    silently dropping the ones the panel does not reach -- a 10-year panel does
    not contain the GFC, and pretending otherwise by returning fewer rows would
    hide that.

    ``names_covered`` matters: an episode where only half the book existed is a
    partial answer, and it says so.
    """
    episodes = dict(episodes or config.STRESS_EPISODES)
    w = weights[weights != 0]
    rows = []
    for name, (start, end) in episodes.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        window = panel.loc[(panel.index >= s) & (panel.index <= e)]
        if len(window) < 2:
            rows.append({"episode": name, "start": s.date(), "end": e.date(),
                         "sessions": len(window), "names_covered": 0,
                         "book_return_%": np.nan, "worst_name": "",
                         "worst_name_%": np.nan, "note": "outside the panel's history"})
            continue
        avail = [t for t in w.index if t in window.columns
                 and window[t].first_valid_index() is not None
                 and window[t].notna().sum() >= 2]
        if not avail:
            rows.append({"episode": name, "start": s.date(), "end": e.date(),
                         "sessions": len(window), "names_covered": 0,
                         "book_return_%": np.nan, "worst_name": "",
                         "worst_name_%": np.nan, "note": "no book name has prices here"})
            continue
        sub = window[avail].ffill()
        name_ret = (sub.iloc[-1] / sub.iloc[0] - 1)
        ww = w.reindex(avail)
        book_ret = float((ww * name_ret).sum())
        # Renormalise to the covered gross so a half-covered episode is not
        # reported as a small loss simply because half the book was missing.
        covered_gross = float(ww.abs().sum())
        full_gross = float(w.abs().sum())
        worst = (ww * name_ret).idxmin()
        rows.append({
            "episode": name, "start": s.date(), "end": e.date(),
            "sessions": int(len(window)), "names_covered": len(avail),
            "names_in_book": int(len(w)),
            "book_return_%": book_ret * 100,
            "scaled_to_full_gross_%": (book_ret * full_gross / covered_gross * 100
                                       if covered_gross > 0 else np.nan),
            "worst_name": str(worst),
            "worst_name_%": float((ww * name_ret).min() * 100),
            "note": "" if len(avail) == len(w) else f"only {len(avail)}/{len(w)} names listed",
        })
    out = pd.DataFrame(rows).set_index("episode")
    out.attrs["episodes_covered"] = int(out["book_return_%"].notna().sum())
    out.attrs["episodes_total"] = len(out)
    return out


# ==========================================================================
# Forward scenarios
# ==========================================================================
def factor_returns(period: str = config.BACKTEST_PERIOD,
                   factors: Mapping[str, str] | None = None,
                   calendar: str = "NSE") -> pd.DataFrame:
    """Daily returns of the scenario factors, aligned to the equity calendar."""
    factors = dict(factors or config.SCENARIO_FACTORS)
    cols = {}
    for key, ticker in factors.items():
        try:
            cols[key] = fetch.price_history(ticker, period=period, quiet=True)["Close"]
        except Exception as exc:                        # noqa: BLE001
            print(f"  [risk] factor {key} ({ticker}) unavailable: "
                  f"{type(exc).__name__}: {str(exc)[:70]}")
    if not cols:
        return pd.DataFrame()
    px = align.align_to_calendar(pd.DataFrame(cols), calendar=calendar)
    return align.to_returns(px).dropna(how="all")


def factor_betas(rets: pd.DataFrame, fac_rets: pd.DataFrame) -> pd.DataFrame:
    """Multivariate OLS betas of each name on the scenario factors.

    Reported with R-squared attached, because a beta from a regression that
    explains 3% of the variance is a number, not a relationship, and the
    scenario built on it should be read accordingly.
    """
    rows = {}
    for t in rets.columns:
        j = pd.concat([rets[t], fac_rets], axis=1).dropna()
        if len(j) < 60:
            continue
        y = j.iloc[:, 0].to_numpy(dtype=float)
        X = j.iloc[:, 1:].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        row = dict(zip(fac_rets.columns, coef[1:]))
        row["alpha"] = coef[0]
        row["r2"] = r2
        row["n_obs"] = len(j)
        rows[t] = row
    return pd.DataFrame(rows).T


def scenario_shock(weights: pd.Series, betas: pd.DataFrame,
                   scenarios: Mapping[str, Mapping[str, float]] | None = None,
                   capital: float | None = None) -> pd.DataFrame:
    """Map factor shocks through estimated betas into a book-level P&L.

    These are estimates on top of estimates: a beta from a linear regression,
    applied to a shock nobody has observed. The mean R-squared is reported next
    to every scenario so the reader can discount accordingly.
    """
    scenarios = dict(scenarios or config.SCENARIOS)
    w = weights[weights != 0]
    common = [t for t in w.index if t in betas.index]
    if not common:
        return pd.DataFrame()
    ww = w.reindex(common)
    b = betas.loc[common]

    rows = []
    for name, shocks in scenarios.items():
        used = {k: v for k, v in shocks.items() if k in b.columns}
        if not used:
            rows.append({"scenario": name, "book_return_%": np.nan,
                         "note": "no factor in this scenario has an estimated beta"})
            continue
        impact = sum(b[k] * v for k, v in used.items())
        book = float((ww * impact).sum())
        contrib = (ww * impact).sort_values()
        row = {
            "scenario": name,
            "book_return_%": book * 100,
            "factors": ", ".join(f"{k} {v:+.0%}" for k, v in used.items()),
            "mean_r2": float(b["r2"].mean()),
            "worst_name": str(contrib.index[0]),
            "worst_name_%": float(contrib.iloc[0] * 100),
            "best_name": str(contrib.index[-1]),
            "best_name_%": float(contrib.iloc[-1] * 100),
        }
        if capital:
            row["book_INR"] = book * capital
        rows.append(row)
    out = pd.DataFrame(rows).set_index("scenario")
    out.attrs["names_used"] = len(common)
    out.attrs["names_in_book"] = int(len(w))
    return out


# ==========================================================================
# Drawdown summary
# ==========================================================================
def drawdown_summary(port_rets: pd.Series) -> dict:
    """Realised risk of the fixed-weight book over the sample."""
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    dd = ind.drawdown(eq)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)
    ann_ret = float((eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1) * 100)
    return {
        "n_days": int(len(r)),
        "ann_return_%": ann_ret,
        "ann_vol_%": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown_%": float(dd.min()),
        "current_drawdown_%": float(dd.iloc[-1]),
        "worst_day_%": float(r.min() * 100),
        "best_day_%": float(r.max() * 100),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
    }


# ==========================================================================
# VaR BACKTESTING -- model validation
# ==========================================================================
# The original module computed VaR four ways and cross-checked two of them
# against each other. That catches a covariance/panel mismatch and nothing else.
# It cannot catch the failure that matters: a VaR model that is simply wrong
# about the tail, consistently, for a year.
#
# A VaR number is a falsifiable forecast -- "losses will exceed this on 5 days in
# 100" -- and a forecast nobody scores is not a risk measure, it is a decoration.
# Everything below scores it. This is the standard regulatory battery, and it is
# what separates a risk report from a risk system.


def var_exceptions(port_rets: pd.Series, var_series: pd.Series | float,
                   conf: float = 0.95) -> pd.DataFrame:
    """Line up realised losses against the VaR forecast that preceded them.

    ``var_series`` may be a single number -- a static VaR held all year -- or a
    series of daily forecasts, which is the honest version because a real risk
    system re-estimates. Either way the forecast used on day t must be the one
    known at t-1, so a Series input is shifted.
    """
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if np.isscalar(var_series):
        v = pd.Series(float(var_series), index=r.index)
    else:
        v = pd.to_numeric(var_series, errors="coerce").reindex(r.index).shift(1)
    j = pd.concat([r.rename("ret"), v.rename("var")], axis=1).dropna()
    j["loss"] = -j["ret"]
    j["exception"] = j["loss"] > j["var"]
    j.attrs["confidence"] = conf
    j.attrs["n_obs"] = int(len(j))
    j.attrs["n_exceptions"] = int(j["exception"].sum())
    j.attrs["expected_exceptions"] = float(len(j) * (1 - conf))
    return j


def kupiec_pof(n_obs: int, n_exceptions: int, conf: float = 0.95) -> dict:
    """Kupiec unconditional coverage test: is the exception *rate* right?

    The likelihood ratio of the observed exception count against the expected one,
    distributed chi-squared with one degree of freedom. A model that breaches on
    9% of days at a 95% VaR is not conservative and is not unlucky -- it is
    rejected, and this is the test that says so with a p-value rather than an
    opinion.

    Its known weakness is that it says nothing about *when* the breaches happened,
    which is why it is never quoted here without Christoffersen alongside it.
    """
    from scipy import stats
    T, x = int(n_obs), int(n_exceptions)
    p = 1.0 - float(conf)
    if T < 30:
        return {"lr_pof": np.nan, "p_value": np.nan, "note": "too few observations"}
    pi_hat = x / T
    if x == 0:
        lr = -2.0 * (T * np.log(1 - p))
    elif x == T:
        lr = -2.0 * (T * np.log(p))
    else:
        ll_null = (T - x) * np.log(1 - p) + x * np.log(p)
        ll_alt = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
        lr = -2.0 * (ll_null - ll_alt)
    pv = float(stats.chi2.sf(lr, 1))
    return {
        "test": "Kupiec POF (unconditional coverage)",
        "n_obs": T,
        "n_exceptions": x,
        "expected_exceptions": T * p,
        "exception_rate_%": pi_hat * 100,
        "expected_rate_%": p * 100,
        "lr_pof": float(lr),
        "p_value": pv,
        "reject_5pct": bool(pv < 0.05),
        "verdict": ("coverage rejected -- the VaR level is wrong" if pv < 0.05
                    else "coverage acceptable"),
    }


def christoffersen_independence(exceptions: pd.Series) -> dict:
    """Are the breaches independent, or do they cluster?

    This is the test that matters more, and the one almost nobody runs. A model
    can produce exactly the right *number* of exceptions and still be useless if
    they all arrive in the same fortnight -- because that is precisely the
    fortnight the book needed the forecast to work.

    Clustering means the model is not adapting to volatility: it under-forecasts
    through a whole stressed period, takes its breaches in a run, and looks
    perfectly calibrated over the year. Kupiec passes it. This does not.
    """
    from scipy import stats
    e = pd.Series(exceptions).astype(bool).to_numpy()
    if len(e) < 30:
        return {"lr_ind": np.nan, "p_value": np.nan, "note": "too few observations"}

    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum(~prev & ~cur))
    n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur))
    n11 = int(np.sum(prev & cur))

    denom0, denom1 = n00 + n01, n10 + n11
    if denom0 == 0 or denom1 == 0 or (n01 + n11) == 0:
        return {"test": "Christoffersen independence", "lr_ind": 0.0,
                "p_value": 1.0, "n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "reject_5pct": False,
                "verdict": "too few exceptions to test independence"}

    pi01 = n01 / denom0
    pi11 = n11 / denom1
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p_, n_yes, n_no):
        if p_ <= 0 or p_ >= 1:
            return 0.0
        return n_yes * np.log(p_) + n_no * np.log(1 - p_)

    ll_null = _ll(pi, n01 + n11, n00 + n10)
    ll_alt = _ll(pi01, n01, n00) + _ll(pi11, n11, n10)
    lr = -2.0 * (ll_null - ll_alt)
    pv = float(stats.chi2.sf(lr, 1))
    return {
        "test": "Christoffersen independence",
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p_breach_after_calm_%": pi01 * 100,
        "p_breach_after_breach_%": pi11 * 100,
        "lr_ind": float(lr),
        "p_value": pv,
        "reject_5pct": bool(pv < 0.05),
        "verdict": ("breaches cluster -- the model does not react to volatility"
                    if pv < 0.05 else "breaches are independent"),
    }


def christoffersen_cc(n_obs: int, exceptions: pd.Series, conf: float = 0.95) -> dict:
    """Joint conditional coverage: right rate *and* independent. chi2(2).

    The sum of the two likelihood ratios. This is the single number to report,
    with the components underneath it -- because when it rejects, the interesting
    question is immediately which half failed.
    """
    from scipy import stats
    pof = kupiec_pof(n_obs, int(pd.Series(exceptions).astype(bool).sum()), conf)
    ind = christoffersen_independence(exceptions)
    if not np.isfinite(pof.get("lr_pof", np.nan)) or not np.isfinite(ind.get("lr_ind", np.nan)):
        return {"lr_cc": np.nan, "p_value": np.nan, "note": "components unavailable"}
    lr = float(pof["lr_pof"] + ind["lr_ind"])
    pv = float(stats.chi2.sf(lr, 2))
    return {
        "test": "Christoffersen conditional coverage",
        "lr_pof": pof["lr_pof"], "lr_ind": ind["lr_ind"], "lr_cc": lr,
        "p_value": pv,
        "reject_5pct": bool(pv < 0.05),
        "failed_component": ("coverage" if pof["reject_5pct"] and not ind["reject_5pct"]
                             else "independence" if ind["reject_5pct"] and not pof["reject_5pct"]
                             else "both" if pof["reject_5pct"] and ind["reject_5pct"]
                             else "none"),
        "verdict": ("VaR model rejected" if pv < 0.05 else "VaR model not rejected"),
    }


def basel_traffic_light(n_obs: int, n_exceptions: int,
                        conf: float = 0.99) -> dict:
    """The Basel zone, derived from the binomial null rather than hardcoded.

    The familiar boundaries -- green 0-4, yellow 5-9, red 10+ -- are not
    arbitrary constants. They are quantiles of ``Binomial(250, 0.01)``: green is
    where the cumulative probability of that many exceptions is still below 95%,
    red is where it passes 99.99%. Deriving them that way rather than writing 4
    and 9 into the code is what lets the test work at a different sample length
    or a different confidence level.

    That distinction is not academic. An earlier draft of this function scaled the
    250-day boundaries linearly and applied them to a **95%** VaR, where 5% of
    1,248 days is 62 expected exceptions and a threshold of 45 is nonsense -- it
    put a perfectly well-calibrated model in the red zone. Any implementation that
    does not reproduce 4 and 9 at ``T=250, p=0.01`` is wrong, which is the check
    ``self_test`` below performs.

    Zones: green -- model accepted; yellow -- questioned, and a capital add-on
    applies; red -- presumed wrong.
    """
    from scipy import stats
    T, x = int(n_obs), int(n_exceptions)
    p = 1.0 - float(conf)
    if T < 50:
        return {"zone": "n/a", "note": f"needs 50+ observations, has {T}"}

    # Largest exception count still inside each zone, from the binomial CDF.
    ks = np.arange(0, T + 1)
    cdf = stats.binom.cdf(ks, T, p)
    green_max = int(ks[cdf < 0.95].max()) if (cdf < 0.95).any() else 0
    yellow_max = int(ks[cdf < 0.9999].max()) if (cdf < 0.9999).any() else green_max

    if x <= green_max:
        zone, mult = "green", 3.00
    elif x <= yellow_max:
        zone = "yellow"
        # Standard Basel add-on schedule, indexed by position within the band so
        # it degrades sensibly when the band is not the canonical 5..9.
        span = max(yellow_max - green_max, 1)
        steps = [3.40, 3.50, 3.65, 3.75, 3.85]
        pos = int(np.clip((x - green_max - 1) / span * len(steps), 0, len(steps) - 1))
        mult = steps[pos]
    else:
        zone, mult = "red", 4.00

    return {
        "zone": zone,
        "n_obs": T,
        "n_exceptions": x,
        "expected_exceptions": T * p,
        "green_up_to": green_max,
        "yellow_up_to": yellow_max,
        "capital_multiplier": mult,
        "confidence": conf,
        "basis": "binomial quantiles (95% / 99.99%) of the exception count",
        "verdict": {"green": "model accepted",
                    "yellow": "model questioned -- capital add-on applies",
                    "red": "model presumed wrong"}[zone],
    }


def var_backtest(port_rets: pd.Series, var_series: pd.Series | float,
                 conf: float = 0.95) -> pd.DataFrame:
    """The full battery on one VaR series. This is the table to put in the report.

    Coverage, independence, joint, Basel zone, and the average severity of the
    breaches that did happen -- because two models with identical exception counts
    are not equally good if one of them is wrong by 0.3pp and the other by 4pp.
    """
    ex = var_exceptions(port_rets, var_series, conf)
    T = int(len(ex))
    x = int(ex["exception"].sum())
    pof = kupiec_pof(T, x, conf)
    ind = christoffersen_independence(ex["exception"])
    cc = christoffersen_cc(T, ex["exception"], conf)
    basel = basel_traffic_light(T, x, conf)

    breaches = ex[ex["exception"]]
    severity = ((breaches["loss"] - breaches["var"]) * 100) if len(breaches) else pd.Series(dtype=float)

    rows = [{
        "confidence": conf,
        "n_obs": T,
        "n_exceptions": x,
        "expected_exceptions": round(T * (1 - conf), 1),
        "exception_rate_%": round(x / T * 100, 2) if T else np.nan,
        "kupiec_p": round(pof.get("p_value", np.nan), 4),
        "independence_p": round(ind.get("p_value", np.nan), 4),
        "conditional_coverage_p": round(cc.get("p_value", np.nan), 4),
        "failed_component": cc.get("failed_component", ""),
        "basel_zone": basel.get("zone", ""),
        "mean_breach_severity_pp": round(float(severity.mean()), 3) if len(severity) else np.nan,
        "worst_breach_severity_pp": round(float(severity.max()), 3) if len(severity) else np.nan,
        "verdict": cc.get("verdict", ""),
    }]
    out = pd.DataFrame(rows).set_index("confidence")
    out.attrs["kupiec"] = pof
    out.attrs["independence"] = ind
    out.attrs["conditional_coverage"] = cc
    out.attrs["basel"] = basel
    out.attrs["exceptions_frame"] = ex
    return out


def rolling_var_forecast(port_rets: pd.Series, window: int = 252,
                         conf: float = 0.95, method: str = "historical") -> pd.Series:
    """A genuine out-of-sample VaR forecast series, for the backtest to score.

    Each day's forecast uses only the preceding ``window`` days. That is the whole
    point -- a VaR backtest run against a VaR computed on the full sample is
    scoring the model on data it has already seen, which is the same look-ahead
    error trap #4 guards against, committed inside the risk system instead of the
    alpha model.
    """
    from scipy.stats import norm
    r = pd.to_numeric(port_rets, errors="coerce").dropna()
    if method == "historical":
        f = r.rolling(window).quantile(1 - conf).mul(-1)
    elif method == "parametric":
        f = -(r.rolling(window).mean() + norm.ppf(1 - conf) * r.rolling(window).std(ddof=1))
    elif method == "ewma":
        lam = 0.94
        var = r.ewm(alpha=1 - lam).var(bias=True)
        f = -(norm.ppf(1 - conf) * np.sqrt(var))
    else:
        raise ValueError(f"rolling_var_forecast: unknown method {method!r}")
    out = f.rename(f"var_{method}_{int(conf * 100)}")
    out.attrs["window"] = window
    out.attrs["method"] = method
    out.attrs["confidence"] = conf
    return out


def var_model_comparison(port_rets: pd.Series, window: int = 252,
                         conf: float = 0.95) -> pd.DataFrame:
    """Score every VaR method against the same realised losses, side by side.

    Which method to trust is an empirical question with a testable answer, and
    this is the table that answers it. Historical VaR is slow to react and clusters
    its breaches; EWMA reacts fast and can be jumpy. The right one is whichever
    passes conditional coverage on this book.
    """
    rows = []
    for method in ("historical", "parametric", "ewma"):
        try:
            f = rolling_var_forecast(port_rets, window, conf, method)
            bt = var_backtest(port_rets, f, conf)
            row = bt.reset_index().iloc[0].to_dict()
            row["method"] = method
            row["mean_var_%"] = float(f.dropna().mean() * 100)
            rows.append(row)
        except Exception as exc:                              # noqa: BLE001
            rows.append({"method": method, "verdict": f"failed: {type(exc).__name__}"})
    cols = ["method", "mean_var_%", "n_exceptions", "expected_exceptions",
            "exception_rate_%", "kupiec_p", "independence_p",
            "conditional_coverage_p", "basel_zone", "verdict"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]].set_index("method")


def bias_test(port_rets: pd.Series, vol_forecast: pd.Series,
              window: int = 252) -> dict:
    """Ex-ante versus ex-post: is the volatility forecast the right size?

    Standardise each realised return by the volatility that was forecast for it.
    If the forecast is honest, the standardised series has a standard deviation of
    1. Above 1 the model under-forecasts risk; below 1 it over-forecasts and the
    book is being throttled for no reason.

    Reported monthly by real risk teams under exactly this name, and absent from
    almost every backtest. The ex-ante number is the one used to size positions,
    so a bias of 1.4 means every position is 40% larger than intended.
    """
    r = pd.to_numeric(port_rets, errors="coerce")
    v = pd.to_numeric(vol_forecast, errors="coerce").reindex(r.index).shift(1)
    j = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
    j = j[j["v"] > 0]
    if len(j) < 60:
        return {"bias": np.nan, "note": f"needs 60+ paired observations, has {len(j)}"}
    z = j["r"] / j["v"]
    bias = float(z.std(ddof=1))
    n = len(z)
    # Standard error of a standard deviation estimate, normal approximation.
    se = 1.0 / np.sqrt(2.0 * (n - 1))
    return {
        "bias": bias,
        "n_obs": n,
        "se": se,
        "t_stat_vs_1": (bias - 1.0) / se,
        "ci_low": bias - 1.96 * se,
        "ci_high": bias + 1.96 * se,
        "significant": bool(abs(bias - 1.0) > 1.96 * se),
        "mean_forecast_vol_ann_%": float(j["v"].mean() * np.sqrt(TRADING_DAYS) * 100),
        "realised_vol_ann_%": float(j["r"].std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
        "verdict": ("under-forecasting risk -- positions are larger than intended"
                    if bias > 1.0 + 1.96 * se else
                    "over-forecasting risk -- the book is throttled unnecessarily"
                    if bias < 1.0 - 1.96 * se else
                    "forecast is unbiased"),
    }


def assert_var_model_validated(bt: pd.DataFrame, label: str = "VaR model",
                               require_zone: str = "yellow") -> pd.DataFrame:
    """Regression guard: refuse a VaR number from a model that failed its backtest.

    ``require_zone`` is the worst Basel zone tolerated -- ``"green"`` insists the
    model is accepted outright, ``"yellow"`` allows a questioned model through with
    the failure named, and nothing lets a red zone pass.

    This is the guard that closes the loop. The chain already refuses a
    contaminated calendar, an untracked provenance tag, a look-ahead panel and an
    untradeable short leg. Refusing a falsified risk forecast belongs in the same
    list, and its absence was the largest hole in the risk layer.
    """
    if bt.empty:
        raise AssertionError(f"{label}: empty backtest table")
    order = {"green": 0, "yellow": 1, "red": 2}
    row = bt.iloc[0]
    zone = str(row.get("basel_zone", "red"))
    if order.get(zone, 2) > order.get(require_zone, 1):
        raise AssertionError(
            f"{label}: VaR backtest lands in the {zone.upper()} zone "
            f"({int(row['n_exceptions'])} exceptions against "
            f"{row['expected_exceptions']} expected over {int(row['n_obs'])} days). "
            f"The model's tail forecast has been falsified on this book -- do not "
            f"size against it until it is re-estimated.")
    if str(row.get("failed_component", "none")) == "both":
        raise AssertionError(
            f"{label}: VaR model fails both coverage and independence "
            f"(Kupiec p={row['kupiec_p']}, independence p={row['independence_p']}). "
            f"Wrong level and clustered breaches together mean the model is not "
            f"reacting to volatility at all.")
    return pd.DataFrame([{
        "basel_zone": zone,
        "n_exceptions": int(row["n_exceptions"]),
        "expected": row["expected_exceptions"],
        "kupiec_p": row["kupiec_p"],
        "independence_p": row["independence_p"],
        "failed_component": row.get("failed_component", "none"),
        "result": f"VaR model survives backtesting at the {require_zone} bar",
    }])


# ==========================================================================
# Marginal and incremental VaR
# ==========================================================================
def marginal_var(weights: pd.Series, cov: pd.DataFrame,
                 conf: float = 0.95, capital: float | None = None) -> pd.DataFrame:
    """Per-name marginal and incremental VaR.

    Two different questions that get confused constantly:

    * **marginal** -- the derivative of book VaR with respect to a name's weight.
      What a *small* change costs. Sums (weighted) to total VaR.
    * **incremental** -- what actually happens to VaR if the position is removed
      entirely. Not the same number, because removing a position changes the
      correlation structure the rest of the book sits in, and for a hedge the
      incremental figure can be *positive* -- taking it off raises risk.

    A book is trimmed on incremental VaR and traded on marginal VaR.
    """
    from scipy.stats import norm
    idx = cov.index
    w = weights.reindex(idx).fillna(0.0)
    S = cov.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    z = -norm.ppf(1 - conf)

    var_total = float(wv @ S @ wv)
    if var_total <= 0:
        return pd.DataFrame(index=idx)
    sd = np.sqrt(var_total)
    total_var = z * sd

    mvar = z * (S @ wv) / sd
    rows = {"weight": w, "marginal_var_%": mvar * 100,
            "component_var_%": wv * mvar * 100}

    inc = []
    for i in range(len(idx)):
        w2 = wv.copy()
        w2[i] = 0.0
        v2 = float(w2 @ S @ w2)
        inc.append(total_var - z * np.sqrt(max(v2, 0.0)))
    rows["incremental_var_%"] = np.asarray(inc) * 100

    out = pd.DataFrame(rows, index=idx)
    out["pct_of_var"] = out["component_var_%"] / (total_var * 100) * 100
    if capital:
        out["component_var_INR"] = out["component_var_%"] / 100 * capital
        out["incremental_var_INR"] = out["incremental_var_%"] / 100 * capital
    out.attrs["total_var_%"] = float(total_var * 100)
    out.attrs["confidence"] = conf
    out.attrs["sum_check_%"] = float(out["component_var_%"].sum())
    out.attrs["n_hedges"] = int((out["incremental_var_%"] < 0).sum())
    return out.sort_values("component_var_%", ascending=False)


def liquidity_adjusted_var(weights: pd.Series, cov: pd.DataFrame,
                           adv_value: pd.Series,
                           capital: float = config.DEFAULT_CAPITAL,
                           participation: float = 0.20,
                           conf: float = 0.95) -> pd.DataFrame:
    """VaR over the horizon it would actually take to get out.

    One-day VaR silently assumes the book can be liquidated in one day. For a
    position that is 60% of a day's volume at a tradeable participation rate, the
    real exit horizon is three days and the risk over that horizon is roughly
    sqrt(3) times larger.

    The horizon used is the position's own days-to-liquidate, so an illiquid name
    is charged for its illiquidity instead of being averaged in with a mega-cap.
    ``portfolio.liquidity_check`` already reports days-to-trade; this converts that
    into the number that belongs in a risk limit.
    """
    from scipy.stats import norm
    idx = cov.index
    w = weights.reindex(idx).fillna(0.0)
    adv = pd.to_numeric(adv_value.reindex(idx), errors="coerce")
    sd_i = pd.Series(np.sqrt(np.diag(cov.to_numpy(dtype=float))), index=idx)

    value = w.abs() * capital
    daily_capacity = adv * participation
    days = (value / daily_capacity.replace(0, np.nan)).clip(lower=1.0)
    z = -norm.ppf(1 - conf)

    out = pd.DataFrame({
        "weight": w,
        "position_value": value,
        "adv_value": adv,
        "days_to_liquidate": days,
        "var_1d_%": z * sd_i * w.abs() * 100,
        "var_liq_%": z * sd_i * w.abs() * np.sqrt(days) * 100,
    })
    out["liquidity_add_on_%"] = out["var_liq_%"] - out["var_1d_%"]

    wv = w.to_numpy(dtype=float)
    book_1d = z * float(np.sqrt(max(wv @ cov.to_numpy(dtype=float) @ wv, 0.0)))
    horizon = float(np.nanpercentile(days.dropna(), 95)) if days.notna().any() else 1.0
    out.attrs["book_var_1d_%"] = book_1d * 100
    out.attrs["book_var_liq_%"] = book_1d * np.sqrt(horizon) * 100
    out.attrs["p95_days_to_liquidate"] = horizon
    out.attrs["participation_assumed"] = participation
    out.attrs["worst_name"] = str(days.idxmax()) if days.notna().any() else ""
    return out.sort_values("days_to_liquidate", ascending=False)
