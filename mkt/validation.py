"""Statistical validation: multiple testing, deflated Sharpe, PBO, purged CV.

Why this module exists
----------------------
Notebook 07 already does the hard, unfashionable thing -- it tests the screen and
is allowed to say no. But its own README names a problem it does not then correct
for: *"fifteen correlated tests on fifty names is exactly the selection problem
the notebook warns about."* Naming a selection problem is not the same as
adjusting for it.

Six of fifteen factors cleared |t| > 2 in that run. Under the null, at a 5% level,
you expect 0.75 of fifteen to clear by chance alone -- but you also expect the
*largest* |t| among fifteen correlated tests to sit near 2.3 with no signal
present at all. Reporting the best t-statistic from a family of tests as though it
were a single pre-registered test is the single most common way a backtest lies,
and it lies most convincingly when the researcher is honest about everything else.

So this module supplies the four corrections that turn "the strongest factor had
t = 2.6" into a statement that survives contact with a referee:

1. **Family-wise and false-discovery control** -- Bonferroni and Benjamini-Hochberg
   over the whole factor family, so the number of discoveries is adjusted for the
   number of chances taken.
2. **The deflated Sharpe ratio** -- Bailey and Lopez de Prado's correction, which
   asks what the best Sharpe would have been under the null *given how many
   variants were tried*, and how non-normal the returns were.
3. **Probability of backtest overfitting** -- combinatorially symmetric
   cross-validation: does the configuration that wins in-sample keep winning out
   of sample, or does it land below median?
4. **Purged and embargoed splits** -- because a 126-day forward return sampled
   monthly leaks across any naive train/test boundary. Newey-West fixes the
   *standard error*; it does not fix a train set that contains the test set's
   future.

None of this rescues a signal that is not there. It exists so that a signal that
*is* there can be believed, and so the verdict on one that is not cannot later be
overturned by someone quietly running the search again.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config

EULER_GAMMA = 0.5772156649015329


# ==========================================================================
# Multiple testing
# ==========================================================================
def p_from_t(t: float | pd.Series, dof: int | None = None) -> float | pd.Series:
    """Two-sided p-value from a t-statistic.

    Uses the t distribution when ``dof`` is given and the normal otherwise. At the
    sample sizes here -- roughly 100 monthly rebalances -- the two differ in the
    third decimal, which matters only near a threshold, which is exactly where
    decisions get made.
    """
    from scipy import stats
    arr = np.abs(np.asarray(t, dtype=float))
    if dof is not None and dof > 2:
        p = 2.0 * stats.t.sf(arr, dof)
    else:
        p = 2.0 * stats.norm.sf(arr)
    return pd.Series(p, index=t.index, name="p_value") if isinstance(t, pd.Series) else float(p)


def benjamini_hochberg(p: pd.Series, alpha: float | None = None) -> pd.DataFrame:
    """Benjamini-Hochberg step-up: control the false discovery rate.

    FDR rather than family-wise error because the question here is not "is any
    factor real" but "of the factors I am about to act on, what share are noise".
    Bonferroni answers the first and is brutally conservative on fifteen
    correlated tests; BH answers the second, which is the one that sizes a book.

    Returns the ranked table with adjusted p-values and the reject decision, so
    the discoveries can be read off directly.
    """
    a = config.FDR_ALPHA if alpha is None else float(alpha)
    s = pd.to_numeric(p, errors="coerce").dropna().sort_values()
    m = len(s)
    if m == 0:
        return pd.DataFrame()
    rank = np.arange(1, m + 1)
    crit = rank / m * a
    below = s.to_numpy() <= crit
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    reject = pd.Series(rank <= k, index=s.index)

    # Step-up adjusted p-values: monotone from the largest down.
    adj = np.minimum.accumulate((s.to_numpy() * m / rank)[::-1])[::-1]
    out = pd.DataFrame({
        "p_value": s,
        "rank": rank,
        "bh_critical": crit,
        "p_adj_bh": np.clip(adj, 0, 1),
        "p_adj_bonferroni": np.clip(s.to_numpy() * m, 0, 1),
        "reject_bh": reject,
        "reject_bonferroni": s <= a / m,
    })
    out.attrs["alpha"] = a
    out.attrs["n_tests"] = m
    out.attrs["n_discoveries_bh"] = int(reject.sum())
    out.attrs["n_discoveries_bonferroni"] = int((s <= a / m).sum())
    out.attrs["n_naive"] = int((s <= a).sum())
    out.attrs["expected_false_positives_naive"] = m * a
    return out


def factor_family_test(factor_table: pd.DataFrame, t_col: str = "t_stat",
                       n_col: str = "n_periods",
                       alpha: float | None = None) -> pd.DataFrame:
    """Apply multiple-testing control to a whole ``factor_ic_table``.

    Drops straight onto the output of ``backtest.factor_ic_table``. The
    ``survives`` column is the honest one to quote: a factor that clears |t| > 2
    on its own but not after correction has not been shown to work, it has been
    shown to be the best of fifteen tries.
    """
    tbl = factor_table.copy()
    if t_col not in tbl.columns:
        raise KeyError(f"factor_family_test: no {t_col!r} column; "
                       f"have {list(tbl.columns)}")
    dof = (int(pd.to_numeric(tbl[n_col], errors="coerce").median() - 1)
           if n_col in tbl.columns else None)
    tbl["p_value"] = p_from_t(pd.to_numeric(tbl[t_col], errors="coerce"), dof)
    bh = benjamini_hochberg(tbl["p_value"].dropna(), alpha)
    if bh.empty:
        return tbl

    out = tbl.join(bh[["p_adj_bh", "p_adj_bonferroni",
                       "reject_bh", "reject_bonferroni"]], how="left")
    out["naive_significant"] = pd.to_numeric(out[t_col], errors="coerce").abs() > 2.0
    out["survives_fdr"] = out["reject_bh"].fillna(False)
    out["survives_fwer"] = out["reject_bonferroni"].fillna(False)
    out = out.sort_values(t_col, key=lambda s: s.abs(), ascending=False)
    out.attrs.update(bh.attrs)
    out.attrs["n_naive_significant"] = int(out["naive_significant"].sum())
    out.attrs["n_survives_fdr"] = int(out["survives_fdr"].sum())
    out.attrs["n_survives_fwer"] = int(out["survives_fwer"].sum())
    return out


def effective_n_tests(n_tests: int, avg_correlation: float = 0.0) -> float:
    """How many *independent* tests a family of correlated ones is worth.

    ``n / (1 + (n-1) * rho)`` -- the standard first-order adjustment. Fifteen
    factors correlated at 0.3 behave like about three independent tries, which is
    why the correlation matters: correcting as though all fifteen were
    independent over-penalises, and correcting for one under-penalises.
    """
    n = max(int(n_tests), 1)
    rho = float(np.clip(avg_correlation, 0.0, 0.99))
    return max(n / (1.0 + (n - 1) * rho), 1.0) if n > 1 else 1.0


def expected_max_t(n_tests: int, avg_correlation: float = 0.0) -> float:
    """The |t| the *best* of n tests reaches under the null. Median of the maximum.

    This is the number that makes the selection problem concrete. With fifteen
    factors and no signal whatsoever, the largest |t| in the family lands near 2 --
    so "our best factor hit 2.1" is not evidence, it is arithmetic.

    Derived exactly rather than approximated. For n iid standard normals,
    ``P(max|Z| <= x) = (2*Phi(x) - 1) ** n``; setting that to 0.5 and inverting
    gives the median of the maximum. The n=1 case correctly returns 0.674, the
    median of a single |Z| -- an approximation that fails that check (an earlier
    draft of this function returned 0) is not usable near a decision threshold.
    """
    from scipy import stats
    n_eff = effective_n_tests(n_tests, avg_correlation)
    q = (1.0 + 0.5 ** (1.0 / n_eff)) / 2.0
    return float(stats.norm.ppf(q))


def family_critical_t(n_tests: int, avg_correlation: float = 0.0,
                      alpha: float = 0.05) -> float:
    """The |t| a factor must clear for family-wise significance. Sidak, exact.

    ``P(max|Z| <= x) = 1 - alpha`` under the null, so
    ``x = Phi^-1((1 + (1-alpha)**(1/n)) / 2)``. This is the threshold to quote in
    a memo instead of 2.0 -- on fifteen correlated factors it is materially
    higher, and the difference is precisely the licence a search takes.
    """
    from scipy import stats
    n_eff = effective_n_tests(n_tests, avg_correlation)
    q = (1.0 + (1.0 - float(alpha)) ** (1.0 / n_eff)) / 2.0
    return float(stats.norm.ppf(q))


# ==========================================================================
# Deflated Sharpe ratio
# ==========================================================================
def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` under a null of no skill.

    Bailey and Lopez de Prado's ``SR*``. ``sr_variance`` is the variance of the
    Sharpe ratios actually tried -- the more the variants disagreed, the higher
    the bar the winner has to clear, because dispersion is what a search exploits.
    """
    from scipy import stats
    n = max(int(n_trials), 2)
    sd = float(np.sqrt(max(sr_variance, 0.0)))
    if sd == 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(observed_sr: float, n_obs: int, n_trials: int,
                    sr_variance: float, skew: float = 0.0,
                    excess_kurtosis: float = 0.0,
                    periods_per_year: float | None = None) -> dict:
    """Probability the observed Sharpe is real, given the search that found it.

    Two corrections at once, and both matter here:

    * **Selection.** The benchmark is not zero, it is ``SR*`` -- the Sharpe the
      best of ``n_trials`` would reach with no skill at all.
    * **Non-normality.** The standard error of a Sharpe ratio depends on the
      third and fourth moments. A left-skewed, fat-tailed equity return series
      has a *wider* sampling distribution than the textbook formula assumes, so
      the textbook t-stat is too generous exactly where it is most often quoted.

    ``observed_sr`` and the returned ``sr_star`` are **per period**, not
    annualised. Pass ``periods_per_year`` and the annualised equivalents are
    added for readability, but the test is done per period, which is where the
    moments live.

    A DSR below 0.95 means the Sharpe has not been shown to be different from
    what the search would have produced by luck.
    """
    from scipy import stats
    t = int(n_obs)
    sr = float(observed_sr)
    if t < 4 or not np.isfinite(sr):
        return {"deflated_sharpe_prob": np.nan, "note": "too few observations"}

    sr_star = expected_max_sharpe(n_trials, sr_variance)
    g3, g4 = float(skew), float(excess_kurtosis)
    denom_sq = 1.0 - g3 * sr + (g4 / 4.0) * sr ** 2
    if denom_sq <= 0:
        return {"deflated_sharpe_prob": np.nan, "sr_star": sr_star,
                "note": "moment correction is non-positive; sample is too "
                        "non-normal for this approximation"}
    z = (sr - sr_star) * np.sqrt(t - 1) / np.sqrt(denom_sq)
    prob = float(stats.norm.cdf(z))

    out = {
        "observed_sr_per_period": sr,
        "sr_star_per_period": sr_star,
        "n_trials": int(n_trials),
        "n_obs": t,
        "skew": g3,
        "excess_kurtosis": g4,
        "z": float(z),
        "deflated_sharpe_prob": prob,
        "verdict": ("skill supported (DSR >= 0.95)" if prob >= 0.95
                    else "not distinguishable from selection luck"),
    }
    if periods_per_year:
        k = np.sqrt(periods_per_year)
        out["observed_sr_ann"] = sr * k
        out["sr_star_ann"] = sr_star * k
    return out


def min_track_record_length(observed_sr: float, n_obs: int,
                            target_sr: float = 0.0,
                            skew: float = 0.0, excess_kurtosis: float = 0.0,
                            confidence: float = 0.95) -> dict:
    """How long a track record must be before this Sharpe means anything.

    The answer is frequently longer than the track record. Reporting it turns
    "our Sharpe is 0.9" into "our Sharpe is 0.9 and we would need eleven years to
    be 95% sure it is above zero" -- which is the same fact, stated so that
    nobody can act on it carelessly.
    """
    from scipy import stats
    sr, t = float(observed_sr), int(n_obs)
    if sr <= target_sr or t < 4:
        return {"min_trl": np.inf, "n_obs": t,
                "note": "observed Sharpe does not exceed the target"}
    g3, g4 = float(skew), float(excess_kurtosis)
    var_term = 1.0 - g3 * sr + (g4 / 4.0) * sr ** 2
    if var_term <= 0:
        return {"min_trl": np.nan, "note": "moment correction is non-positive"}
    z = stats.norm.ppf(confidence)
    trl = 1.0 + var_term * (z / (sr - target_sr)) ** 2
    return {
        "min_trl_periods": float(trl),
        "n_obs": t,
        "sufficient": bool(t >= trl),
        "shortfall_periods": float(max(trl - t, 0.0)),
        "confidence": confidence,
        "target_sr_per_period": target_sr,
    }


# ==========================================================================
# Probability of backtest overfitting (CSCV)
# ==========================================================================
def pbo_cscv(perf_matrix: pd.DataFrame, n_splits: int | None = None,
             seed: int | None = None) -> dict:
    """Combinatorially symmetric cross-validation: does the in-sample winner hold up?

    ``perf_matrix`` is periods x configurations -- one column per variant tried
    (weighting scheme, horizon, factor subset). The sample is cut into ``S`` equal
    blocks; every way of choosing ``S/2`` blocks as in-sample is evaluated; the
    configuration that wins in-sample is looked up out-of-sample, and its relative
    rank recorded.

    If selection carried information, the in-sample winner lands high out of
    sample. If it was overfitting, it lands around the middle -- or below it,
    which happens more often than intuition allows. **PBO is the share of splits
    where the in-sample winner finished below the out-of-sample median.**

    Above roughly 0.5 the selection process is worse than picking at random, and
    the sensible response is to stop selecting rather than to select harder.

    **Read one PBO figure loosely.** Calibrated here on 60 pure-noise matrices of
    120 periods x 20 configurations: the mean PBO came back at 0.507, exactly the
    0.5 the null predicts, but the 5th-95th percentile range across those matrices
    was 0.20 to 0.83. A single run therefore locates the answer to within about
    +/-0.3, which is enough to separate 0.1 from 0.9 and not enough to distinguish
    0.45 from 0.55. Quote it as a band, never as a third decimal.
    """
    M = perf_matrix.dropna(how="all", axis=1).dropna()
    T, N = M.shape
    S = config.CSCV_SPLITS if n_splits is None else int(n_splits)
    S = max(2, S - (S % 2))                       # must be even to halve
    if N < 2:
        return {"pbo": np.nan, "note": "need at least two configurations"}
    if T < S * 2:
        return {"pbo": np.nan, "note": f"need at least {S * 2} periods, have {T}"}

    blocks = np.array_split(np.arange(T), S)
    logits, ranks = [], []
    for is_idx in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_idx])
        is_perf = M.iloc[is_rows].mean()
        oos_perf = M.iloc[oos_rows].mean()
        best = is_perf.idxmax()
        # Relative rank of the IS winner among OOS results, in (0, 1).
        r = float(oos_perf.rank(pct=True)[best])
        r = float(np.clip(r, 1.0 / (N + 1), 1.0 - 1.0 / (N + 1)))
        ranks.append(r)
        logits.append(np.log(r / (1.0 - r)))

    logits = np.asarray(logits, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    return {
        "pbo": float((logits <= 0).mean()),
        "n_splits_evaluated": int(len(logits)),
        "n_configurations": int(N),
        "n_periods": int(T),
        "median_oos_rank": float(np.median(ranks)),
        "mean_logit": float(logits.mean()),
        "verdict": ("selection is not adding information"
                    if float((logits <= 0).mean()) >= 0.5
                    else "in-sample winner tends to hold up out of sample"),
    }


# ==========================================================================
# Purged and embargoed splits
# ==========================================================================
def purged_splits(index: pd.DatetimeIndex, n_splits: int = 5,
                  horizon_periods: int = 1,
                  embargo_pct: float | None = None) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Walk-forward train/test splits with purging and an embargo.

    The leak this fixes is specific and invisible. A score dated at t is evaluated
    against the return from t to t+h. If t sits in the training set and t+h sits
    inside the test set, the training label overlaps the test period -- the model
    has been shown the test set's returns without anyone writing a line of code
    that looks wrong.

    * **Purge** -- drop training observations whose label window reaches into the
      test set.
    * **Embargo** -- additionally drop a short run immediately after the test set,
      because serial correlation makes the periods just after the test window
      nearly as informative as the window itself.

    Returns ``[(train_index, test_index), ...]`` in chronological order. Splits are
    expanding-window, which is how a strategy would actually have been run.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    emb_pct = config.PURGE_EMBARGO_PCT if embargo_pct is None else float(embargo_pct)
    embargo = int(np.ceil(T * emb_pct))
    h = max(int(horizon_periods), 1)
    if n_splits < 2 or T < n_splits * (h + 2):
        raise ValueError(f"purged_splits: {T} periods cannot support {n_splits} "
                         f"splits at horizon {h}")

    fold_edges = np.linspace(0, T, n_splits + 1).astype(int)
    out = []
    for k in range(1, n_splits):
        test_start, test_end = fold_edges[k], fold_edges[k + 1]
        test_pos = np.arange(test_start, test_end)
        if len(test_pos) == 0:
            continue
        # Training is everything before the test block, minus the observations
        # whose forward label reaches into it, minus the embargo after it.
        train_pos = np.arange(0, max(test_start - h, 0))
        emb_lo, emb_hi = test_end, min(test_end + embargo, T)
        train_pos = train_pos[(train_pos < emb_lo) | (train_pos >= emb_hi)]
        if len(train_pos) < h + 2:
            continue
        out.append((idx[train_pos], idx[test_pos]))
    return out


def walk_forward_ic(scores: pd.DataFrame, fwd: pd.DataFrame,
                    n_splits: int = 5, horizon_periods: int = 1,
                    embargo_pct: float | None = None) -> pd.DataFrame:
    """Rank IC computed inside each purged out-of-sample block.

    Nothing is fitted here -- the score is a stated spec, not a trained model --
    so this is not cross-validation in the machine-learning sense. It answers a
    different and more useful question: **was the signal present in every part of
    the sample, or in one part of it?** A mean IC of 0.03 built entirely from
    2020-21 is a regime artefact wearing a full-sample disguise, and only a split
    like this shows it.
    """
    from . import backtest as bt

    idx = scores.index.intersection(fwd.index)
    splits = purged_splits(idx, n_splits, horizon_periods, embargo_pct)
    rows = []
    for i, (train, test) in enumerate(splits, start=1):
        ic_tr = bt.rank_ic(scores.loc[scores.index.isin(train)],
                           fwd.loc[fwd.index.isin(train)])
        ic_te = bt.rank_ic(scores.loc[scores.index.isin(test)],
                           fwd.loc[fwd.index.isin(test)])
        s_tr = bt.ic_summary(ic_tr) if len(ic_tr) >= 3 else {}
        s_te = bt.ic_summary(ic_te) if len(ic_te) >= 3 else {}
        rows.append({
            "fold": i,
            "train_start": train.min().date(), "train_end": train.max().date(),
            "test_start": test.min().date(), "test_end": test.max().date(),
            "n_train": len(ic_tr), "n_test": len(ic_te),
            "train_mean_ic": s_tr.get("mean_ic", np.nan),
            "test_mean_ic": s_te.get("mean_ic", np.nan),
            "test_t_stat": s_te.get("t_stat", np.nan),
            "test_hit_rate_%": s_te.get("hit_rate_%", np.nan),
        })
    out = pd.DataFrame(rows).set_index("fold")
    if len(out):
        pos = out["test_mean_ic"] > 0
        out.attrs["folds_positive"] = int(pos.sum())
        out.attrs["folds_total"] = int(out["test_mean_ic"].notna().sum())
        out.attrs["oos_mean_ic"] = float(out["test_mean_ic"].mean())
        out.attrs["consistent"] = bool(pos.all())
    return out


# ==========================================================================
# Bootstrap
# ==========================================================================
def _stationary_indices(T: int, n_boot: int, mean_block: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano resampling indices: ``(n_boot, T)``, wrapping at the end.

    Factored out so that a family of candidates can be resampled on the **same**
    draws. That is not a convenience -- it is what preserves the cross-sectional
    correlation between candidates, and therefore what makes fifteen correlated
    factors count as fewer effective tries in ``reality_check``.
    """
    p = 1.0 / max(int(mean_block), 1)
    starts = rng.integers(0, T, size=(n_boot, T))
    cont = rng.random((n_boot, T)) >= p
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for j in range(1, T):
        idx[:, j] = np.where(cont[:, j], (idx[:, j - 1] + 1) % T, starts[:, j])
    return idx


def stationary_bootstrap(x: pd.Series, n_boot: int | None = None,
                         mean_block: int | None = None,
                         seed: int | None = None) -> np.ndarray:
    """Politis-Romano stationary bootstrap of the sample mean.

    An IID bootstrap destroys the serial dependence that overlapping forward
    windows create, and therefore reports a confidence interval that is too
    narrow -- the same error Newey-West exists to fix, made a second way. The
    stationary bootstrap resamples blocks of geometrically distributed length,
    which preserves short-range dependence while keeping the resampled series
    stationary.

    Returns the bootstrap distribution of the mean.
    """
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    T = len(v)
    if T < 8:
        return np.array([])
    B = config.BOOTSTRAP_N if n_boot is None else int(n_boot)
    L = config.BOOTSTRAP_BLOCK if mean_block is None else int(mean_block)
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    return v[_stationary_indices(T, B, L, rng)].mean(axis=1)


def bootstrap_ci(x: pd.Series, confidence: float = 0.95,
                 n_boot: int | None = None, mean_block: int | None = None,
                 seed: int | None = None) -> dict:
    """Block-bootstrap confidence interval for a mean IC or mean return.

    Quoted alongside the Newey-West t-statistic rather than instead of it. When
    the two disagree the sample is telling you something about its own tails, and
    the bootstrap interval -- which assumes nothing about the distribution -- is
    the one to trust.
    """
    dist = stationary_bootstrap(x, n_boot, mean_block, seed)
    if dist.size == 0:
        return {"note": "too few observations to bootstrap"}
    lo, hi = np.percentile(dist, [(1 - confidence) / 2 * 100,
                                  (1 + confidence) / 2 * 100])
    obs = float(pd.to_numeric(x, errors="coerce").dropna().mean())
    return {
        "observed_mean": obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "p_two_sided": float(2 * min((dist <= 0).mean(), (dist >= 0).mean())),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": int(dist.size),
        "mean_block": config.BOOTSTRAP_BLOCK if mean_block is None else mean_block,
    }


def reality_check(candidate_panel: pd.DataFrame, n_boot: int | None = None,
                  mean_block: int | None = None, seed: int | None = None) -> dict:
    """White's Reality Check over a family of candidates. Block bootstrap.

    ``candidate_panel`` is **periods x candidates** -- one column per variant
    tried, one row per rebalance. For a factor scan that is the per-factor rank-IC
    series; for a strategy scan it is the per-period return of each variant. The
    per-period series is required rather than a summary mean, because the null
    distribution of the *maximum* depends on how the candidates co-move, and that
    information does not survive being averaged away.

    The procedure is the standard one:

    * the statistic is ``V = sqrt(T) * max_j mean(x_j)``;
    * each bootstrap replication resamples **time**, using the same block draws
      for every candidate so their correlation is preserved, and recentres each
      candidate on its own observed mean to impose the null;
    * ``p`` is the share of replications whose recentred maximum exceeds ``V``.

    Recentring both sides on the same basis is the part that is easy to get wrong.
    An earlier draft of this function centred the null but compared it against an
    uncentred observed maximum, which returned ``p = 0`` for any family with a
    positive average -- a test that always passes is not a test.

    Fifteen correlated factors resampled on shared draws behave like about three
    independent ones, which is exactly the effect ``effective_n_tests``
    approximates parametrically and this measures directly.
    """
    M = candidate_panel.dropna(how="all", axis=1).dropna()
    T, N = M.shape
    if N < 2:
        return {"note": "need at least two candidates"}
    if T < 8:
        return {"note": f"need at least 8 periods, have {T}"}

    B = config.BOOTSTRAP_N if n_boot is None else int(n_boot)
    L = config.BOOTSTRAP_BLOCK if mean_block is None else int(mean_block)
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    x = M.to_numpy(dtype=float)
    means = x.mean(axis=0)
    root_t = np.sqrt(T)
    observed_v = float(root_t * means.max())

    idx = _stationary_indices(T, B, L, rng)
    boot_means = x[idx].mean(axis=1)                 # (B, N)
    null_v = root_t * (boot_means - means).max(axis=1)

    p = float((null_v >= observed_v).mean())
    best = str(M.columns[int(np.argmax(means))])
    return {
        "best_candidate": best,
        "best_mean": float(means.max()),
        "observed_V": observed_v,
        "null_V_p95": float(np.percentile(null_v, 95)),
        "p_value": p,
        "n_candidates": int(N),
        "n_periods": int(T),
        "n_boot": B,
        "mean_block": L,
        "verdict": ("best candidate survives the search (p < 0.05)" if p < 0.05
                    else "best candidate is within what the search alone produces"),
    }


# ==========================================================================
# Guard
# ==========================================================================
def assert_multiple_testing_applied(table: pd.DataFrame,
                                    label: str = "factor family",
                                    avg_correlation: float = 0.3) -> pd.DataFrame:
    """Regression guard: refuse a factor table that quotes raw t-stats alone.

    The failure mode is social, not technical -- someone reruns the factor scan,
    sees a t of 2.6, and writes it into a memo. This makes that table structurally
    unquotable until the correction has been applied to it, and reports the gap
    between naive and corrected discoveries so the size of the selection effect is
    on the page.
    """
    required = {"p_adj_bh", "survives_fdr", "naive_significant"}
    missing = required - set(table.columns)
    if missing:
        raise AssertionError(
            f"{label}: factor table has not been through factor_family_test -- "
            f"missing {sorted(missing)}. Raw t-statistics from a family of tests "
            f"are not evidence; the best of n tries clears |t|>2 by construction.")

    naive = int(table["naive_significant"].sum())
    survive = int(table["survives_fdr"].sum())
    n = len(table)
    return pd.DataFrame([{
        "n_tests": n,
        "naive_significant": naive,
        "survives_fdr": survive,
        "survives_fwer": int(table.get("survives_fwer", pd.Series(dtype=bool)).sum()),
        "median_max_t_under_null": round(expected_max_t(n, avg_correlation), 2),
        "family_critical_t": round(family_critical_t(n, avg_correlation), 2),
        "naive_critical_t": 1.96,
        "selection_effect": naive - survive,
        "result": (f"{survive} of {naive} naive discoveries survive multiple-testing "
                   f"control across {n} tests; the bar is |t| > "
                   f"{family_critical_t(n, avg_correlation):.2f}, not 1.96"),
    }])
