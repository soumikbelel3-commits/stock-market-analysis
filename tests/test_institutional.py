"""Regression tests for the institutional layer. Offline, no network.

Same standard as ``test_guards.py``: a guard nobody has seen fail is not a guard,
and a statistic nobody has checked against a known answer is not a statistic. So
these tests do two things:

**Prove the new guards fire.**

    perf.assert_excess_return_basis        a Sharpe computed against zero
    costs.assert_cost_sanity               a short leg priced as cash equity
    validation.assert_multiple_testing_applied   raw t-stats from a family scan
    risk.assert_var_model_validated        a VaR model falsified by its backtest
    factors.assert_factor_pit              exposures with no provenance
    factors.assert_model_vs_sample         a factor model that describes another book
    margin.assert_capital_adequate         a book the account cannot carry
    limits.assert_within_limits            a hard limit breach, and an unmeasured one
    report.assert_memo_honesty             a memo claiming more than its evidence

**Prove the maths is right**, by checking closed forms against simulation and
against constructions whose answer is known in advance. Three real bugs were
found this way while the layer was being written, and each has a test here:

* ``expected_max_t`` returned 0 for a single test where the answer is 0.674;
* ``basel_traffic_light`` applied the 250-day/99% thresholds at 95%, putting a
  well-calibrated model in the red zone;
* ``reality_check`` centred its null but not its observed statistic, so it
  returned p = 0 for any family with a positive mean.

Run with::

    python tests/test_institutional.py
    pytest tests/test_institutional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from mkt import (config, costs, factors as fx, limits, margin,  # noqa: E402
                 perf, report, risk as rk, validation as val)

SEED = 12345


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rets(n: int = 900, mu: float = 0.0006, sd: float = 0.011,
          seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    x = rng.normal(0, sd, n)
    return pd.Series(x - x.mean() + mu, index=idx)


def _book() -> pd.DataFrame:
    b = pd.DataFrame(
        {"side": ["long"] * 4 + ["short"] * 3,
         "weight": [0.20, 0.18, 0.15, 0.12, -0.14, -0.11, -0.10],
         "beta": [1.10, 0.85, 1.05, 0.90, 1.20, 0.75, 0.95]},
        index=[f"N{i}.NS" for i in range(7)])
    b["beta_contrib"] = b["weight"] * b["beta"]
    return b


def _panel_rets(n_days: int = 700, n_names: int = 7, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    mkt = rng.normal(0, 0.009, n_days)[:, None]
    return pd.DataFrame(mkt * rng.uniform(0.7, 1.3, n_names)
                        + rng.normal(0, 0.010, (n_days, n_names)),
                        index=idx, columns=[f"N{i}.NS" for i in range(n_names)])


def _raises(fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except AssertionError as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise")


# ==========================================================================
# perf -- the risk-free defect
# ==========================================================================
def test_excess_return_guard_fires_on_a_zero_risk_free_rate():
    r = _rets()
    zero = perf.perf_stats(r, rf_annual_pct=0.0)
    msg = _raises(perf.assert_excess_return_basis, zero, "long book")
    assert "risk-free rate is 0" in msg, msg

    # A genuinely market-neutral book funded on collateral is the documented
    # exception -- but it has to be asked for explicitly.
    ok = perf.assert_excess_return_basis(zero, "neutral book", allow_zero_rf=True)
    assert ok.loc[0, "basis"].startswith("neutral book")

    real = perf.perf_stats(r, rf_annual_pct=6.25)
    perf.assert_excess_return_basis(real, "long book")
    print(f"  perf      raised on rf=0; Sharpe {zero['sharpe']:.3f} -> "
          f"{real['sharpe']:.3f} once measured against cash")


def test_sharpe_overstatement_equals_rf_over_vol():
    """The size of the defect is predictable, which is how we know it is the defect."""
    r = _rets(n=2000, mu=0.0005, sd=0.010)
    rf = 6.25
    c = perf.sharpe_correction(r, rf).iloc[0]
    vol = float(r.std(ddof=1) * np.sqrt(252))
    # Sharpe(0) - Sharpe(rf) = rf_per_period / sd * sqrt(252) ~= rf_annual / vol
    predicted = (rf / 100.0) / vol
    assert abs(c["overstatement"] - predicted) < 0.02, (c["overstatement"], predicted)
    print(f"  perf      overstatement {c['overstatement']:.3f} matches rf/vol "
          f"{predicted:.3f} at {vol * 100:.1f}% vol")


def test_perf_stats_match_hand_calculation():
    r = _rets(n=1200, mu=0.0006, sd=0.012)
    s = perf.perf_stats(r, rf_annual_pct=6.25)
    rf_p = (1.0625) ** (1 / 252) - 1
    hand = float((r.mean() - rf_p) / r.std(ddof=1) * np.sqrt(252))
    assert abs(s["sharpe"] - hand) < 1e-9, (s["sharpe"], hand)
    eq = float((1 + r).cumprod().iloc[-1])
    hand_cagr = (eq ** (252 / len(r)) - 1) * 100
    assert abs(s["cagr_%"] - hand_cagr) < 1e-9
    print(f"  perf      Sharpe and CAGR match hand calculation to 1e-9")


def test_brinson_attribution_is_exact_when_weights_sum_to_one():
    pw = pd.Series({"A": 0.30, "B": 0.25, "C": 0.25, "D": 0.20})
    bw = pd.Series({"A": 0.20, "B": 0.20, "C": 0.30, "D": 0.20, "E": 0.10})
    nr = pd.Series({"A": 0.12, "B": -0.04, "C": 0.06, "D": 0.02, "E": 0.09})
    sec = {"A": "IT", "B": "IT", "C": "Fin", "D": "Fin", "E": "FMCG"}
    br = perf.brinson_attribution(pw, bw, nr, sec)
    assert abs(br.attrs["residual_pp"]) < 1e-9, br.attrs["residual_pp"]
    assert abs(br["total_pp"].sum() - br.attrs["active_return_pp"]) < 1e-9
    print(f"  perf      Brinson allocation+selection+interaction reconciles to "
          f"{br.attrs['active_return_pp']:.4f}pp active, residual "
          f"{br.attrs['residual_pp']:.2e}")


# ==========================================================================
# costs
# ==========================================================================
def test_cost_guard_fires_when_a_short_is_priced_as_cash_equity():
    book = _book()
    tc = costs.trade_cost(book, capital=1e7)
    costs.assert_cost_sanity(tc)                      # clean book passes

    broken = tc.copy()
    broken.loc[broken["side"] == "short", "instrument"] = costs.CASH
    msg = _raises(costs.assert_cost_sanity, broken, 200.0, "broken")
    assert "future" in msg and "trap #5" in msg, msg
    print(f"  costs     raised when a short leg was priced as cash equity")


def test_short_leg_statutory_cost_is_far_below_the_long_leg():
    """The asymmetry a flat round-trip number hides."""
    long_rt = costs.round_trip_bps(costs.CASH, is_short=False)
    short_rt = costs.round_trip_bps(costs.FUT, is_short=True)
    assert long_rt > 2.5 * short_rt, (long_rt, short_rt)
    # STT dominates and is charged both ways on cash, one way on futures
    # (0.05% on the futures sell since Budget 2026; it was 0.02%).
    stt_cash = (costs.explicit_bps(costs.CASH, "buy")["stt"]
                + costs.explicit_bps(costs.CASH, "sell")["stt"])
    stt_fut = (costs.explicit_bps(costs.FUT, "buy")["stt"]
               + costs.explicit_bps(costs.FUT, "sell")["stt"])
    assert abs(stt_cash - 20.0) < 1e-9 and abs(stt_fut - 5.0) < 1e-9, (stt_cash, stt_fut)
    print(f"  costs     long round trip {long_rt:.2f}bps vs short {short_rt:.2f}bps "
          f"({long_rt / short_rt:.1f}x); STT {stt_cash:.0f} vs {stt_fut:.0f}bps")


def test_gst_is_not_charged_on_stt_or_stamp_duty():
    """The most common error in a hand-rolled India cost model."""
    e = costs.explicit_bps(costs.CASH, "buy")
    expected_gst = (e["brokerage"] + e["exchange_txn"] + e["sebi_fee"]) * 0.18
    assert abs(e["gst"] - expected_gst) < 1e-9, (e["gst"], expected_gst)
    assert e["gst"] < e["stt"] * 0.18, "GST looks like it is taxing STT"
    print(f"  costs     GST {e['gst']:.4f}bps is 18% of brokerage+exchange+SEBI only")


def test_impact_is_concave_in_size():
    """Square-root law: 4x the order is 2x the impact, not 4x."""
    a = costs.impact_bps(1e6, 1e9, 0.015, costs.CASH)
    b = costs.impact_bps(4e6, 1e9, 0.015, costs.CASH)
    half = config.HALF_SPREAD_BPS[costs.CASH]
    ratio = (b - half) / (a - half)
    assert abs(ratio - 2.0) < 1e-6, ratio
    print(f"  costs     4x order size -> {ratio:.4f}x impact (square-root law)")


# ==========================================================================
# validation
# ==========================================================================
def test_multiple_testing_guard_fires_on_a_raw_factor_table():
    rng = np.random.default_rng(SEED)
    raw = pd.DataFrame({"t_stat": rng.normal(0, 1, 15), "n_periods": 107},
                       index=[f"f{i}" for i in range(15)])
    msg = _raises(val.assert_multiple_testing_applied, raw, "family")
    assert "factor_family_test" in msg, msg
    tested = val.factor_family_test(raw)
    rep = val.assert_multiple_testing_applied(tested)
    assert rep.loc[0, "family_critical_t"] > 1.96
    print(f"  validation raised on a raw table; corrected bar is |t| > "
          f"{rep.loc[0, 'family_critical_t']:.2f}, not 1.96")


def test_expected_max_t_matches_simulation_and_the_n_equals_one_case():
    """The bug: an earlier approximation returned 0 for a single test."""
    assert abs(val.expected_max_t(1, 0.0) - 0.6745) < 0.002, val.expected_max_t(1, 0.0)
    rng = np.random.default_rng(7)
    for n in (5, 15, 40):
        sim = float(np.median(np.abs(rng.standard_normal((60000, n))).max(axis=1)))
        closed = val.expected_max_t(n, 0.0)
        assert abs(closed - sim) < 0.02, (n, closed, sim)
    crit = val.family_critical_t(15, 0.0)
    sim_crit = float(np.percentile(np.abs(rng.standard_normal((60000, 15))).max(axis=1), 95))
    assert abs(crit - sim_crit) < 0.03, (crit, sim_crit)
    print(f"  validation expected_max_t matches simulation; n=1 gives "
          f"{val.expected_max_t(1, 0.0):.4f} (not 0)")


def test_deflated_sharpe_punishes_a_searched_result():
    sr_ann, ppy, n = 0.90, 12, 107
    sr_p = sr_ann / np.sqrt(ppy)
    kw = dict(n_obs=n, sr_variance=(0.5 / np.sqrt(ppy)) ** 2,
              skew=-0.6, excess_kurtosis=2.5, periods_per_year=ppy)
    single = val.deflated_sharpe(sr_p, n_trials=1, **kw)["deflated_sharpe_prob"]
    searched = val.deflated_sharpe(sr_p, n_trials=40, **kw)["deflated_sharpe_prob"]
    assert single > 0.95 > searched, (single, searched)
    print(f"  validation same Sharpe: DSR {single:.3f} pre-registered vs "
          f"{searched:.3f} after 40 tries")


def test_pbo_is_calibrated_on_noise_and_detects_a_real_winner():
    rng = np.random.default_rng(3)
    pbos = []
    for _ in range(12):
        M = pd.DataFrame(rng.normal(0, 0.03, (120, 20)),
                         index=pd.date_range("2016-01-31", periods=120, freq="ME"))
        pbos.append(val.pbo_cscv(M, n_splits=8)["pbo"])
    assert 0.25 < float(np.mean(pbos)) < 0.75, np.mean(pbos)
    M2 = pd.DataFrame(rng.normal(0, 0.03, (120, 20)),
                      index=pd.date_range("2016-01-31", periods=120, freq="ME"))
    M2[0] += 0.03
    assert val.pbo_cscv(M2, n_splits=8)["pbo"] < 0.1
    print(f"  validation PBO on noise averages {np.mean(pbos):.3f} (null is 0.5); "
          f"a real winner scores <0.1")


def test_reality_check_is_calibrated_and_not_trivially_significant():
    """The bug: centring the null but not the observed statistic gave p=0 always."""
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-31", periods=107, freq="ME")
    ps = []
    for _ in range(25):
        P = pd.DataFrame(rng.normal(0, 0.20, (107, 15)), index=idx,
                         columns=[f"f{j}" for j in range(15)])
        ps.append(val.reality_check(P, n_boot=300)["p_value"])
    fp = float(np.mean(np.array(ps) < 0.05))
    assert fp < 0.20, f"false-positive rate {fp:.3f} far above nominal 0.05"
    assert float(np.mean(ps)) > 0.15, f"p-values collapsed to zero: {np.mean(ps):.3f}"
    P = pd.DataFrame(rng.normal(0, 0.20, (107, 15)), index=idx,
                     columns=[f"f{j}" for j in range(15)])
    P["f0"] += 0.12
    assert val.reality_check(P, n_boot=300)["p_value"] < 0.05
    print(f"  validation reality check: false-positive rate {fp:.3f}, mean p "
          f"{np.mean(ps):.3f}; detects a planted signal")


def test_purged_splits_never_leak_the_horizon_across_the_boundary():
    idx = pd.date_range("2016-01-31", periods=107, freq="ME")
    h = 6
    for train, test in val.purged_splits(idx, n_splits=5, horizon_periods=h,
                                         embargo_pct=0.01):
        assert train.max() < test.min(), "train reaches into test"
        gap = (test.min().to_period("M") - train.max().to_period("M")).n
        assert gap >= h, f"purge gap {gap} is shorter than the {h}-period horizon"
    print(f"  validation purged splits keep a >= {h}-rebalance gap at every boundary")


# ==========================================================================
# risk -- VaR model validation
# ==========================================================================
def test_basel_zones_reproduce_the_published_boundaries():
    """The bug: linearly scaled 250-day thresholds applied at 95%."""
    b = rk.basel_traffic_light(250, 0, conf=0.99)
    assert b["green_up_to"] == 4 and b["yellow_up_to"] == 9, b
    # At 95% over 1250 days the expected count is 62.5, so a threshold near 4
    # would be nonsense. A well-calibrated model must land green.
    b95 = rk.basel_traffic_light(1250, 63, conf=0.95)
    assert b95["zone"] == "green", b95
    assert rk.basel_traffic_light(1250, 200, conf=0.95)["zone"] == "red"
    print(f"  risk      Basel reproduces 4/9 at 250d/99%, and scales correctly to "
          f"95% (green<={b95['green_up_to']})")


def test_var_backtest_passes_a_good_model_and_rejects_a_stale_one():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2019-01-01", periods=1500)
    r = pd.Series(rng.standard_t(8, 1500) / np.sqrt(8 / 6) * 0.011, index=idx)

    good = rk.rolling_var_forecast(r, 252, 0.95, "historical")
    bt_good = rk.var_backtest(r, good, 0.95)
    rk.assert_var_model_validated(bt_good)

    stale = r.copy()
    stale.iloc[750:] *= 3.0
    frozen = float(-np.percentile(stale.iloc[:750], 5))
    bt_bad = rk.var_backtest(stale, frozen, 0.95)
    msg = _raises(rk.assert_var_model_validated, bt_bad, "stale VaR")
    assert "RED zone" in msg, msg
    print(f"  risk      VaR guard passed a good model ({int(bt_good.iloc[0]['n_exceptions'])} "
          f"exceptions vs {bt_good.iloc[0]['expected_exceptions']:.0f} expected) and "
          f"rejected a frozen one")


def test_christoffersen_detects_clustered_breaches_that_kupiec_misses():
    """The whole reason independence is tested separately."""
    n = 1000
    ex = pd.Series(False, index=range(n))
    ex.iloc[400:450] = True            # 50 breaches, all consecutive
    pof = rk.kupiec_pof(n, int(ex.sum()), 0.95)
    ind = rk.christoffersen_independence(ex)
    assert not pof["reject_5pct"], "coverage should look fine at 5%"
    assert ind["reject_5pct"], "clustering should be rejected"
    print(f"  risk      50 consecutive breaches: Kupiec p={pof['p_value']:.3f} (passes), "
          f"independence p={ind['p_value']:.2e} (rejects)")


def test_marginal_var_components_sum_to_total_and_flag_a_hedge():
    rng = np.random.default_rng(SEED)
    names = [f"N{i}" for i in range(6)]
    R = pd.DataFrame(rng.normal(0, 0.013, (600, 6)), columns=names)
    R["N0"] = R["N0"] * 0.4 - R["N1"] * 0.5          # N0 hedges N1
    cov = R.cov()
    w = pd.Series([0.25, 0.25, 0.20, 0.15, -0.10, -0.05], index=names)
    mv = rk.marginal_var(w, cov, 0.95)
    assert abs(mv.attrs["sum_check_%"] - mv.attrs["total_var_%"]) < 1e-9
    assert mv.attrs["n_hedges"] >= 1, "the constructed hedge was not detected"
    print(f"  risk      component VaR sums exactly to {mv.attrs['total_var_%']:.4f}%; "
          f"{mv.attrs['n_hedges']} position(s) have negative incremental VaR")


def test_bias_test_detects_an_under_forecast():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2020-01-01", periods=800)
    r = pd.Series(rng.normal(0, 0.014, 800), index=idx)
    honest = pd.Series(0.014, index=idx)
    low = pd.Series(0.007, index=idx)                # forecasts half the true vol
    assert not rk.bias_test(r, honest)["significant"]
    b = rk.bias_test(r, low)
    assert b["significant"] and b["bias"] > 1.8, b
    print(f"  risk      bias {b['bias']:.3f} on a half-sized forecast -> "
          f"{b['verdict']}")


# ==========================================================================
# factors
# ==========================================================================
def test_factor_model_recovers_a_known_factor_structure():
    rng = np.random.default_rng(SEED)
    N, T, K = 40, 600, 3
    names = [f"S{i:02d}.NS" for i in range(N)]
    dates = pd.bdate_range("2022-01-03", periods=T)
    Xtrue = rng.normal(0, 1, (N, K))
    ftrue = rng.normal(0, 0.006, (T, K))
    R = pd.DataFrame(rng.normal(0.0003, 0.010, (T, 1))
                     + ftrue @ Xtrue.T + rng.normal(0, 0.012, (T, N)),
                     index=dates, columns=names)
    exp = {f"k{k}": pd.DataFrame(np.tile(Xtrue[:, k], (T, 1)), index=dates,
                                 columns=names) for k in range(K)}
    F, U, R2 = fx.estimate_factor_returns(exp, R)
    for k in range(K):
        c = float(np.corrcoef(F[f"k{k}"], ftrue[:, k])[0, 1])
        assert c > 0.85, (k, c)
    print(f"  factors   recovered {K} known factor return series at corr > 0.85; "
          f"mean cross-sectional R2 {R2.mean():.3f}")


def test_factor_variance_contributions_sum_to_systematic_variance():
    rng = np.random.default_rng(SEED)
    names = [f"S{i}" for i in range(20)]
    X = pd.DataFrame(rng.normal(0, 1, (20, 4)), index=names,
                     columns=["a", "b", "c", "d"])
    X.attrs["pit"] = {c: True for c in X.columns}
    A = rng.normal(0, 0.01, (4, 4))
    Fc = pd.DataFrame(A @ A.T, index=X.columns, columns=X.columns)
    sv = pd.Series(rng.uniform(0.01, 0.05, 20), index=names)
    w = pd.Series(rng.normal(0, 0.06, 20), index=names)
    fe = fx.factor_exposures_of_book(w, X, Fc)
    sys_var = (fe.attrs["systematic_vol_%"] / 100) ** 2
    assert abs(fe.attrs["sum_check"] - sys_var) < 1e-12, (fe.attrs["sum_check"], sys_var)
    dec = fx.decompose_risk(w, X, Fc, sv)
    assert abs(dec["systematic_share_%"] + dec["specific_share_%"] - 100) < 1e-9
    print(f"  factors   per-factor contributions sum exactly to systematic variance; "
          f"shares sum to 100%")


def test_factor_pit_guard_fires_on_exposures_with_no_provenance():
    X = pd.DataFrame(np.zeros((5, 2)), index=[f"S{i}" for i in range(5)],
                     columns=["momentum", "value"])
    msg = _raises(fx.assert_factor_pit, X, "exposures")
    assert "provenance" in msg and "trap #4" in msg, msg
    X.attrs["pit"] = {"momentum": True, "value": False}
    rep = fx.assert_factor_pit(X)
    assert rep.attrs["n_pit"] == 1 and rep.attrs["n_static"] == 1
    print(f"  factors   raised on exposures with no point-in-time provenance map")


def test_factor_model_guard_fires_when_it_describes_another_book():
    rets = _panel_rets()
    names = list(rets.columns)
    X = pd.DataFrame(np.ones((len(names), 1)), index=names, columns=["one"])
    X.attrs["pit"] = {"one": True}
    w = pd.Series(1.0 / len(names), index=names)
    # A factor covariance inflated 400x describes a book nothing like this one.
    Fc = pd.DataFrame([[4.0]], index=["one"], columns=["one"])
    sv = pd.Series(0.01, index=names)
    bad = fx.factor_model_cov(X, Fc, sv)
    msg = _raises(fx.assert_model_vs_sample, w, bad, rets, 2.0, "inflated model")
    assert "different books" in msg, msg
    print(f"  factors   raised when the model volatility and the return panel "
          f"disagreed")


# ==========================================================================
# margin
# ==========================================================================
def test_margin_guard_fires_on_a_book_the_account_cannot_carry():
    book = _book()
    px = pd.Series(1000.0, index=book.index)
    lots = pd.Series(500.0, index=book.index)
    dv = pd.Series(0.018, index=book.index)

    mt = margin.book_margin(book, px, lots, dv, capital=1e7)
    margin.assert_capital_adequate(margin.capital_adequacy(book, mt, capital=1e7))

    over = book.copy()
    over["weight"] = over["weight"] * 3.0
    mt2 = margin.book_margin(over, px, lots, dv, capital=1e7)
    msg = _raises(margin.assert_capital_adequate,
                  margin.capital_adequacy(over, mt2, capital=1e7), "over-levered")
    assert "Minimum capital" in msg, msg
    print(f"  margin    raised on a 3x book and reported the capital that clears it")


def test_span_floor_binds_on_a_low_volatility_name():
    low = margin.span_proxy(0.010)          # 3.5 * 1.0% = 3.5%, below the 5% floor
    high = margin.span_proxy(0.030)         # 3.5 * 3.0% = 10.5%, above it
    assert low == config.SPAN_MIN_SCAN_PCT, low
    assert abs(high - 0.105) < 1e-12, high
    print(f"  margin    floor binds at {low:.1%} on a 1% daily-vol name; "
          f"scan range gives {high:.1%} at 3%")


def test_margin_rises_non_linearly_with_volatility():
    book = _book()
    px = pd.Series(1000.0, index=book.index)
    lots = pd.Series(500.0, index=book.index)
    dv = pd.Series([0.010, 0.012, 0.014, 0.016, 0.011, 0.013, 0.030], index=book.index)
    mt = margin.book_margin(book, px, lots, dv, capital=1e7)
    st = margin.margin_stress(mt, capital=1e7, vol_multiples=(1.0, 1.5, 3.0))
    inc15 = st.loc[1.5, "increase_vs_base_%"]
    inc30 = st.loc[3.0, "increase_vs_base_%"]
    assert 0 <= inc15 < inc30, (inc15, inc30)
    print(f"  margin    a 3x vol spike raises margin {inc30:.0f}% with no trade "
          f"(1.5x raises it {inc15:.0f}%)")


# ==========================================================================
# limits
# ==========================================================================
def test_limits_guard_fires_on_a_hard_breach_and_on_an_unmeasured_hard_limit():
    book = _book()
    book.loc["N0.NS", "weight"] = 0.40                    # breaches max_position
    sec = pd.Series({t: "Fin" for t in book.index})
    m = limits.book_metrics(book, sec)
    mon = limits.monitor(m)
    msg = _raises(limits.assert_within_limits, mon, "book")
    assert "hard limit breach" in msg, msg

    # An unmeasured hard limit is not a satisfied one.
    reg = [("gross_exposure", "<=", 2.0, "hard"),
           ("margin_utilisation", "<=", 0.65, "hard")]
    mon2 = limits.monitor({"gross_exposure": 1.0}, reg)
    msg2 = _raises(limits.assert_within_limits, mon2, "book")
    assert "could not be measured" in msg2, msg2
    print(f"  limits    raised on a hard breach, and on an unmeasured hard limit")


def test_soft_breaches_pass_by_default_and_are_reported():
    reg = [("gross_exposure", "<=", 2.0, "hard"),
           ("net_beta_abs", "<=", 0.20, "soft")]
    mon = limits.monitor({"gross_exposure": 1.0, "net_beta_abs": 0.55}, reg)
    rep = limits.assert_within_limits(mon, allow_soft=True)
    assert rep.loc[0, "soft_breaches"] == 1
    _raises(limits.assert_within_limits, mon, "strict", False)
    print(f"  limits    soft breach passes by default and fails with "
          f"allow_soft=False")


def test_headroom_is_signed_consistently_in_both_directions():
    reg = [("ex_ante_vol_%", "<=", 15.0, "soft"),
           ("effective_bets", ">=", 4.0, "soft")]
    mon = limits.monitor({"ex_ante_vol_%": 12.0, "effective_bets": 6.0}, reg)
    assert (mon["headroom"] > 0).all(), mon["headroom"].to_dict()
    mon2 = limits.monitor({"ex_ante_vol_%": 18.0, "effective_bets": 2.0}, reg)
    assert (mon2["headroom"] < 0).all(), mon2["headroom"].to_dict()
    print(f"  limits    positive headroom means room left for both <= and >= limits")


# ==========================================================================
# report
# ==========================================================================
def test_memo_honesty_guard_fires_when_the_memo_outruns_the_evidence():
    fake = {"07_signal_validation": {"verdict": "Technical leg: Not proven"}}
    dishonest = ("# Investment Committee Memo\n"
                 "> ## Recommendation: DEPLOY AT REDUCED SIZE\n"
                 "trap #4 applies. This is not investment advice.\n")
    msg = _raises(report.assert_memo_honesty, dishonest, fake)
    assert "recommends deployment" in msg, msg

    honest = ("# Investment Committee Memo\n"
              "> ## Recommendation: DO NOT DEPLOY -- paper only\n"
              "trap #4 makes the fundamental leg untestable. "
              "This is not investment advice.\n")
    rep = report.assert_memo_honesty(honest, fake)
    assert rep.loc[0, "stance"].startswith("DO NOT DEPLOY")
    print(f"  report    raised when the memo recommended deploying a screen that "
          f"failed validation")


def test_signal_verdict_derives_the_stance_rather_than_choosing_it():
    for verdict, expect in (("Technical leg: Not proven", "DO NOT DEPLOY"),
                            ("Technical leg: Proven", "DEPLOY"),
                            ("", "HOLD")):
        sv = report.signal_verdict({"07_signal_validation": {"verdict": verdict}})
        assert sv["stance"].startswith(expect), (verdict, sv["stance"])
    print(f"  report    stance is derived from notebook 07's verdict in all three "
          f"cases")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    print(f"running {len(tests)} institutional-layer regression tests\n")
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                          # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        for name, exc in failed:
            print(f"   {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
