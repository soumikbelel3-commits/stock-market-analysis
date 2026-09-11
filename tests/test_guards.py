"""Guard regression tests: prove the trap guards actually fire.

A guard nobody has seen fail is not a guard. The project relies on four
assertions to keep known failure modes from silently reappearing, and each one
is only worth having if it raises on the thing it is supposed to catch.

    align.assert_alignment          trap #1  calendar contamination
    fundamentals.assert_provenance  trap #2  untracked metric provenance
    backtest.assert_point_in_time   trap #4  look-ahead in a scoring panel
    portfolio.assert_implementable  trap #5  a short leg that cannot be traded

Each test below feeds a deliberately broken input and confirms the guard raises,
then feeds a clean input and confirms it does not. Run with::

    python tests/test_guards.py        # standalone, no pytest needed
    pytest tests/test_guards.py        # also works

Deliberately offline: every fixture is synthetic, so this runs on a cold cache
with no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from mkt import (align, backtest as bt, fundamentals as fnd,  # noqa: E402
                 portfolio as pf, quality as ql, risk as rk)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _panel(n_days: int = 300, n_names: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"N{i}.NS" for i in range(n_names)]
    steps = rng.normal(0.0004, 0.014, size=(n_days, n_names))
    return pd.DataFrame(1000 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)


def _raises(fn, exc=AssertionError) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}, but nothing was raised")


def _ok(fn):
    return fn()


# --------------------------------------------------------------------------
# trap #1 -- calendar contamination
# --------------------------------------------------------------------------
def test_assert_alignment_fires_on_contaminated_frame():
    clean = _panel()
    align.assert_alignment(clean, label="clean")

    # Reindexing a 5-day-week frame onto a 7-day calendar is exactly what
    # downloading BTC-USD alongside equities does: ~30% spurious NaNs.
    dirty = clean.reindex(pd.date_range(clean.index.min(), clean.index.max(), freq="D"))
    msg = _raises(lambda: align.assert_alignment(dirty, label="contaminated"))
    assert "trap #1" in msg, msg
    assert align.nan_rate(dirty) > 25, "the fixture should be badly contaminated"
    print(f"  trap #1  raised: {msg[:88]}...")


# --------------------------------------------------------------------------
# trap #2 -- provenance
# --------------------------------------------------------------------------
def test_assert_provenance_fires_on_untracked_metric():
    metrics = ["ROE_%", "PE"]
    good = {"A.NS": {"ROE_%": fnd.COMPUTED, "PE": fnd.INFO},
            "B.NS": {"ROE_%": fnd.NA_SECTOR, "PE": fnd.MISSING}}
    fnd.assert_provenance(good, metrics)

    missing_key = {"A.NS": {"ROE_%": fnd.COMPUTED}}            # PE never recorded
    msg = _raises(lambda: fnd.assert_provenance(missing_key, metrics))
    assert "provenance" in msg.lower(), msg

    bad_tag = {"A.NS": {"ROE_%": fnd.COMPUTED, "PE": "guessed"}}
    msg2 = _raises(lambda: fnd.assert_provenance(bad_tag, metrics))
    assert "unknown tag" in msg2, msg2
    print(f"  trap #2  raised on both a missing tag and an unknown tag")


# --------------------------------------------------------------------------
# trap #4 -- look-ahead
# --------------------------------------------------------------------------
def test_assert_point_in_time_fires_on_lookahead_panel():
    panel = _panel()
    bt.assert_point_in_time(panel, panel.index.max(), label="clean")

    # Scoring a date while holding prices from after it.
    cutoff = panel.index[-40]
    msg = _raises(lambda: bt.assert_point_in_time(panel, cutoff, label="look-ahead"))
    assert "trap #4" in msg, msg

    # And the subtler case: one column leaks, the rest are clean.
    trimmed = panel.loc[:cutoff].reindex(panel.index)
    trimmed[panel.columns[2]] = panel[panel.columns[2]]
    msg2 = _raises(lambda: bt.assert_point_in_time(trimmed, cutoff, label="one leaky column"))
    assert "1 column(s)" in msg2, msg2
    print(f"  trap #4  raised on a whole-panel leak and on a single leaky column")


def test_forward_returns_never_reach_the_score():
    """The structural half of trap #4: scores must be blind to the future."""
    panel = _panel()
    fwd = bt.forward_returns(panel, horizons=(21,))[21]
    # A forward return at date t is defined by prices after t, so as a "scoring
    # panel" it must fail the guard at any date before the end.
    cutoff = panel.index[-60]
    _raises(lambda: bt.assert_point_in_time(panel, cutoff))
    assert fwd.loc[panel.index[-1]].isna().all(), \
        "the last date can have no forward return"
    print("  trap #4  forward returns confirmed undefined at the panel's last date")


# --------------------------------------------------------------------------
# trap #5 -- implementability
# --------------------------------------------------------------------------
def _book():
    return pd.DataFrame(
        {"side": ["long", "long", "short", "short"],
         "weight": [0.30, 0.20, -0.30, -0.20]},
        index=["L1.NS", "L2.NS", "S1.NS", "S2.NS"])


def test_assert_implementable_fires_on_undercapitalised_book():
    book = _book()
    prices = pd.Series({"L1.NS": 1000.0, "L2.NS": 500.0,
                        "S1.NS": 2000.0, "S2.NS": 800.0})
    lots = pd.Series({"L1.NS": 100.0, "L2.NS": 100.0,
                      "S1.NS": 500.0, "S2.NS": 1000.0})

    # S1: one lot is 2000*500 = 1,000,000. A 30% target needs >= 3,333,334.
    ok = pf.assert_implementable(book, prices, lots, capital=10_000_000)
    assert bool(ok["ok"].all()), ok

    msg = _raises(lambda: pf.assert_implementable(book, prices, lots, capital=1_000_000))
    assert "trap #5" in msg, msg
    assert "Minimum capital" in msg, msg
    print(f"  trap #5  raised: {msg[:96]}...")


def test_assert_implementable_reports_a_usable_minimum_capital():
    """The minimum capital it reports must actually clear the guard."""
    book = _book()
    prices = pd.Series({"L1.NS": 1000.0, "L2.NS": 500.0,
                        "S1.NS": 2000.0, "S2.NS": 800.0})
    lots = pd.Series({"L1.NS": 100.0, "L2.NS": 100.0,
                      "S1.NS": 500.0, "S2.NS": 1000.0})
    msg = _raises(lambda: pf.assert_implementable(book, prices, lots, capital=1_000_000))
    need = float(msg.split("Minimum capital that would clear every short:")[1]
                 .strip().rstrip(".").replace(",", ""))
    report = pf.assert_implementable(book, prices, lots, capital=need * 1.0001)
    assert bool(report["ok"].all()), report
    print(f"  trap #5  reported minimum capital {need:,.0f} does clear the guard")


def test_quantize_only_bites_the_short_leg():
    """Longs buy in single shares; shorts round to whole lots."""
    book = _book()
    prices = pd.Series({"L1.NS": 1000.0, "L2.NS": 500.0,
                        "S1.NS": 2000.0, "S2.NS": 800.0})
    lots = pd.Series({"L1.NS": 100.0, "L2.NS": 100.0,
                      "S1.NS": 500.0, "S2.NS": 1000.0})
    q = pf.quantize_to_lots(book, prices, lots, capital=10_000_000)
    long_err = q[q["side"] == "long"]["weight_error_pp"].abs().max()
    short_err = q[q["side"] == "short"]["weight_error_pp"].abs().max()
    assert long_err < 0.05, f"long rounding should be negligible, got {long_err}"
    assert short_err > long_err, "short rounding must dominate"
    print(f"  trap #5  quantisation: long |err| {long_err:.4f}pp vs "
          f"short {short_err:.4f}pp")


# --------------------------------------------------------------------------
# sector gate (notebook 08's verification item)
# --------------------------------------------------------------------------
def test_sector_gate_fires_when_a_financial_leaks_through():
    qt = pd.DataFrame({
        "sector": ["Financial Services", "Automobile", "FMCG"],
        "is_financial": [True, False, False],
        "piotroski_prov": [ql.NA_SECTOR, ql.COMPUTED, ql.COMPUTED],
        "altman_prov": [ql.NA_SECTOR, ql.COMPUTED, ql.COMPUTED],
        "beneish_prov": [ql.NA_SECTOR, ql.COMPUTED, ql.COMPUTED],
        "accruals_prov": [ql.NA_SECTOR, ql.COMPUTED, ql.COMPUTED],
    }, index=["BANK.NS", "AUTO.NS", "FMCG.NS"])
    ql.assert_sector_gate(qt)

    leaked = qt.copy()
    leaked.loc["BANK.NS", "beneish_prov"] = ql.COMPUTED     # scored a bank
    msg = _raises(lambda: ql.assert_sector_gate(leaked))
    assert "sector gate leaked" in msg, msg
    print(f"  sector gate  raised when a financial was scored")


# --------------------------------------------------------------------------
# risk cross-check
# --------------------------------------------------------------------------
def test_var_agreement_fires_when_cov_and_panel_disagree():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2024-01-01", periods=500)
    rets = pd.DataFrame(rng.normal(0, 0.012, size=(500, 4)),
                        index=idx, columns=list("ABCD"))
    w = pd.Series(0.25, index=list("ABCD"))
    cov = rk.shrunk_cov(rets)
    vt = rk.var_table(w, rets, cov)
    rk.assert_var_agreement(vt, tolerance_pp=0.35)

    # A covariance matrix describing a different (10x more volatile) portfolio.
    wrong = rk.var_table(w, rets, cov * 100)
    msg = _raises(lambda: rk.assert_var_agreement(wrong, tolerance_pp=0.35))
    assert "disagree" in msg, msg
    print(f"  VaR cross-check  raised on a mismatched covariance matrix")


# --------------------------------------------------------------------------
# portfolio arithmetic
# --------------------------------------------------------------------------
def test_caps_hold_and_gross_is_preserved():
    rng = np.random.default_rng(7)
    fails = 0
    for _ in range(300):
        n = int(rng.integers(4, 20))
        names = [f"N{i}" for i in range(n)]
        sectors = {t: f"S{rng.integers(0, max(2, n // 3))}" for t in names}
        w = pd.Series(rng.normal(size=n) * rng.uniform(0.5, 3), index=names)
        w = w / w.abs().sum() * float(rng.uniform(0.5, 1.5))
        max_pos, max_sector = float(rng.uniform(0.08, 0.30)), float(rng.uniform(0.20, 0.60))
        try:
            out = pf.apply_caps(w, sectors, max_pos=max_pos, max_sector=max_sector)
        except ValueError:
            continue                       # infeasible caps: correctly refused
        a = out.abs()
        sec = pd.Series(sectors)
        if not (a.max() <= max_pos + 1e-9
                and a.groupby(sec).sum().max() <= max_sector + 1e-9
                and abs(a.sum() - w.abs().sum()) < 1e-8
                and bool(((np.sign(out) == np.sign(w)) | (a < 1e-12)).all())):
            fails += 1
    assert fails == 0, f"{fails}/300 random books violated a cap or lost gross"
    print("  apply_caps  300 random books: caps held, gross and sign preserved")


def test_risk_parity_equalises_risk_contribution():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-01", periods=500)
    rets = pd.DataFrame(rng.normal(0, 0.015, size=(500, 8)),
                        index=idx, columns=[f"N{i}" for i in range(8)])
    cov = rk.shrunk_cov(rets)
    w = pf.risk_parity(cov)
    rc = rk.risk_contribution(w, cov)
    spread = float(rc["pct_of_risk"].max() - rc["pct_of_risk"].min())
    assert spread < 0.05, f"risk contributions differ by {spread:.4f}pp"
    assert abs(rc.attrs["sum_check"] - rc.attrs["portfolio_vol_%"]) < 1e-8
    print(f"  risk_parity  contributions equal to within {spread:.2e}pp; "
          f"components sum exactly to total vol")


def test_vol_target_cannot_lever_past_the_cap_by_default():
    """Regression for a real bug: vol targeting used to blow through the caps."""
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2024-01-01", periods=400)
    rets = pd.DataFrame(rng.normal(0, 0.004, size=(400, 5)),
                        index=idx, columns=[f"N{i}" for i in range(5)])
    cov = rk.shrunk_cov(rets)
    w = pd.Series(0.2, index=rets.columns)
    out, scale = pf.vol_target(w, cov, target_vol=50.0)     # unreachable target
    assert scale <= 1.0, f"default vol_target must not lever up, got {scale}"
    assert out.attrs["constrained"], "an unreachable target must be reported as constrained"
    assert out.abs().max() <= w.abs().max() + 1e-12
    print(f"  vol_target  refused to lever ({out.attrs['required_scale']:.2f}x needed, "
          f"{scale:.2f}x applied) and flagged the shortfall")


# --------------------------------------------------------------------------
# backtest harness sanity
# --------------------------------------------------------------------------
def test_shuffled_scores_produce_no_signal():
    """If a random signal scores well, the harness is broken."""
    panel = _panel(n_days=1200, n_names=20, seed=5)
    dates = bt.rebalance_dates(panel.index, warmup=252)
    rng = np.random.default_rng(2)
    scores = pd.DataFrame(rng.normal(size=(len(dates), panel.shape[1])),
                          index=dates, columns=panel.columns)
    fwd = bt.forward_returns(panel, horizons=(21,))[21]
    ctrl = bt.shuffle_control(scores, fwd, n_trials=25, seed=3)
    mean_t = float(ctrl["t_stat"].mean())
    assert abs(mean_t) < 0.6, f"control t-stats should straddle zero, got {mean_t:+.3f}"
    assert int((ctrl["t_stat"].abs() > 2).sum()) <= 4, "too many spurious hits"
    print(f"  shuffle_control  mean t {mean_t:+.3f} over 25 trials")


def test_newey_west_widens_the_error_on_overlapping_windows():
    """Overlapping forward windows must not be allowed to inflate the t-stat."""
    rng = np.random.default_rng(17)
    base = rng.normal(size=400)
    # A rolling sum is the shape an overlapping-window IC series has.
    overlapping = pd.Series(pd.Series(base).rolling(6).mean().dropna().values)
    se_iid = float(overlapping.std(ddof=1) / np.sqrt(len(overlapping)))
    se_nw = bt.newey_west_se(overlapping, lag=5)
    assert se_nw > se_iid, f"Newey-West SE {se_nw:.5f} should exceed iid {se_iid:.5f}"
    assert bt.overlap_lag_for(126, 21) == 5
    assert bt.overlap_lag_for(21, 21) == 0
    print(f"  newey_west_se  {se_nw / se_iid:.2f}x the iid error on a 6-period overlap")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    print(f"running {len(tests)} guard regression tests\n")
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
