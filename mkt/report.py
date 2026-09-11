"""The investment committee memo and the risk tearsheet.

Why this module exists
----------------------
Eleven notebooks and a JSON dashboard are a research process. They are not a
deliverable. Nobody approves a book by scrolling a notebook, and the discipline of
having to state a case in one page is not presentational -- it is where a chain of
individually defensible steps either adds up to a decision or visibly does not.

Two documents, with different jobs:

* ``ic_memo`` -- the case for the book. What it holds, why, what could go wrong,
  what would falsify it, and what the process itself says about how much to trust
  it. Written to ``outputs/ic_memo.md``.
* ``tearsheet`` -- the numbers, in the order an allocator asks for them.

The house rule
--------------
The memo is generated from the run's own artefacts and **may not overstate them**.
When notebook 07's verdict is "Not proven", the memo says so at the top, in the
recommendation line, not in a footnote -- and ``assert_memo_honesty`` fails the
build if the two ever disagree. A memo that can quietly diverge from its evidence
is worse than no memo, because it launders the evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from . import config, dashboard

RULE = "\n---\n"


# ==========================================================================
# Helpers
# ==========================================================================
def _fmt(v, spec: str = ".2f", dash: str = "n/a") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return dash if not np.isfinite(f) else format(f, spec)


def _inr(v) -> str:
    """Indian numbering, because a book in India is discussed in lakh and crore."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    sign = "-" if f < 0 else ""
    f = abs(f)
    if f >= 1e7:
        return f"{sign}INR {f / 1e7:,.2f} cr"
    if f >= 1e5:
        return f"{sign}INR {f / 1e5:,.2f} lakh"
    return f"{sign}INR {f:,.0f}"


def _table(df: pd.DataFrame, max_rows: int = 12, floatfmt: str = "{:.3f}") -> str:
    """Markdown table from a frame, without adding a dependency."""
    if df is None or len(df) == 0:
        return "_no data_\n"
    d = df.head(max_rows).copy()
    d = d.reset_index()
    cols = [str(c) for c in d.columns]

    def cell(x):
        if isinstance(x, float):
            return "" if not np.isfinite(x) else floatfmt.format(x)
        return str(x)

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"| _... {len(df) - max_rows} more rows_ |" +
                     " |" * (len(cols) - 1))
    return "\n".join(lines) + "\n"


# ==========================================================================
# The verdict
# ==========================================================================
def signal_verdict(data: Mapping | None = None) -> dict:
    """Read notebook 07's verdict and turn it into a recommendation stance.

    This is the single most important function in the module, because it is what
    stops the memo from being marketing. The stance is *derived* from the
    validation result rather than chosen: a screen that has not been shown to
    predict returns cannot produce a "deploy" recommendation, however good the
    construction and risk work downstream of it are.
    """
    data = dashboard.read() if data is None else data
    node = data.get("07_signal_validation", {}) or {}
    verdict = str(node.get("verdict", node.get("signal", ""))).strip()
    low = verdict.lower()

    if not verdict:
        stance, why = "HOLD -- not validated", (
            "notebook 07 has not published a verdict; the screen is untested on "
            "this run")
    elif "not proven" in low or "not validated" in low or "fail" in low:
        stance, why = "DO NOT DEPLOY -- paper only", (
            "notebook 07 could not distinguish the technical leg's information "
            "coefficient from zero")
    elif "proven" in low or "validated" in low or "pass" in low:
        stance, why = "DEPLOY AT REDUCED SIZE", (
            "the technical leg cleared validation; the fundamental leg remains "
            "untestable (trap #4), so only half the composite is evidenced")
    else:
        stance, why = f"REVIEW -- verdict '{verdict}'", "verdict not recognised"

    # Notebook 07 publishes the IC statistics nested under `ic_by_horizon`, keyed
    # by horizon. Reading only the top level silently produced a memo that said
    # "mean rank IC n/a" beside a verdict derived from that very number -- the
    # headline was right and its evidence was blank, which is the sort of gap a
    # reader is entitled to assume does not exist.
    mean_ic, t_stat, horizon = node.get("mean_ic"), node.get("t_stat"), None
    by_h = node.get("ic_by_horizon") or {}
    if mean_ic is None and isinstance(by_h, dict) and by_h:
        try:
            horizon = sorted(by_h, key=lambda k: int(k))[0]
        except (TypeError, ValueError):
            horizon = next(iter(by_h))
        blk = by_h.get(horizon) or {}
        mean_ic, t_stat = blk.get("mean_ic"), blk.get("t_stat")

    return {
        "verdict": verdict or "(none published)",
        "stance": stance,
        "rationale": why,
        "mean_ic": mean_ic,
        "t_stat": t_stat,
        "horizon_days": horizon,
        "n_rebalances": node.get("n_periods", node.get("rebalances")),
    }


# ==========================================================================
# The memo
# ==========================================================================
def ic_memo(book: pd.DataFrame | None = None,
            perf_block: Mapping | None = None,
            risk_block: Mapping | None = None,
            factor_block: Mapping | None = None,
            limits_table: pd.DataFrame | None = None,
            cost_table: pd.DataFrame | None = None,
            margin_block: Mapping | None = None,
            validation_block: Mapping | None = None,
            stress_table: pd.DataFrame | None = None,
            manifest_block: Mapping | None = None,
            capital: float = config.DEFAULT_CAPITAL,
            path=None, write: bool = True) -> str:
    """Assemble the memo. Every section degrades to a stated absence.

    Missing inputs produce "not measured on this run" rather than a silent gap --
    the same principle as ``limits.monitor``. A reader must be able to tell the
    difference between a risk that was measured and found small and a risk nobody
    looked at.
    """
    data = dashboard.read()
    sv = signal_verdict(data)
    now = datetime.now(timezone.utc)

    parts: list[str] = []
    a = parts.append

    # ---------------------------------------------------------------- header
    a(f"# Investment Committee Memo\n")
    a(f"**Book:** India long/short, Nifty 50 universe  \n"
      f"**Capital:** {_inr(capital)}  \n"
      f"**Prepared:** {now:%Y-%m-%d %H:%M} UTC  \n"
      f"**Run id:** `{(manifest_block or {}).get('run_id', 'not recorded')}`\n")

    a(f"> ## Recommendation: {sv['stance']}\n>\n"
      f"> {sv['rationale'].capitalize()}.\n")
    a("This memo is generated from the run's own artefacts. It is arithmetic on "
      "the chain's output and a description of a model book -- not investment "
      "advice, and not a recommendation to trade.\n")
    a(RULE)

    # ------------------------------------------------------- 1. the evidence
    a("## 1. Does the signal work?\n")
    horizon = (f" at a {sv['horizon_days']}-day horizon"
               if sv.get("horizon_days") else "")
    a(f"Notebook 07's verdict on the technical leg: **{sv['verdict']}**. "
      f"Mean rank IC {_fmt(sv['mean_ic'], '.4f')}{horizon} with t = "
      f"{_fmt(sv['t_stat'], '.2f')} over {_fmt(sv['n_rebalances'], '.0f')} "
      f"rebalances.\n")
    a("Two asymmetries govern how much of this book is evidenced at all:\n\n"
      "1. **The technical leg is backtested properly** -- every input derives from "
      "prices known on the scoring date, proven per run by "
      "`backtest.assert_point_in_time`.\n"
      "2. **The fundamental leg is not backtested at all.** `yfinance` serves only "
      "current statements, so there is no point-in-time history to test against "
      "(trap #4). Its weight in the composite is a stated 50/50 prior, never a "
      "fitted result.\n")

    if validation_block:
        a("\n### Multiple-testing control\n")
        a(f"The factor scan runs {validation_block.get('n_tests', 'n')} tests on one "
          f"universe. Under the null the *largest* |t| in a family that size lands "
          f"near {_fmt(validation_block.get('median_max_t_under_null'), '.2f')}, and "
          f"family-wise significance requires "
          f"|t| > {_fmt(validation_block.get('family_critical_t'), '.2f')} -- "
          f"not 1.96.\n\n"
          f"- Naively significant: **{validation_block.get('naive_significant', 'n/a')}**\n"
          f"- Survive false-discovery control: "
          f"**{validation_block.get('survives_fdr', 'n/a')}**\n"
          f"- Survive family-wise control: "
          f"**{validation_block.get('survives_fwer', 'n/a')}**\n")
        if validation_block.get("deflated_sharpe_prob") is not None:
            a(f"\nDeflated Sharpe probability: "
              f"**{_fmt(validation_block.get('deflated_sharpe_prob'), '.3f')}** "
              f"(needs 0.95 to support skill after accounting for the search).\n")
        if validation_block.get("pbo") is not None:
            a(f"Probability of backtest overfitting: "
              f"**{_fmt(validation_block.get('pbo'), '.2f')}** "
              f"(above 0.5 means selection is subtracting information). "
              f"Read as a band of about +/-0.3, not a point estimate.\n")
    else:
        a("\n_Multiple-testing control not measured on this run._\n")
    a(RULE)

    # ------------------------------------------------------------ 2. the book
    a("## 2. The book\n")
    if book is not None and len(book):
        w = pd.to_numeric(book["weight"], errors="coerce")
        a(f"- Gross **{_fmt(w.abs().sum(), '.2f')}x**, "
          f"net **{_fmt(w.sum(), '+.2f')}x**\n"
          f"- {int((w > 0).sum())} long, {int((w < 0).sum())} short, "
          f"largest position {_fmt(w.abs().max() * 100, '.1f')}%\n")
        if "beta_contrib" in book.columns:
            net_beta = pd.to_numeric(book["beta_contrib"], errors="coerce").sum()
            method = book.attrs.get("method", "method not recorded")
            a(f"- Net beta **{_fmt(net_beta, '+.3f')}** ({method})\n")
        a("\n")
        show = book.reindex(w.abs().sort_values(ascending=False).index)
        a(_table(show[[c for c in ("side", "weight", "beta") if c in show.columns]],
                 max_rows=15, floatfmt="{:.4f}"))
        a("\n**The short leg is stock futures, not stock.** Cash-market shorts "
          "cannot be carried overnight in India, so the short book trades in whole "
          "lots and cannot express any weight below roughly one lot of capital "
          "(trap #5).\n")
    else:
        a("_No book supplied to this run._\n")
    a(RULE)

    # ----------------------------------------------------------- 3. the risk
    a("## 3. What can it lose?\n")
    if risk_block:
        a(f"- Ex-ante volatility **{_fmt(risk_block.get('ex_ante_vol_%'), '.1f')}%** "
          f"against a {_fmt(config.TARGET_PORTFOLIO_VOL_PCT, '.0f')}% target\n"
          f"- 1-day VaR (95%) **{_fmt(risk_block.get('var_95_1d_%'), '.2f')}%** "
          f"= {_inr((risk_block.get('var_95_1d_%') or 0) / 100 * capital)}\n"
          f"- 1-day CVaR (95%) **{_fmt(risk_block.get('cvar_95_1d_%'), '.2f')}%** "
          f"= {_inr((risk_block.get('cvar_95_1d_%') or 0) / 100 * capital)}\n")
        if risk_block.get("var_backtest_zone"):
            a(f"- VaR model backtest: **{risk_block['var_backtest_zone'].upper()} zone** "
              f"({risk_block.get('var_exceptions', 'n/a')} exceptions against "
              f"{_fmt(risk_block.get('var_expected'), '.0f')} expected)\n")
        if risk_block.get("bias") is not None:
            a(f"- Ex-ante vs ex-post bias **{_fmt(risk_block.get('bias'), '.3f')}** "
              f"(1.00 is unbiased; above 1 means positions are larger than intended)\n")
    else:
        a("_Risk block not supplied._\n")

    if factor_block:
        a(f"\n### What is it a bet on?\n")
        a(f"Style factors explain **{_fmt(factor_block.get('systematic_share_%'), '.0f')}%** "
          f"of the book's variance; the remaining "
          f"**{_fmt(factor_block.get('specific_share_%'), '.0f')}%** is stock-specific.\n")
        if factor_block.get("top_exposures") is not None:
            a("\n" + _table(factor_block["top_exposures"], max_rows=8))
        a("\nA market-neutral book that is mostly systematic has not removed risk, "
          "it has changed which risk it holds.\n")

    if stress_table is not None and len(stress_table):
        a("\n### Crisis replay -- today's weights through real episodes\n")
        keep = [c for c in ("start", "end", "book_return_%", "names_covered", "note")
                if c in stress_table.columns]
        a(_table(stress_table[keep], max_rows=8, floatfmt="{:.2f}"))
    a(RULE)

    # --------------------------------------------------------- 4. the limits
    a("## 4. Limits\n")
    if limits_table is not None and len(limits_table):
        n_hard = limits_table.attrs.get("n_hard_breach", 0)
        n_soft = limits_table.attrs.get("n_soft_breach", 0)
        n_un = limits_table.attrs.get("n_not_measured", 0)
        a(f"**{limits_table.attrs.get('n_limits', len(limits_table))} limits in the "
          f"register; {n_hard} hard breach(es), {n_soft} soft, {n_un} not measured.**\n\n")
        breached = limits_table[limits_table["status"] != "ok"]
        a(_table(breached if len(breached) else limits_table,
                 max_rows=14, floatfmt="{:.3f}"))
        near = limits_table.attrs.get("near_limit", [])
        if near:
            a(f"\nWithin 10% of a limit without breaching: **{', '.join(near)}**.\n")
    else:
        a("_Limit monitor not run._\n")
    a(RULE)

    # ---------------------------------------------- 5. costs and implementation
    a("## 5. Costs, capacity and capital\n")
    if cost_table is not None and len(cost_table):
        a(f"- All-in round trip, weighted **"
          f"{_fmt(cost_table.attrs.get('weighted_avg_bps'), '.1f')} bps** "
          f"= {_inr(cost_table.attrs.get('total_INR'))} on this book\n"
          f"- Long leg is cash delivery (STT both sides); short leg is futures "
          f"(STT on the sell only) -- the statutory cost of the two legs differs "
          f"by roughly 4x\n")
        if cost_table.attrs.get("n_extrapolating", 0):
            a(f"- **{cost_table.attrs['n_extrapolating']} position(s) exceed the "
              f"participation rate the impact model was calibrated on** -- their "
              f"cost is an extrapolation\n")
    else:
        a("_Cost model not run._\n")

    if margin_block:
        a(f"\n### Capital adequacy\n"
          f"- Long leg cash {_inr(margin_block.get('long_leg_cash'))}\n"
          f"- Short-leg margin (SPAN approximated + ELM) "
          f"{_inr(margin_block.get('short_leg_margin'))}\n"
          f"- Mark-to-market buffer, "
          f"{margin_block.get('mtm_buffer_days', '?')} days "
          f"{_inr(margin_block.get('mtm_buffer'))}\n"
          f"- Free cash **{_fmt(margin_block.get('free_cash_%'), '.1f')}%**, "
          f"margin utilisation "
          f"**{_fmt(margin_block.get('margin_utilisation'), '.1%')}**\n\n"
          f"SPAN is computed by the exchange from a proprietary risk array and "
          f"cannot be reproduced from public data. The figure above is a documented "
          f"approximation -- the scan-range proxy plus an exact ELM -- and should be "
          f"read as an early warning, not as a broker's number.\n")
    a(RULE)

    # ------------------------------------------------------- 6. falsification
    a("## 6. What would change this view\n")
    a("Stated in advance, so the answer cannot be chosen after the fact:\n\n"
      "- **The screen becomes deployable** if the technical leg's rank IC clears "
      "family-wise significance out of sample on a wider universe -- "
      "`config.BACKTEST_UNIVERSE` is a one-line change and 50 names is too thin "
      "for cross-sectional statistics.\n"
      "- **The book comes off** if a hard limit breaches, if the VaR model enters "
      "the Basel red zone, or if margin utilisation exceeds "
      f"{config.MARGIN_UTILISATION_LIMIT:.0%} at 2x volatility.\n"
      "- **The cost assumption fails** if realised slippage exceeds the modelled "
      "impact on names inside the calibrated participation range.\n"
      "- **The short-reversal pattern** found in notebook 07 is a hypothesis for "
      "out-of-sample testing, explicitly *not* a licence to refit the weights on "
      "the sample that produced it.\n")
    a(RULE)

    # ------------------------------------------------------- 7. reproducibility
    a("## 7. Reproducibility\n")
    if manifest_block:
        env = manifest_block.get("environment", {})
        cfg = manifest_block.get("config", {})
        ch = manifest_block.get("chain_state", {})
        a(f"- Run id `{manifest_block.get('run_id', '')}`, "
          f"config fingerprint `{cfg.get('fingerprint', '')}`\n"
          f"- mkt {env.get('mkt_version', '')} on Python {env.get('python', '')}, "
          f"pandas {env.get('packages', {}).get('pandas', '')}\n"
          f"- Chain links published {ch.get('notebooks_present', 0)}/"
          f"{ch.get('notebooks_expected', 0)}; oldest signal "
          f"{_fmt(ch.get('max_stale_hours'), '.1f')} h old\n"
          f"- Risk-free {config.RISK_FREE_ANNUAL_PCT}% as of "
          f"{config.RISK_FREE_ASOF}; cost rates as of {config.COST_ASOF}; "
          f"lot sizes as of {config.LOT_SIZE_FALLBACK_ASOF}\n")
    else:
        a("_No manifest recorded for this run._\n")

    a("\n_All figures are derived live and move on every run. "
      "Re-read the run id before quoting any number in this memo._\n")

    text = "\n".join(parts)
    if write:
        p = config.IC_MEMO if path is None else path
        p.write_text(text, encoding="utf-8")
    return text


# ==========================================================================
# Tearsheet
# ==========================================================================
def tearsheet(perf_block: Mapping | None = None,
              relative_block: Mapping | None = None,
              risk_block: Mapping | None = None,
              factor_block: Mapping | None = None,
              margin_block: Mapping | None = None) -> pd.DataFrame:
    """The numbers, grouped the way an allocator reads them.

    One frame rather than eight, because the point of a tearsheet is that nothing
    is on a different page from the number that qualifies it -- Sharpe next to the
    risk-free rate it was computed against, VaR next to whether the VaR model
    passed its backtest.
    """
    rows: list[tuple[str, str, object]] = []

    def add(group, label, value, spec=".3f"):
        rows.append((group, label, _fmt(value, spec) if value is not None else "n/a"))

    if perf_block:
        add("Return", "CAGR %", perf_block.get("cagr_%"), ".2f")
        add("Return", "excess of cash %", perf_block.get("excess_cagr_%"), ".2f")
        add("Return", "hit rate %", perf_block.get("hit_rate_%"), ".1f")
        add("Risk-adjusted", "Sharpe (excess)", perf_block.get("sharpe"))
        add("Risk-adjusted", "Sortino", perf_block.get("sortino"))
        add("Risk-adjusted", "Calmar", perf_block.get("calmar"))
        add("Risk-adjusted", "risk-free used %", perf_block.get("rf_annual_%"), ".2f")
        add("Risk", "annual vol %", perf_block.get("ann_vol_%"), ".2f")
        add("Risk", "max drawdown %", perf_block.get("max_drawdown_%"), ".2f")
        add("Risk", "skew", perf_block.get("skew"))
        add("Risk", "excess kurtosis", perf_block.get("excess_kurtosis"))

    if relative_block:
        add("Benchmark", "beta", relative_block.get("beta"))
        add("Benchmark", "alpha (Jensen) ann %",
            relative_block.get("jensen_alpha_ann_%"), ".2f")
        add("Benchmark", "tracking error %",
            relative_block.get("tracking_error_ann_%"), ".2f")
        add("Benchmark", "information ratio", relative_block.get("information_ratio"))
        add("Benchmark", "up capture %", relative_block.get("up_capture_%"), ".1f")
        add("Benchmark", "down capture %", relative_block.get("down_capture_%"), ".1f")

    if risk_block:
        add("Tail", "VaR 95% 1d %", risk_block.get("var_95_1d_%"), ".2f")
        add("Tail", "CVaR 95% 1d %", risk_block.get("cvar_95_1d_%"), ".2f")
        add("Tail", "VaR backtest zone", risk_block.get("var_backtest_zone"), "s")
        add("Tail", "ex-ante bias", risk_block.get("bias"))
        add("Tail", "worst crisis replay %", risk_block.get("worst_stress_loss_%"), ".2f")

    if factor_block:
        add("Attribution", "systematic share %",
            factor_block.get("systematic_share_%"), ".1f")
        add("Attribution", "specific share %",
            factor_block.get("specific_share_%"), ".1f")
        add("Attribution", "effective bets", factor_block.get("effective_bets"), ".2f")

    if margin_block:
        add("Capital", "margin utilisation",
            margin_block.get("margin_utilisation"), ".3f")
        add("Capital", "free cash %", margin_block.get("free_cash_%"), ".1f")

    out = pd.DataFrame(rows, columns=["group", "metric", "value"])
    return out.set_index(["group", "metric"]) if len(out) else out


# ==========================================================================
# Guard
# ==========================================================================
def assert_memo_honesty(memo_text: str, data: Mapping | None = None) -> pd.DataFrame:
    """Regression guard: the memo may not claim more than the evidence supports.

    Specifically: when notebook 07's verdict is negative, the memo's recommendation
    must be negative too. This is a guard against a failure that is social rather
    than technical -- the natural drift of a document towards the conclusion its
    author wants -- and it is the reason the recommendation line is *derived* from
    the verdict rather than written.

    Also refuses a memo that has lost its "not advice" framing, because a generated
    document that reads as a personal recommendation is a different object from a
    generated document that reads as arithmetic.
    """
    sv = signal_verdict(data)
    low_verdict = sv["verdict"].lower()
    problems = []

    negative_evidence = any(k in low_verdict for k in
                            ("not proven", "not validated", "fail"))
    head = memo_text[:1200].lower()
    claims_deploy = ("deploy at reduced size" in head or
                     ("## recommendation: deploy" in head and "do not" not in head))
    if negative_evidence and claims_deploy:
        problems.append(
            f"notebook 07's verdict is {sv['verdict']!r} but the memo recommends "
            f"deployment -- the recommendation must be derived from the validation "
            f"result, not written over it")

    if "not investment advice" not in memo_text.lower():
        problems.append("memo has lost its 'not investment advice' framing")

    if negative_evidence and "trap #4" not in memo_text.lower():
        problems.append("memo omits the point-in-time limitation that makes half "
                        "the composite untestable (trap #4)")

    if problems:
        raise AssertionError("ic_memo: " + "; ".join(problems))

    return pd.DataFrame([{
        "verdict": sv["verdict"],
        "stance": sv["stance"],
        "memo_chars": len(memo_text),
        "result": "memo is consistent with the evidence it cites",
    }])
