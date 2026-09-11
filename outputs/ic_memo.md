# Investment Committee Memo

**Book:** India long/short, Nifty 50 universe  
**Capital:** INR 1.00 cr  
**Prepared:** 2026-09-11 07:21 UTC  
**Run id:** `da4113493a7dc63a`

> ## Recommendation: DO NOT DEPLOY -- paper only
>
> Notebook 07 could not distinguish the technical leg's information coefficient from zero.

This memo is generated from the run's own artefacts. It is arithmetic on the chain's output and a description of a model book -- not investment advice, and not a recommendation to trade.


---

## 1. Does the signal work?

Notebook 07's verdict on the technical leg: **Technical leg: Not proven**. Mean rank IC -0.0368 at a 21-day horizon with t = -1.62 over 109 rebalances.

Two asymmetries govern how much of this book is evidenced at all:

1. **The technical leg is backtested properly** -- every input derives from prices known on the scoring date, proven per run by `backtest.assert_point_in_time`.
2. **The fundamental leg is not backtested at all.** `yfinance` serves only current statements, so there is no point-in-time history to test against (trap #4). Its weight in the composite is a stated 50/50 prior, never a fitted result.


### Multiple-testing control

The factor scan runs 15 tests on one universe. Under the null the *largest* |t| in a family that size lands near 1.67, and family-wise significance requires |t| > 2.68 -- not 1.96.

- Naively significant: **6**
- Survive false-discovery control: **5**
- Survive family-wise control: **0**


Deflated Sharpe probability: **0.000** (needs 0.95 to support skill after accounting for the search).


---

## 2. The book

- Gross **0.82x**, net **-0.08x**
- 6 long, 5 short, largest position 11.6%

- Net beta **-0.000** (beta-neutral)



| ticker | side | weight | beta |
|---|---|---|---|
| HINDUNILVR.NS | short | -0.1162 | 0.6117 |
| TATACONSUM.NS | short | -0.1094 | 0.7346 |
| WIPRO.NS | short | -0.1002 | 0.8936 |
| ICICIBANK.NS | long | 0.0748 | 0.9153 |
| DRREDDY.NS | short | -0.0736 | 0.5679 |
| COALINDIA.NS | long | 0.0686 | 0.5594 |
| TITAN.NS | long | 0.0634 | 0.9134 |
| KOTAKBANK.NS | long | 0.0625 | 0.9887 |
| JSWSTEEL.NS | long | 0.0541 | 1.1386 |
| TMPV.NS | short | -0.0484 | 1.3730 |
| ADANIPORTS.NS | long | 0.0438 | 1.3916 |


**The short leg is stock futures, not stock.** Cash-market shorts cannot be carried overnight in India, so the short book trades in whole lots and cannot express any weight below roughly one lot of capital (trap #5).


---

## 3. What can it lose?

- Ex-ante volatility **5.5%** against a 12% target
- 1-day VaR (95%) **0.66%** = INR 66,395
- 1-day CVaR (95%) **0.97%** = INR 97,413

- VaR model backtest: **GREEN zone** (119 exceptions against 111 expected)

- Ex-ante vs ex-post bias **1.050** (1.00 is unbiased; above 1 means positions are larger than intended)


### What is it a bet on?

Style factors explain **33%** of the book's variance; the remaining **67%** is stock-specific.


A market-neutral book that is mostly systematic has not removed risk, it has changed which risk it holds.


### Crisis replay -- today's weights through real episodes

| episode | start | end | book_return_% | names_covered | note |
|---|---|---|---|---|---|
| GFC / Lehman | 2008-09-01 | 2009-03-09 |  | 0 | outside the panel's history |
| Taper tantrum | 2013-05-22 | 2013-09-04 |  | 0 | outside the panel's history |
| 2018 tightening | 2018-10-01 | 2018-12-24 | 1.25 | 11 |  |
| COVID crash | 2020-02-19 | 2020-03-23 | -1.09 | 11 |  |
| Inflation shock | 2021-11-01 | 2022-06-16 | 3.68 | 11 |  |
| Hiking cycle / SVB | 2023-03-08 | 2023-03-20 | 0.23 | 11 |  |


---

## 4. Limits

**13 limits in the register; 0 hard breach(es), 0 soft, 0 not measured.**


| metric | operator | threshold | severity | value | status | headroom | headroom_% |
|---|---|---|---|---|---|---|---|
| max_position_weight | <= | 0.120 | hard | 0.116 | ok | 0.004 | 3.185 |
| max_sector_weight | <= | 0.300 | hard | 0.226 | ok | 0.074 | 24.802 |
| gross_exposure | <= | 2.000 | hard | 0.815 | ok | 1.185 | 59.244 |
| net_exposure_abs | <= | 0.500 | hard | 0.081 | ok | 0.419 | 83.869 |
| margin_utilisation | <= | 0.650 | hard | 0.039 | ok | 0.611 | 94.024 |
| worst_stress_loss_% | >= | -20.000 | hard | -1.092 | ok | 18.908 | 94.541 |
| pct_risk_top_name | <= | 25.000 | soft | 22.518 | ok | 2.482 | 9.930 |
| ex_ante_vol_% | <= | 15.000 | soft | 5.535 | ok | 9.465 | 63.101 |
| cvar_95_1d_% | <= | 3.500 | soft | 0.974 | ok | 2.526 | 72.168 |
| var_95_1d_% | <= | 2.500 | soft | 0.664 | ok | 1.836 | 73.442 |
| days_to_liquidate_p95 | <= | 5.000 | soft | 0.008 | ok | 4.992 | 99.832 |
| net_beta_abs | <= | 0.200 | soft | 0.000 | ok | 0.200 | 99.905 |
| effective_bets | >= | 4.000 | soft | 9.060 | ok | 5.060 | 126.496 |


Within 10% of a limit without breaching: **max_position_weight, pct_risk_top_name**.


---

## 5. Costs, capacity and capital

- All-in round trip, weighted **26.5 bps** = INR 21,572 on this book
- Long leg is cash delivery (STT both sides); short leg is futures (STT on the sell only) -- the explicit cost of the two legs differs by 2.8x


### Capital adequacy
- Long leg cash INR 36.72 lakh
- Short-leg margin (SPAN approximated + ELM) INR 3.88 lakh
- Mark-to-market buffer, 3 days INR 1.12 lakh
- Free cash **58.3%**, margin utilisation **3.9%**

SPAN is computed by the exchange from a proprietary risk array and cannot be reproduced from public data. The figure above is a documented approximation -- the scan-range proxy plus an exact ELM -- and should be read as an early warning, not as a broker's number.


---

## 6. What would change this view

Stated in advance, so the answer cannot be chosen after the fact:

- **The screen becomes deployable** if the technical leg's rank IC clears family-wise significance out of sample on a wider universe -- `config.BACKTEST_UNIVERSE` is a one-line change and 50 names is too thin for cross-sectional statistics.
- **The book comes off** if a hard limit breaches, if the VaR model enters the Basel red zone, or if margin utilisation exceeds 65% at 2x volatility.
- **The cost assumption fails** if realised slippage exceeds the modelled impact on names inside the calibrated participation range.
- **The short-reversal pattern** found in notebook 07 is a hypothesis for out-of-sample testing, explicitly *not* a licence to refit the weights on the sample that produced it.


---

## 7. Reproducibility

- Run id `da4113493a7dc63a`, config fingerprint `dd25346b7f4acb8b`
- mkt 2.0.0 on Python 3.14.2, pandas 3.0.3
- Chain links published 14/14; oldest signal 50.1 h old
- Risk-free 6.25% as of 2026-09-09; cost rates as of 2026-09-11; lot sizes as of 2026-09-08


_All figures are derived live and move on every run. Re-read the run id before quoting any number in this memo._
