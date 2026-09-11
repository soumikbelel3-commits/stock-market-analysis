# Investment Committee Memo

**Book:** India long/short, Nifty 50 universe  
**Capital:** INR 1.00 cr  
**Prepared:** 2026-09-09 20:37 UTC  
**Run id:** `77dc11b7f0d9cf64`

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

The factor scan runs 15 tests on one universe. Under the null the *largest* |t| in a family that size lands near 1.68, and family-wise significance requires |t| > 2.69 -- not 1.96.

- Naively significant: **6**
- Survive false-discovery control: **5**
- Survive family-wise control: **0**


Deflated Sharpe probability: **0.000** (needs 0.95 to support skill after accounting for the search).


---

## 2. The book

- Gross **1.00x**, net **-0.08x**
- 6 long, 5 short, largest position 12.0%

- Net beta **+0.012** (beta-neutral)



| ticker | side | weight | beta |
|---|---|---|---|
| DRREDDY.NS | short | -0.1200 | 0.5745 |
| TATACONSUM.NS | short | -0.1200 | 0.7328 |
| HINDUNILVR.NS | short | -0.1200 | 0.6105 |
| WIPRO.NS | short | -0.1127 | 0.8933 |
| ICICIBANK.NS | long | 0.0935 | 0.9162 |
| COALINDIA.NS | long | 0.0858 | 0.5608 |
| TITAN.NS | long | 0.0793 | 0.9161 |
| KOTAKBANK.NS | long | 0.0781 | 0.9870 |
| TMPV.NS | short | -0.0682 | 1.3813 |
| JSWSTEEL.NS | long | 0.0676 | 1.1378 |
| ADANIPORTS.NS | long | 0.0549 | 1.4009 |


**The short leg is stock futures, not stock.** Cash-market shorts cannot be carried overnight in India, so the short book trades in whole lots and cannot express any weight below roughly one lot of capital (trap #5).


---

## 3. What can it lose?

- Ex-ante volatility **6.7%** against a 12% target
- 1-day VaR (95%) **0.81%** = INR 80,912
- 1-day CVaR (95%) **1.19%** = INR 1.19 lakh

- VaR model backtest: **GREEN zone** (122 exceptions against 111 expected)

- Ex-ante vs ex-post bias **1.053** (1.00 is unbiased; above 1 means positions are larger than intended)


### What is it a bet on?

Style factors explain **34%** of the book's variance; the remaining **66%** is stock-specific.


A market-neutral book that is mostly systematic has not removed risk, it has changed which risk it holds.


### Crisis replay -- today's weights through real episodes

| episode | start | end | book_return_% | names_covered | note |
|---|---|---|---|---|---|
| GFC / Lehman | 2008-09-01 | 2009-03-09 |  | 0 | outside the panel's history |
| Taper tantrum | 2013-05-22 | 2013-09-04 |  | 0 | outside the panel's history |
| 2018 tightening | 2018-10-01 | 2018-12-24 | 1.76 | 11 |  |
| COVID crash | 2020-02-19 | 2020-03-23 | -2.03 | 11 |  |
| Inflation shock | 2021-11-01 | 2022-06-16 | 4.13 | 11 |  |
| Hiking cycle / SVB | 2023-03-08 | 2023-03-20 | 0.24 | 11 |  |


---

## 4. Limits

**13 limits in the register; 0 hard breach(es), 0 soft, 0 not measured.**


| metric | operator | threshold | severity | value | status | headroom | headroom_% |
|---|---|---|---|---|---|---|---|
| max_position_weight | <= | 0.120 | hard | 0.120 | ok | 0.000 | 0.000 |
| max_sector_weight | <= | 0.300 | hard | 0.240 | ok | 0.060 | 20.000 |
| gross_exposure | <= | 2.000 | hard | 1.000 | ok | 1.000 | 50.000 |
| net_exposure_abs | <= | 0.500 | hard | 0.082 | ok | 0.418 | 83.623 |
| worst_stress_loss_% | >= | -20.000 | hard | -2.028 | ok | 17.972 | 89.860 |
| margin_utilisation | <= | 0.650 | hard | 0.043 | ok | 0.607 | 93.372 |
| pct_risk_top_name | <= | 25.000 | soft | 17.872 | ok | 7.128 | 28.511 |
| ex_ante_vol_% | <= | 15.000 | soft | 6.682 | ok | 8.318 | 55.455 |
| cvar_95_1d_% | <= | 3.500 | soft | 1.189 | ok | 2.311 | 66.028 |
| var_95_1d_% | <= | 2.500 | soft | 0.809 | ok | 1.691 | 67.635 |
| net_beta_abs | <= | 0.200 | soft | 0.012 | ok | 0.188 | 93.916 |
| days_to_liquidate_p95 | <= | 5.000 | soft | 0.009 | ok | 4.991 | 99.820 |
| effective_bets | >= | 4.000 | soft | 9.056 | ok | 5.056 | 126.403 |


Within 10% of a limit without breaching: **max_position_weight**.


---

## 5. Costs, capacity and capital

- All-in round trip, weighted **25.3 bps** = INR 25,267 on this book
- Long leg is cash delivery (STT both sides); short leg is futures (STT on the sell only) -- the statutory cost of the two legs differs by roughly 4x


### Capital adequacy
- Long leg cash INR 45.91 lakh
- Short-leg margin (SPAN approximated + ELM) INR 4.31 lakh
- Mark-to-market buffer, 3 days INR 1.31 lakh
- Free cash **48.5%**, margin utilisation **4.3%**

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

- Run id `77dc11b7f0d9cf64`, config fingerprint `909377f149627e78`
- mkt 2.0.0 on Python 3.14.2, pandas 3.0.3
- Chain links published 14/14; oldest signal 15.4 h old
- Risk-free 6.25% as of 2026-09-09; cost rates as of 2026-09-09; lot sizes as of 2026-09-08


_All figures are derived live and move on every run. Re-read the run id before quoting any number in this memo._
