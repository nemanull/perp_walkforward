# Walk-Forward Tests of Short-Horizon Return Models on Five Binance Perpetuals

HYPE, TRX, DOGE, UNI and AAVE on Binance USD-M perpetual futures, June 2025 to August 2026.

Sections 1 to 5 describe the protocol, phase 2 included.
Sections 6 and 7 will hold the results.

## 1. Introduction

Most short-horizon backtests I have read go wrong in a few familiar places.
A random train and test split lets the model see the future, and overlapping labels make a few hundred independent observations look like tens of thousands.
The signal is often reported before fees, although at these horizons the fees decide the result.
And the configuration that gets reported is usually the best of many that were tried.
Sections 4.3, 4.5, 4.6 and 4.8 deal with these four in turn.

I chose five coins by type before training any model.
HYPE is the token of the Hyperliquid perpetuals exchange, TRX the token of the TRON payments chain, and DOGE the largest meme coin.
UNI and AAVE are DeFi tokens, of the Uniswap exchange and the Aave lending protocol.
BTC, ETH and SOL, the three most liquid perpetuals on the venue, are context inputs for every coin, because if market-wide moves reach a coin late, their recent returns carry information about its next return.

Each coin gets the same five questions, and each answer decides how the next experiment runs.

1. Is the coin's forward return predictable out of sample at 20 minutes, 1 hour, 3 hours or 6 hours?
2. Does information from BTC, ETH and SOL add to the coin's own history?
3. Does a nonlinear model beat a linear model and one-feature rules?
4. How fast does a model decay without retraining?
5. Does the selected configuration earn more than its fees and funding?

Three more questions compare the coins.

1. Does the cost of trading depend on volatility?
   Fees are fixed in basis points while the edge of a trade grows with volatility, so a calm coin needs a stronger signal to pay the same fee.
2. Does BTC information help more for coins that follow BTC more closely?
3. Does thin trading create predictability that cannot be traded?

The set also contains two close pairs.
DOGE and AAVE moved together with a daily correlation of 0.84 from August to November 2025, and UNI and AAVE are both DeFi tokens.
A result that holds on one coin of a close pair but not on the other is more likely noise than structure.

A second phase, run after the first, adds three questions.

1. Is volatility predictable, and does LightGBM beat a three-input linear model?
2. Does a volatility forecast make the direction strategy pay, by trading only when the expected edge covers the fee and by sizing positions to predicted volatility?
3. Does one model trained on all five coins beat five models trained on one coin each?

## 2. Related work

Return predictors have a poor record out of sample.
Welch and Goyal (2008) showed that most predictors of the equity premium fail out of sample and would not have helped an investor in real time.
Campbell and Thompson (2008) showed that out-of-sample R² is small even for useful predictors and can still matter economically.
Gu, Kelly and Xiu (2020) found that trees and neural networks beat linear models out of sample.
They measured R² against a zero forecast and refit on an expanding sample, which is the template for the R² used here.

The walk-forward design follows the forecasting literature.
Tashman (2000) reviews rolling-origin evaluation and favours many test periods with re-estimation.
Cerqueira, Torgo and Mozetič (2020) find that repeated out-of-sample tests estimate performance best for non-stationary series.
López de Prado (2018) shows how overlapping labels leak across a train and test boundary and introduces purging.

The statistics are standard.
The information coefficient is from Grinold and Kahn (2000), the sampling error of the Sharpe ratio from Lo (2002), and the stationary bootstrap for dependent data from Politis and Romano (1994).
Harvey, Liu and Zhu (2016) argue that a new factor should clear a t-statistic of 3 now that so many have been tried, which is where the bar of 3 below comes from.

For cryptocurrencies, Liu and Tsyvinski (2021) find strong time-series momentum.
Liu, Tsyvinski and Wu (2022) find a market, size and momentum factor structure.
At intraday horizons, Shen, Urquhart and Wang (2022) find that Bitcoin's first half-hour return predicts its last half-hour return.
Wen, Bouri, Xu and Zhao (2022) find both intraday momentum and intraday reversal.
Jaquart, Dann and Weinhardt (2021) predict 1- to 60-minute Bitcoin direction with 51 to 56 percent accuracy, but their strategies lose money at 30 bps per round trip.
Fischer, Krauss and Deinert (2019) find a 120-minute cross-sectional strategy that survives costs but has little capacity.

Across coins, Guo, Sang, Tu and Wang (2024) find that lagged returns of other coins predict a coin's return on Binance.
Jia, Wu, Yan and Liu (2023) find that large coins predict other coins with a negative sign intraday.
Kurihara and Matsumoto (2026) find on 1-minute Binance data that Bitcoin leads altcoins, but that large coins absorb its moves within about a minute.
That leaves little room for a 5-minute lead from BTC to coins as large as these five, and the audit checks it directly.
Makarov and Schoar (2020) document large price gaps across crypto exchanges, which is one reason this study stays on one venue.

The second phase builds on the volatility literature.
Andersen, Bollerslev, Diebold and Labys (2003) showed how to measure and forecast volatility from intraday returns, which they call realised volatility.
Corsi (2009) proposed the HAR model, a regression of future realised volatility on its daily, weekly and monthly averages, which captures long memory with three inputs.
Moreira and Muir (2017) showed that scaling exposure down when volatility is high raises the Sharpe ratio of many factor strategies.

The momentum rule follows the time-series momentum of Moskowitz, Ooi and Pedersen (2012), and the 1/H tranches follow the overlapping portfolios of Jegadeesh and Titman (1993).
Ridge regression is from Hoerl and Kennard (1970) and LightGBM from Ke et al. (2017).

## 3. Data

### 3.1 Source

Bars are the monthly 5-minute kline archives of Binance USD-M perpetual futures, published at data.binance.vision.
Each bar carries open, high, low and close prices, base and quote volume, trade count, and the base and quote volume bought by takers.
Funding comes from the monthly funding rate archive of the same site.
Every downloaded zip is checked against its published SHA-256 checksum.

### 3.2 Sample

The study uses eight pairs: the five targets HYPEUSDT, TRXUSDT, DOGEUSDT, UNIUSDT and AAVEUSDT, and the context pairs BTCUSDT, ETHUSDT and SOLUSDT.
HYPEUSDT has the shortest history, with a first bar at 2025-05-30 10:30 UTC, and every pair uses the same calendar from that bar.
The other seven pairs have years of history, but a longer history for some coins would make the comparison unfair.
The sample ends at 2026-08-31 23:55 UTC.
A complete grid holds 132,066 bars per pair, which is 450 bars in May 2025 and 288 bars on each of the following 457 days.
The longest feature window is 576 bars, so the first complete feature row is 2025-06-01 10:30 UTC.

The five targets differ in liquidity, volatility and dependence on BTC.
The table describes the first training window, June to November 2025.

| Coin | Type | Funding | Median value traded per day | Daily volatility | Correlation with BTC | 5-minute volatility | Lag-1 autocorrelation | Drop on 2025-10-10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HYPE | perpetuals exchange token | 4 h | 372M | 4.90% | 0.65 | 33.5 bps | -0.028 | -51.5% |
| TRX | payments chain | 8 h | 113M | 1.74% | 0.51 | 10.1 bps | 0.034 | -10.3% |
| DOGE | meme coin | 8 h | 1,366M | 4.61% | 0.76 | 26.7 bps | -0.008 | -64.2% |
| UNI | DeFi exchange token | 8 h | 206M | 6.28% | 0.62 | 31.8 bps | -0.020 | -69.7% |
| AAVE | DeFi lending token | 8 h | 228M | 4.65% | 0.76 | 26.6 bps | -0.014 | -69.2% |

Value traded is in USDT.
Daily volatility and correlation come from daily returns.
5-minute volatility and the lag-1 autocorrelation of 5-minute returns leave out 2025-10-10 and 2025-10-11.
The drop runs from the high between 20:00 and 21:00 UTC on 2025-10-10 to the low of that evening, which every coin reached at 21:15 or 21:20 UTC.

Funding settles every 4 hours for HYPEUSDT and every 8 hours for the other four.
One HYPEUSDT settlement, 2026-06-24 04:00 UTC, is missing from both the archive and the API.

The sample holds two opposite regimes.
HYPE fell 66 percent from its September 2025 high to its January 2026 low and then rose to a new high in August 2026.
BTC fell 54 percent from its October 2025 high to its July 2026 low.
The liquidation cascade of 2025-10-10 sits in the first training window.

### 3.3 Alignment and audit

All eight pairs sit on one complete 5-minute UTC grid.
A missing bar stays missing and is never filled.
Binance writes a trading halt as a flat bar with zero trades rather than as a missing row.
Such a bar is set to missing, because nothing traded and its zero return is not an observation.
It happened on 2025-08-29 from 06:20 to 06:30 UTC on all eight pairs, and the audit counts any other case.
Every 576-bar window that touches the halt is undefined, so it costs about two days of training rows.
A rolling window over a missing bar is undefined.
A return between two bars is defined when both ends exist.

A ticker can also change hands.
I ran into this with PUMPUSDT while choosing the coins.
Another token traded under that ticker until 2025-06-30, the archive then holds nine days of flat zero-trade bars, and after a gap of seven and a half hours a new token starts on 2025-07-10 at a price about nine times lower.
Joined naively, the two make a fake drop of 89 percent.
So the audit flags every close that moves by a ratio above 1.3 or below 1/1.3 across a gap or a run of zero-trade bars.

One extreme bar can also dominate a statistic.
With 2025-10-10 and 2025-10-11 included, AAVE's lag-1 autocorrelation of 5-minute returns from June to November 2025 is -0.225.
Without those two days it is -0.014, and a rank version that no single bar can dominate gives -0.017.
Extreme bars are reported and kept, and a bar is excluded only if the data itself is broken.
The bounded label and the rank IC below limit what one bar can do to a result.

## 4. Method

### 4.1 Features

The features are the 111 columns of an earlier TypeScript version of this project, with formulas in appendix A.
They are computed once for each target, with that coin in the role the schema calls `x`.
Every feature at bar $t$ uses bars up to and including $t$.
A test enforces this by overwriting every bar after $t$ and checking that no feature at or before $t$ moves.

| Group | Columns per asset | Content | Model inputs per asset |
| --- | ---: | --- | --- |
| Price and trend | 13 | 1-bar log return, trailing returns over 4 to 72 bars, rolling mean returns, 576-bar volatility, candle range, body and wicks | 10, without the rolling mean returns |
| Activity | 6 | log of volume, quote volume and trade count, and their 576-bar z-scores | the 3 z-scores |
| Flow | 6 | share of volume bought by takers, by base and by quote volume, and their 12-bar and 36-bar means | the 3 base-volume columns |
| Cross-asset, target only | 11 in total | the target's trailing return minus BTC's, ETH's and SOL's, 72-bar return correlation, 288-bar beta | all 11 |

Three kinds of column are kept in the table but are not model inputs.
The log levels of volume, quote volume and trade count drift over months, and a model given levels can learn which month it is in.
The rolling mean of the last $w$ log returns is the trailing $w$-bar return divided by $w$.
The quote-volume taker share is almost a copy of the base-volume share, because the price barely moves inside a bar.
A copy adds nothing to a tree, and in ridge it halves the penalty on its direction.
The differences such as `x_minus_btc_trail_ret_12` stay, because a tree cannot form the difference of two inputs by itself.
That leaves 75 inputs, 16 per asset and 11 cross-asset, with the same names for every target.
The 576-bar volatility stays an input.
It is built from returns rather than from traded amounts, and the centred signal of section 4.5 keeps a month-level bet from scoring through its level alone.

### 4.2 Target

Let $c_t$ be the target's close at bar $t$ and $H$ the horizon in bars.
The forward log return is $r_{t,H} = \ln(c_{t+H} / c_t)$.
Its scale is $\sigma_{t,H} = \hat\sigma_t \sqrt{H}$, where $\hat\sigma_t$ is the population standard deviation of the last 576 one-bar log returns.
The model target is $y_{t,H} = \tanh(r_{t,H} / \sigma_{t,H})$.
The scale uses past bars only, so it is known when the prediction is made.
The tanh bounds the label and limits the weight of crash bars in training.
Costs are not part of the label.
They are charged in the backtest, where both directions pay them.

### 4.3 Walk-forward protocol

A fold is one calendar month in UTC, and every coin uses the same folds.
One rule decides which rows a step may read.
A row is used only if its label is known when the step's data ends.

- Training for test month $M$ ends when $M$ starts, so the last $H$ rows before $M$ are purged.
- The inner model's training ends when the validation month starts, so the last $H$ rows before that month are purged too.
- Validation scoring ends when the test month starts, so the last $H$ rows of the validation month are not scored.
- Research scoring ends on 2026-05-31, so the last $H$ rows of May 2026 are not scored.

By default the training window expands from 2025-06-01 10:30 UTC.
The last month of each training window is the inner validation month.
A model fitted on the earlier rows is scored there, and the score picks the ridge penalty and the LightGBM tree count.
The model is then refit on the inner training rows and the validation month with those settings and predicts the test month.
The $H$ rows just before the validation month stay out of the refit, because their labels were not known when the inner model trained, which is slightly conservative.
The refit model's predictions on the validation month set the strategy thresholds.
Each coin's models train on that coin's rows only.

| Period | Test month | Training data | Inner validation |
| --- | --- | --- | --- |
| Research | 2025-12 | 2025-06 to 2025-11 | 2025-11 |
| Research | 2026-01 | 2025-06 to 2025-12 | 2025-12 |
| Research | 2026-02 | 2025-06 to 2026-01 | 2026-01 |
| Research | 2026-03 | 2025-06 to 2026-02 | 2026-02 |
| Research | 2026-04 | 2025-06 to 2026-03 | 2026-03 |
| Research | 2026-05 | 2025-06 to 2026-04 | 2026-04 |
| Forward | 2026-06 | 2025-06 to 2026-05 | 2026-05 |
| Forward | 2026-07 | 2025-06 to 2026-06 | 2026-06 |
| Forward | 2026-08 | 2025-06 to 2026-07 | 2026-07 |

Every choice of horizon, model, features, retraining policy and threshold is made on the six research months, once for all five coins.
Every experiment before the forward run reads rows up to 2026-05-31 only.
The forward months are opened once, with the configuration frozen at the end of May.

### 4.4 Models

`momentum` is a one-input least-squares fit $\hat y = a + b x$ of the target on the coin's own trailing return over the horizon.
The sign of $b$ decides whether the rule follows the trend or fades it.

`best_feature` is the same fit on the single input with the highest absolute Spearman correlation to the target on the training window.
Ties go to the earlier input in the input list.

`ridge` standardises each input with training means and standard deviations, clips it to $[-5, 5]$, and fits an L2-penalised linear regression.
With an R² near 0.001, a useful penalty is near the number of inputs divided by R², about $10^5$.
The penalty is chosen from $10^2, 10^3, \dots, 10^7$ by mean squared error on the inner validation month, and the choice is logged per coin and fold.

`lightgbm` is a gradient-boosted tree regressor with squared loss and fixed settings.
The learning rate is 0.03 with 15 leaves and an L2 penalty of 10.
Each tree samples 70 percent of the rows and 70 percent of the inputs.
Each leaf holds at least $100H$ rows, which is about 100 independent labels at every horizon.
With 70 percent row sampling a split then needs about $286H$ training rows.
The two-month inner model of a three-month window cannot split at 6 hours.
When the inner model gives one constant prediction, the month gets no model, its days score zero and it is not traded.
The tree count, at most 2,000, is chosen by early stopping on the inner validation month with a patience of 100 trees.
Training is deterministic with a fixed seed.

All four models predict on the scale of $y$, so their R² and their thresholds compare directly.

### 4.5 Evaluation

Each coin and horizon has one fixed set of scored rows.
A row is scored when all 75 inputs and its label are defined and the label is known by the end of the period.
Every model and every feature set is scored on that same set.

Every month has its own model with its own level and scale, so a prediction is first read on its month's validation scale, $s_t = (\hat y_t - q_{50}) / (q_{90} - q_{10})$, where the quantiles are those of the refit model's predictions on the validation month.
Ranking raw predictions across months would let the month itself carry rank, because each month's model has its own level and spread.
The centred signal is also exactly what the median rule of section 4.6 trades.
The information coefficient of a coin is the Spearman correlation between $s_t$ and $r_{t,H}$ over all scored rows of a period.
Let $u_t$ and $v_t$ be the ranks of $s_t$ and of $r_{t,H}$ over those rows, each standardised to mean 0 and standard deviation 1.
The daily value $c_d$ is the mean of $u_t v_t$ over the rows of day $d$.
The IC is the mean of the daily values, which equals the Spearman correlation of the period when every day is equally full, and the IC of a month is the mean of its daily values.
The t-statistic is the mean of $c_d$ divided by its standard error from a stationary bootstrap over days, with a mean block length of 5 days and 10,000 draws.
Labels overlap within a day, and the bootstrap blocks also absorb the correlation between neighbouring days.

The pooled daily value of a configuration is the mean of the five coins' daily values on each day.
The pooled bootstrap resamples whole days, so the five coins of a day stay together and their correlation is kept.
Pooled evidence decides every choice that applies to all five coins.

The obvious alternative, a Spearman correlation inside each day, is biased.
Inside a day of $n = 288$ bars, the trailing and the forward $H$-bar returns share increments with the day's mean.
Removing that mean gives an expected correlation of about $-H/n$ for any prediction built from recent returns, even on a random walk.
At 6 hours that is $-0.25$, and a mean-reversion rule would show a t-statistic near 12 over 182 days with no signal at all.
Ranking over the whole period shrinks the bias to about $-H/N$, with $N$ between 26,000 and 52,000 rows.
A test runs both versions on simulated random walks.

Besides the IC the study reports the following.

- Hit rate is the share of bars where the side taken by the median rule of section 4.6 matches the sign of $r_{t,H}$, leaving out bars with no move and bars where the rule is flat.
- Out-of-sample R² is $1 - \sum (y - \hat y)^2 / \sum y^2$ on the model target, which compares the model with a forecast of zero.
- The decile table is the mean $r_{t,H}$ in each decile of the predictions within each test month.
- Two models are compared by the daily difference of their IC values, with the same bootstrap.
- Agreement between two models, or between the signals of two coins, is the Spearman correlation of their centred signals inside each test month, averaged over months.

### 4.6 Strategy and costs

A prediction becomes a signal $s_t \in \{-1, 0, +1\}$ through two thresholds.
The thresholds are quantiles of the refit model's predictions on the inner validation month.
Quantiles of predictions read no labels, so they leak nothing.
Three rules are run.
The median rule is long above the median and short below it.
A prediction exactly on a threshold gives a flat signal, because a model with few distinct values would otherwise send every tie one way.
The 30 percent rule trades the outer 30 percent on each side, and the 10 percent rule the outer 10 percent.
A bar whose prediction is undefined gives a flat signal.

Each bar opens a tranche of size $1/H$ in the direction of its signal and closes it $H$ bars later.
The position is the mean of the last $H$ signals, $p_t = \frac{1}{H}\sum_{k=0}^{H-1} s_{t-k-\delta}$.
The delay $\delta$ is 0 in the main run, where the position is taken at the close of the signal bar, and 1 in a check.
Tranches in opposite directions net against each other, as orders would.

Profit on bar $t$ is $\pi_t = p_{t-1}(c_t / c_{t-1} - 1) - f\,|p_t - p_{t-1}| - p_{t-1}\phi_t$.
Here $f$ is the fee per side and $\phi_t$ is the funding rate settled at the open of bar $t$, zero on bars without a settlement.
A long pays a positive rate.
Archive funding times carry a few milliseconds of jitter and are floored to 5 minutes, and only settlements that exist are charged.
The fee per side is 5 bps for taker orders and 2 bps for maker orders, the regular Binance USD-M rates for the whole sample.
Paying fees in BNB would cut both by 10 percent.

Bar profits are summed into daily profits for each coin.
The portfolio is the equal-weight mean of the five coins' daily profits.
The Sharpe ratio is $\sqrt{365}$ times the mean daily profit over its standard deviation, with a 90 percent interval from the stationary bootstrap over days.
The breakeven fee is gross profit minus funding, divided by turnover, which is the fee per side at which net profit is zero.
Each result also reports maximum drawdown, turnover per day, time in the market, the long share of the book, the beta of daily profit to the coin's daily return, and buy-and-hold over the same days.
At the end of each period the open position is closed and pays its exit fee.

### 4.7 Power and cost arithmetic

Two calculations set expectations before any model is trained.

The first is the smallest IC the protocol can detect.
With overlapping labels a day holds about $288 / H$ independent observations, so the daily value has a standard deviation near $\sqrt{H / 288}$.
The research period has 182 days and the forward period has 92.
The table gives the IC at which the expected t-statistic equals the threshold, which means a 50 percent chance of detection.
It is the same for every coin.
The rule assumes a prediction that barely changes within $H$ bars, and it is conservative for one that changes as fast as the label.

| Horizon | Research, $t = 3$ | Research, $t = 3.2$ | Forward, $t = 2$ | Forward, $t = 3$ |
| --- | ---: | ---: | ---: | ---: |
| 20 minutes | 0.026 | 0.028 | 0.025 | 0.037 |
| 1 hour | 0.045 | 0.048 | 0.043 | 0.064 |
| 3 hours | 0.079 | 0.084 | 0.074 | 0.111 |
| 6 hours | 0.111 | 0.119 | 0.104 | 0.156 |

Pooling the five coins shrinks the standard error by $\sqrt{(1 + 4\rho)/5}$ when the coins share the same IC, where $\rho$ is the correlation between the coins' daily values.
That factor is 0.66 at $\rho = 0.3$ and 0.77 at $\rho = 0.5$, so the pooled research threshold at 1 hour falls from 0.045 to between 0.030 and 0.035.
The horizon sweep measures $\rho$.

The second calculation is the IC needed to pay fees.
If prediction and return are jointly normal, a trade in the top decile of predictions earns on average about $1.755 \cdot IC \cdot \sigma_H$, where 1.755 is the mean of a standard normal above its 90th percentile.
A taker round trip costs 10 bps and a maker round trip 4 bps.
$\sigma_H$ is the realised standard deviation of $H$-bar log returns from June to November 2025, without 2025-10-10 and 2025-10-11, so that one crash does not set the typical scale.
It is not the label scale.

| Coin | $\sigma_H$ at 20 min, 1 h, 3 h, 6 h | Taker IC at 20 min | 1 h | 3 h | 6 h | Maker IC at 1 h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HYPE | 65, 110, 184, 253 bps | 0.088 | 0.052 | 0.031 | 0.023 | 0.021 |
| TRX | 21, 36, 63, 87 bps | 0.275 | 0.158 | 0.091 | 0.066 | 0.063 |
| DOGE | 52, 89, 153, 218 bps | 0.109 | 0.064 | 0.037 | 0.026 | 0.025 |
| UNI | 62, 107, 192, 279 bps | 0.092 | 0.053 | 0.030 | 0.020 | 0.021 |
| AAVE | 52, 90, 157, 222 bps | 0.109 | 0.064 | 0.036 | 0.026 | 0.025 |

At 20 minutes the research threshold at $t = 3$ is 0.026 while the taker IC needed is 0.088 to 0.275, and at 6 hours the threshold is 0.111 while the taker IC needed is 0.020 to 0.066.
Against the research threshold at $t = 3$, the curves cross at about 70 minutes for HYPE and UNI, 85 minutes for DOGE and AAVE, and 3.5 hours for TRX.
TRX moves about a third as much as the other coins, so at 1 hour it needs about three times their IC.
Both tables are rough.
Tranches in the same direction net out, so real turnover is lower than one round trip per tranche.
Returns are fat-tailed, so the normal approximation is loose.
The horizon sweep and the economics experiment replace both tables with measured values.

The Sharpe ratio is noisy too.
Its standard error is about $\sqrt{365 / D}$ when annualised over $D$ days.
A 90 percent interval excludes zero only above a Sharpe ratio of about 2.3 over the research period and 3.3 over the forward period.

### 4.8 Multiple testing

The horizon sweep scores 80 configurations, five coins times four horizons times four model families.
Choices that apply to all coins use pooled evidence.
Twenty-two pooled configurations feed a choice: 16 recipes in the sweep, one alternative feature set, two alternative retraining policies and three threshold rules.
A pooled recipe counts as detected only at $t \ge 3$, which is above the one-sided Bonferroni bound at 5 percent for 22 tests, 2.84.
A claim about one coin faces all 80 configurations and needs $t \ge 3.2$, the bound for 80 tests rounded down from 3.23.
Every configuration that was run is reported, including the ones that lost.

### 4.9 Phase 2

Phase 2 reuses the data, the folds, the label-known rule and the rank IC of phase 1.

#### Volatility target and models

Realised volatility over the next $H$ bars is $RV_{t,H} = \sqrt{\sum_{k=1}^{H} r_{t+k}^2}$, where $r$ is the 1-bar log return.
The target is $\ln RV_{t,H}$, which is closer to normal than the level.
A window with a missing bar has no label.
The HAR model regresses the target on the log of trailing realised volatility over the last 12, 72 and 288 bars, which is 1 hour, 6 hours and 1 day.
It is fitted by least squares on the training window.
Against it, LightGBM uses the 75 inputs plus the hour of day, with the phase 1 settings.
The hour of day is the one input LightGBM gets beyond the schema, because volatility follows the trading day, and HAR's three inputs are built outside the schema as well.
Both models are scored with the rank IC of section 4.5 and with out-of-sample R² against the training-window mean of the target.

#### Volatility-aware strategy

The phase 1 frozen strategy runs in two more versions.
The gated version trades a bar only when the expected edge of its signal covers the round trip.
The expected edge is $k_q \cdot IC \cdot \hat\sigma_{t,H}$, where $\hat\sigma_{t,H}$ is the forecast of $RV_{t,H}$ and $IC$ is the frozen model's IC on the inner validation month.
$k_q$ is the mean of a standard normal beyond the rule's threshold, 0.80 for the median rule, 1.16 for the 30 percent rule and 1.755 for the 10 percent rule.
A bar is traded when the expected edge is at least twice the fee per side.
A month whose validation IC is not positive is therefore not traded.
The sized version also scales each tranche by the validation-month median of $\hat\sigma$ divided by $\hat\sigma_{t,H}$, capped at 2.
Calm periods then carry larger positions and wild periods smaller ones, as in Moreira and Muir (2017).

#### Pooled model

The pooled model is the frozen family $M^*$ at $H^*$, trained once per fold on the stacked rows of all five coins without a coin identifier.
If $M^*$ is a rule, ridge is used.
Before stacking, each input is standardised per coin with that coin's training rows, so that TRX's small moves are not read as the quiet periods of a volatile coin.
It is scored on each coin's own scored rows.

Phase 2 adds four configurations that feed a choice: the second volatility model, the two strategy versions and the pooled model.
That brings the pooled count to 26, whose one-sided Bonferroni bound at 5 percent is 2.89, still under the threshold of 3.

## 5. Experiments

Results go to `results/<experiment>/` as CSV tables and PNG figures.

### E0. Audit

The audit reads research rows of all eight pairs.
It counts expected, present, missing and zero-trade bars per pair and month, and it flags seams.
It lists bars whose return exceeds ten trailing standard deviations, so that each one can be matched to a market event.
It shows how the coins move together, as a correlation matrix of 5-minute returns and as each coin's rolling 30-day correlation with BTC.
It measures the correlation between BTC's and each coin's 1-bar log returns at lags from minus 6 to plus 6 bars on the first training window, without the two crash days, which dominate lagged statistics as the AAVE example shows.
Kurihara and Matsumoto (2026) found that large coins absorb a Bitcoin move within about a minute, so every lag other than 0 should be close to zero.
It reports the lag-1 autocorrelation of 5-minute returns per coin, with and without the October crash, and each coin's volatility over time.
The audit excludes bars only when the data is broken.

### E1. Horizon and model sweep

Every coin is crossed with every horizon and every model family, 80 configurations, with the expanding window over the six research months.
Each configuration reports its IC with t-statistic, hit rate, out-of-sample R², decile table and IC by month.
Each also reports gross profit, turnover and breakeven fee under the median rule, so the economics of every configuration are visible.
The results are shown as a heatmap of IC by coin and horizon for each model family, and as IC by month for each coin.
The sweep also shows whether the models agree, as the correlation between ridge and LightGBM predictions, and whether the coins share one signal, as the correlation of predictions across coins.
It measures $\rho$, the correlation between the coins' daily values.
The expected result is an IC under 0.05 at short horizons, with LightGBM no better than ridge.

The frozen recipe, a horizon $H^*$ and a model family $M^*$, is chosen once for all coins.
For each of the 16 recipes, the pooled daily value is the mean of the five coins' daily values.
A recipe counts as detected when its pooled t-statistic is at least 3.
Among detected recipes, the frozen one has the highest pooled breakeven fee under the median rule, the five coins' gross profit minus funding divided by their turnover.
Selecting on the t-statistic alone would favour the shortest horizon, where section 4.7 puts the taker IC needed at 0.088 to 0.275.
If nothing is detected, the study says so and continues with the recipe of the highest pooled t-statistic as a descriptive case.
Each coin's own configuration with the highest t-statistic is recorded as well, for the forward run.

### E2. Feature sources

At $H^*$, the model $M^*$ runs for every coin on all 75 inputs and on the coin's own 16 inputs.
If $M^*$ is a rule, ridge runs instead.
All inputs stay unless the coins' own inputs win with a pooled paired t-statistic of at least 2.
For each coin, the gain from the BTC, ETH and SOL inputs is plotted against the coin's correlation with BTC.
With only five points this can describe the coins but cannot test anything.

To describe what the model uses, permutation importance is computed for every coin on the research test months for four groups: the coin itself, BTC, ETH and SOL.
Each cross-asset column joins the group of the asset it compares the coin with.
Without that, a group could be rebuilt from the others, because `x_minus_btc_trail_ret_12` is exactly `x_trail_ret_12` minus `btc_trail_ret_12`.

### E3. Retraining

At $H^*$, with $M^*$ and the feature set from E2, three policies run for every coin over the research months.
They are the expanding window refit monthly, a model trained once on June to November 2025 and never refit, and a rolling 3-month window refit monthly.
The never-refit model is the same June to November 2025 model in every research and forward month.
The output is the IC by month for each policy and coin, and the IC of the never-refit model against months since training.
If the expanding and the never-refit models drop in the same month, the market changed.
If only the old model drops, the model aged.
The expanding window stays unless an alternative beats it with a pooled paired t-statistic of at least 2.

### E4. Economics

At $H^*$, with $M^*$, the E2 feature set and the E3 policy, each threshold rule runs for every coin at taker and at maker fees, with no delay and with a one-bar delay.
Funding is always charged.
The output shows each coin's breakeven fee against the 5 bps taker line, the needed IC against each coin's volatility, and each coin's cumulative net profit against buy-and-hold.
It also shows the correlation of the five strategies' daily profits and the equal-weight portfolio, which says whether combining the coins helps.
A coin whose one-bar delay erases its profit is flagged, because a signal that lives in a single bar is the mark of bid-ask bounce.
These numbers come from the same six months that chose everything else, so they are selection-period numbers and read high.
The frozen rule is the one with the highest net Sharpe ratio of the equal-weight portfolio at taker fees and no delay.

### E5. Forward run

The frozen configuration is retrained under the frozen policy at the start of June, July and August 2026 and runs on every coin.
For context, the other model families at $H^*$ are scored on the same months, and no choice is made from them.
Each coin's own best research configuration from E1 is scored too.
Its drop from research to forward measures how much picking the best overstates a result.
The forward audit and a split of each coin's IC by terciles of its trailing 576-bar volatility are reported with it.

The result for a coin counts as consistent when its forward IC has the research sign and differs from the research IC by less than $1.96\sqrt{se_r^2 + se_f^2}$.
The same rule applies to the pooled IC.
The research IC is the best of 16 recipes, so some shrinkage in the forward months is expected.
A forest plot shows research against forward IC for every coin with its interval.
The economic verdict is the forward net profit of the equal-weight portfolio at taker fees with its 90 percent interval, with every coin reported on its own.
With 92 days the forward threshold at $t = 2$ is 0.025 at 20 minutes and 0.104 at 6 hours, so a small IC cannot be established on the forward months alone.

### Phase 2

The three phase 2 experiments run after phase 1 on the same data and folds.
Their choices are made on the research months, and their forward months are opened once, as in phase 1.

### E6. Volatility forecasts

For every coin and horizon, HAR and LightGBM forecast $\ln RV_{t,H}$ over the research months.
The output is each model's IC and R² per coin and horizon, the IC by month, and the gain of LightGBM over HAR against each coin's volatility.
The expected result is an IC far above any direction IC for both models, with a small gain for LightGBM from the hour of day.
HAR stays the volatility model unless LightGBM beats it with a pooled paired t-statistic of at least 2.
Both models are scored on the forward months.

### E7. Volatility-aware strategy

At the frozen recipe and threshold rule, the base, gated and sized versions run for every coin and for the portfolio, at taker and maker fees with funding, using the volatility model from E6.
The output is the same as in E4, plus the share of bars the gate lets through and the average size of the sized version.
The version with the highest research net Sharpe ratio of the portfolio at taker fees is reported as the phase 2 strategy, and all three versions are scored on the forward months.
If the frozen recipe was not detected in E1, E7 still runs, and its results show whether volatility timing alone can rescue a weak signal.

### E8. Pooled model

At $H^*$, the pooled model and the five per-coin models run over the research months.
The output is the paired IC difference per coin and pooled, with the per-coin difference plotted against each coin's volatility.
The per-coin models stay unless the pooled model wins with a pooled paired t-statistic of at least 2.
Both are scored on the forward months for context.

## References

Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625. https://doi.org/10.1111/1468-0262.00418

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies*, 21(4), 1509-1531. https://doi.org/10.1093/rfs/hhm055

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109(11), 1997-2028. https://doi.org/10.1007/s10994-020-05910-7

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196. https://doi.org/10.1093/jjfinec/nbp001

Fischer, T. G., Krauss, C., & Deinert, A. (2019). Statistical arbitrage in cryptocurrency markets. *Journal of Risk and Financial Management*, 12(1), 31. https://doi.org/10.3390/jrfm12010031

Grinold, R. C., & Kahn, R. N. (2000). *Active portfolio management* (2nd ed.). McGraw-Hill.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhaa009

Guo, L., Sang, B., Tu, J., & Wang, Y. (2024). Cross-cryptocurrency return predictability. *Journal of Economic Dynamics and Control*, 163, 104863. https://doi.org/10.1016/j.jedc.2024.104863

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68. https://doi.org/10.1093/rfs/hhv059

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634

Jaquart, P., Dann, D., & Weinhardt, C. (2021). Short-term bitcoin market prediction via machine learning. *Journal of Finance and Data Science*, 7, 45-66. https://doi.org/10.1016/j.jfds.2021.03.001

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

Jia, Y., Wu, Y., Yan, S., & Liu, Y. (2023). A seesaw effect in the cryptocurrency market: Understanding the return cross predictability of cryptocurrencies. *Journal of Empirical Finance*, 74, 101428. https://doi.org/10.1016/j.jempfin.2023.101428

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.

Kurihara, T., & Matsumoto, T. (2026). Price transmission from Bitcoin to altcoins: High-frequency evidence and implications for trading strategy. *Asia-Pacific Financial Markets*. https://doi.org/10.1007/s10690-026-09589-z

Liu, Y., & Tsyvinski, A. (2021). Risks and returns of cryptocurrency. *Review of Financial Studies*, 34(6), 2689-2727. https://doi.org/10.1093/rfs/hhaa113

Liu, Y., Tsyvinski, A., & Wu, X. (2022). Common risk factors in cryptocurrency. *Journal of Finance*, 77(2), 1133-1177. https://doi.org/10.1111/jofi.13119

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36-52. https://doi.org/10.2469/faj.v58.n4.2453

López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.

Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. *Journal of Financial Economics*, 135(2), 293-319. https://doi.org/10.1016/j.jfineco.2019.07.001

Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644. https://doi.org/10.1111/jofi.12513

Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250. https://doi.org/10.1016/j.jfineco.2011.11.003

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313. https://doi.org/10.1080/01621459.1994.10476870

Shen, D., Urquhart, A., & Wang, P. (2022). Bitcoin intraday time series momentum. *Financial Review*, 57(2), 319-344. https://doi.org/10.1111/fire.12290

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437-450. https://doi.org/10.1016/S0169-2070(00)00065-0

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455-1508. https://doi.org/10.1093/rfs/hhm014

Wen, Z., Bouri, E., Xu, Y., & Zhao, Y. (2022). Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both. *North American Journal of Economics and Finance*, 62, 101733. https://doi.org/10.1016/j.najef.2022.101733

## Appendix A. Feature formulas

`EPS` is $10^{-12}$.
Every window ends at and includes bar $t$, and variance is the population variance over the window.
`lr` is the 1-bar log return, and `x` is the target coin.

| Feature | Formula |
| --- | --- |
| `a_logret_1` | `ln(c[t] / (c[t-1] + EPS))` |
| `a_trail_ret_k`, k in 4, 12, 36, 72 | `ln(c[t] / (c[t-k] + EPS))` |
| `a_roll_mean_logret_w`, w in 12, 36, 72 | mean of `lr` over the last w bars, equal to `a_trail_ret_w / w` |
| `a_roll_vol_logret_576` | population standard deviation of `lr` over the last 576 bars |
| `a_candle_range_pct` | `(h - l) / (c + EPS)` |
| `a_candle_body_pct` | `(c - o) / (c + EPS)` |
| `a_upper_wick_pct` | `(h - max(o, c)) / (c + EPS)` |
| `a_lower_wick_pct` | `(min(o, c) - l) / (c + EPS)` |
| `a_volume_log`, `a_quote_volume_log`, `a_count_log` | `ln(1 + value)` |
| `a_volume_z_576`, `a_quote_volume_z_576`, `a_count_z_576` | `(value - mean576) / (std576 + EPS)` |
| `a_taker_buy_ratio_vol` | `taker_buy_volume / (volume + EPS)` |
| `a_taker_buy_ratio_qv` | `taker_buy_quote_volume / (quote_volume + EPS)` |
| `a_taker_buy_ratio_*_ma12`, `_ma36` | rolling means of the two ratios |
| `x_minus_btc_trail_ret_k`, k in 12, 36, 72 | `x_trail_ret_k - btc_trail_ret_k` |
| `x_minus_eth_trail_ret_12`, `x_minus_sol_trail_ret_12` | the same against ETH and SOL |
| `corr_x_a_logret_72`, a in btc, eth, sol | `cov72(x_lr, a_lr) / (std72(x_lr) * std72(a_lr) + EPS)` |
| `beta_x_a_288`, a in btc, eth, sol | `cov288(x_lr, a_lr) / (var288(a_lr) + EPS)` |
