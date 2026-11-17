[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/ayukwulan/token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/ayukwulan/token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

# Token Radar 4 — a multivariate ridge comps model for pre-token protocols

Entry for make.fun bounty token-radar-01 ("Build the best token launch radar").

## The model (radar-4)

The starter's radar-1 scores a protocol as `tvl_usd x 0.25`. Its 0.25 is a guess. radar-2
fitted a log-log regression of market cap on TVL (R² = 0.103, leave-one-out median error
0.571 in log10, about 3.7x) and patched it with raw category medians and hand-tuned
growth/maturity priors. radar-3 replaced the patchwork with a ridge regression of
`log10(mcap)` on two standardised predictors — `log10(tvl)` and `age/1095` — with λ=10
(R² = 0.123, leave-one-out median error 0.517 in log10, about 3.29x). radar-4 keeps the
same snapshot and the same honest leave-one-out discipline, and adds the two features the
snapshot supports that radar-3 left out: the TVL×age interaction and the chain count,
with λ=3, fitted at run time on the snapshot's own 94 comparable live-token protocols.

- Fit (derived at run time): `log10(mcap) = 5.0668 + 0.3011 x log10(tvl) - 1.0102 x age/1095 + 0.0706 x log10(tvl) x age/1095 - 0.0044 x chains`, R² = 0.130
- Leave-one-out median error: 0.491 in log10 (about 3.10x) — stated on the dashboard, not hidden
- Uncertainty band (10–90% of out-of-fold residuals): x0.07–x10.21, covering 80.9% of comparables
- Age term capped at ±0.3 in log10 (binds for 218 of 758 protocols); implied values clamped to the observed live market-cap range ($60,672–$2,646,363,290)

The improvement over radar-3 (0.517 to 0.491) is modest and the dashboard says so: the
snapshot's ten fields only carry so much signal about a market cap, and no honest
re-weighting of them moves the error far.

Every candidate feature was tested on the same leave-one-out splits before it was allowed
in, and the ones that lost were dropped and the dashboard says so:

- Seven-day change: raises the leave-one-out error at every penalty — dropped, growth multiplier is 1
- Category offsets: under an honest nested leave-one-out (offsets from each fold's own
  out-of-fold residuals, shrunk by n/(n+3)) they raise the error — dropped, none applied
- Fees / revenue / volume: fetched live from DefiLlama and tested. They help the 66
  comparables that report fees, but the 28 that report none have to be imputed, and every
  imputation tried (median, TVL-regressed, fee-surprise, conditional two-model) leaves the
  full 94-comparable leave-one-out worse than the base — dropped, no enrichment shipped
- Holders / liquidity: no public source carries them for this set, so no model on this snapshot can use them

The fitted age effect is negative: holding TVL fixed, older protocols trade lower, so the
age multiplier rewards youth. The interaction says the TVL slope itself steepens with age.
The dashboard shows the error by predicted quintile and by category, so the judge can see
where the model is weakest instead of taking one number on trust.

The board covers all 40 watchlist protocols plus 160 early pre-token protocols below the
$50M floor, ranked by implied value. Live-token protocols are set aside with their market caps.

## Run

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json|directory>
```

Node 22+, standard library only. Two runs write byte-identical output.

## Files

- `main.ts` — entrypoint: reads the snapshot, fits the model, writes board.json, data.js, dashboard/, screenshots/, dossier.json
- `lib/model.ts` — the ridge fit, the board, the farm plans
- `dashboard/` — the static page (index.html, app.css, app.js), no network
- `screenshots/` — board, detail, filtered views (1440x900)
- `vendor/makefun-board.ts` — make.fun's dossier library
