# Token Radar 3 — a multivariate ridge comps model for pre-token protocols

Entry for make.fun bounty token-radar-01 ("Build the best token launch radar").

## The model (radar-3)

The starter's radar-1 scores a protocol as `tvl_usd x 0.25`. Its 0.25 is a guess. radar-2
fitted a log-log regression of market cap on TVL (R² = 0.103, leave-one-out median error
0.571 in log10, about 3.7x) and patched it with raw category medians and hand-tuned
growth/maturity priors. radar-3 replaces the patchwork with a ridge regression of
`log10(mcap)` on two standardised predictors — `log10(tvl)` and `age/1095` — with λ=10,
fitted at run time on the snapshot's own 94 comparable live-token protocols.

- Fit (derived at run time): `log10(mcap) = 4.8501 + 0.3269 x log10(tvl) - 0.4834 x age/1095`, R² = 0.123
- Leave-one-out median error: 0.517 in log10 (about 3.29x) — stated on the dashboard, not hidden
- Uncertainty band (10–90% of out-of-fold residuals): x0.06–x8.07, covering 80.9% of comparables
- Age term capped at ±0.3 in log10 (binds for 61 of 758 protocols); implied values clamped to the observed live market-cap range ($60,672–$2,646,363,290)

Every candidate feature was tested on the same leave-one-out splits before it was allowed
in, and the ones that lost were dropped and the dashboard says so:

- Seven-day change: raises the leave-one-out error at every penalty — dropped, growth multiplier is 1
- Chain count: does not improve the leave-one-out error consistently — dropped
- Category offsets: under an honest nested leave-one-out (offsets from each fold's own
  out-of-fold residuals, shrunk by n/(n+3)) they raise the error — dropped, none applied
- Fees / revenue / volume / holders / liquidity: not fields the snapshot carries, so no model on this snapshot can use them

The fitted age effect is negative: holding TVL fixed, older protocols trade lower, so the
age multiplier rewards youth. The dashboard shows the error by predicted quintile and by
category (Basis Trading ±0.13 and RWA ±0.12 are the most predictable; Dexs ±0.74, Lending
±0.73 and Yield ±0.75 the least).

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
