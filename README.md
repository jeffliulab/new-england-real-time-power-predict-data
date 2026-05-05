# new-england-real-time-power-predict-data

Auto-built rolling-7-day backtest data for the [predict-power HF Space](https://huggingface.co/spaces/jeffliulab/predict-power), refreshed daily by a GitHub Actions cron.

**Do not edit `data/` by hand.** It is overwritten every day at 04:00 UTC by the `refresh.yml` workflow.

## What this repo serves

| File | Purpose |
|---|---|
| `data/backtest_rolling_7d.json` | 7 daily forecasts on the most recent fully-published days (per-zone baseline + Chronos-Bolt-mini + ensemble + ground truth + per-zone & overall MAPE). The HF Space's "Backtest" tab reads this. |
| `data/iso_ne_30d.csv` | 30-day per-zone hourly demand history. The HF Space's Live tab uses this as the base for Chronos's 720-hour context, splicing in the latest day from the live ISO-NE feed. |
| `data/last_built.json` | Metadata: `built_at`, `code_sha` (commit of [`real-time-power-predict`](https://github.com/jeffliulab/real-time-power-predict) used for the build), `data_period`, `summary_mape_pct`. |

The HF Space pulls all three at startup via `https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/...`. No auth, no secrets needed.

## Why a separate repo

1. **HF Space rebuilds**: HF auto-syncs the main code repo's GitHub→Space integration, so committing data to the code repo would trigger a full Space rebuild every day for no reason.
2. **Code repo cleanliness**: The main `real-time-power-predict` repo is the source-of-truth for the paper + code; we don't want it polluted with daily auto-commits.
3. **Reusability**: any third party can `curl` the JSON to inspect current ISO-NE forecasting performance without cloning anything.

## How the cron works

`.github/workflows/refresh.yml` runs every day at 04:00 UTC (after ISO-NE has fully published yesterday's 5-min zonal data). It:

1. Checks out this repo + the source-of-truth [`real-time-power-predict`](https://github.com/jeffliulab/real-time-power-predict)
2. Installs the build dependencies (`herbie-data`, `cfgrib`, `xarray`, `eccodes`, `chronos-forecasting`, `torch`, plus `libeccodes-dev`/`libeccodes-tools` from apt)
3. Runs `scripts/build_rolling_backtest.py --output-dir ../data --parallel 8`
4. If `data/` changed, commits + pushes back to this repo

Public repos have **unrestricted** GitHub Actions usage on the free tier, so this is free indefinitely. A typical run takes ~10–15 minutes.

## Manual trigger

```bash
gh workflow run refresh.yml -R jeffliulab/new-england-real-time-power-predict-data
```

## License

MIT, matching the source-of-truth repo.
