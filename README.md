# trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kuntinku/trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kuntinku/trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy. Entered on make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

The model is refresh-aware: Exophase freezes a game's earned counts between refreshes and dilutes every rate when players-tracked jumps. Per trophy it fits logit(rate) to ln(players_tracked) (elasticity, shrunk toward the pooled value), projects tracked growth, detects a stale launch-week page and falls back to a documented-launch-night dilution prior, and sets a spread from walk + drift-residual + shift uncertainty.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=m88kv07r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
