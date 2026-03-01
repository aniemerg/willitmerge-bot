# Will It Merge? Trading Bot

A trading bot for [WillItMerge](https://willitmerge.com) prediction markets. It uses a
LightGBM model trained on GitHub PR data to predict whether a pull request will merge
within its deadline, then automatically buys and sells positions where the model disagrees
with the market price.

## Quick start

**Requirements:** Python 3.10+, a WillItMerge account, and a GitHub token.

```bash
git clone <this-repo> && cd openclaw-models
pip install -r requirements.txt

cp .env.example .env
# Edit .env: fill in BOT_API_KEY and GITHUB_TOKEN
```

Run in dry-run mode (no real trades, safe to explore):

```bash
python src/bot.py
```

Enable live trading:

```bash
BOT_DRY_RUN=false python src/bot.py
```

The bot wakes every 10 minutes, fetches all open markets, scores each PR using the
pre-trained model, and trades toward a target position wherever it finds edge.

## How it works

Each cycle has three phases:

1. **Fetch** — GET all open markets from the API, fetch your current bankroll and positions
2. **Score** — parallel GitHub + ML scoring of every market (ThreadPoolExecutor)
3. **Trade** — walk results top-down by edge, execute trades to reach target position

**Position sizing:** the bot doesn't blindly add to positions. It computes a target
dollar amount per market based on edge strength and bankroll fraction, then trades only
the delta needed to get there. If the model flips sides, it sells the wrong-side shares
and buys the right side in the same cycle.

## Configuration

Set these in your `.env` (see `.env.example` for all options):

| Variable | Default | Description |
|---|---|---|
| `BOT_API_KEY` | — | WillItMerge API bearer token (required) |
| `GITHUB_TOKEN` | — | GitHub token for live PR data (required) |
| `BOT_DRY_RUN` | `true` | Set `false` for live trading |
| `BOT_LOOP_MINUTES` | `10` | Minutes between cycles |
| `BOT_EDGE_THRESHOLD` | `0.03` | Min probability edge to open/hold a position |
| `BOT_MAX_TRADES_PER_CYCLE` | `5` | Max buys per cycle (live mode only) |
| `BOT_MAX_SPEND_PER_CYCLE` | `50` | Max $ spent on buys per cycle (live mode only) |
| `BOT_MAX_DOLLARS_PER_MARKET` | `25` | Hard dollar cap per position |
| `BOT_MAX_FRACTION_PER_MARKET` | `0.02` | Max bankroll fraction per market |

## The model

Pre-trained model artifacts are included in `models/` — the bot works out of the box
without any training step.

**Performance:** AUC-ROC 0.9042 · calibrated log loss 0.3115 · Brier 0.1019  
Trained on 5,319 fork PRs from the openclaw repository.

Features are extracted in five groups:

| Group | Description |
|---|---|
| A | At-creation signals: title/body stats, code size, branch prefix, labels, author age |
| B | Post-creation signals: CI status, review state, comments, Greptile priority counts |
| C | Time-aware: hours open, deadline fraction, activity recency |
| D | Author history: prior PR count and acceptance rate (temporally correct, no leakage) |
| E | LLM signals: 14 structured quality questions scored by gpt-5-nano |

Greptile inline comments (Group B) are among the strongest predictors: PRs with zero
P0 issues merge at ~14%; PRs with any P0 issues merge at ~4–5%.

---

## Advanced: training your own model

You'll need the dataset (`dataset/prs.jsonl`, ~93 MB) — see `dataset/README.md` for
the schema and collection methodology.

**Train:**

```bash
# With LLM features (requires OPENAI_API_KEY, costs ~$1–2)
python src/train.py

# Without LLM features (faster, slightly lower accuracy)
python src/train.py --no-llm

# Custom deadline window
python src/train.py --deadline 30
```

**Evaluate:**

```bash
python src/evaluate.py   # prints metrics, saves calibration + feature importance plots
```

**Test inference on a single PR:**

```bash
python src/predict.py --pr-number 28378
```

### Source layout

```
src/
├── bot/            trading bot (api, config, loop, scorer, display, trade)
├── ml/             feature engineering library
│   ├── features.py       Groups A/B/C feature extraction
│   ├── author_history.py Group D: per-author temporal history
│   └── llm_features.py   Group E: gpt-5-nano structured extraction + cache
├── bot.py          entry point: python src/bot.py
├── train.py        training pipeline
├── evaluate.py     evaluation + calibration plots
├── predict.py      inference function used by the bot
└── github_live.py  live GitHub GraphQL fetch used by the bot

scripts/
└── discover_questions.py   one-time utility to generate LLM question set
```
