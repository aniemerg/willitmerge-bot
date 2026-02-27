# openclaw-models

LightGBM model predicting P(PR merged within deadline) for the openclaw/openclaw GitHub
repository, paired with a Python trading bot that bets on WillItMerge prediction markets.

---

## Model Performance (current)

| Metric | Value |
|---|---|
| AUC-ROC | **0.9042** |
| Calibrated log loss | **0.3115** |
| Brier score | **0.1019** |
| Baseline log loss | 0.5613 |
| Training set | 5,319 PRs |
| Validation set | 1,329 PRs |
| Base rate (val) | 24.9% |

Trained on fork PRs from openclaw/openclaw collected 2026-02-18. Deadline: 14 days.

---

## Repository Structure

```
openclaw-models/
  src/
    features.py          — Feature extraction (Groups A–C + greptile)
    author_history.py    — Group D: per-author prior accept rate (temporal)
    llm_features.py      — Group E: LLM quality scores via gpt-5-nano
    predict.py           — Inference: predict_pr(pr_dict) → float
    train.py             — Training pipeline → models/
    github_live.py       — Fetch live PR from GitHub GraphQL (for bot)
    env.py               — Loads .env at import time
    bot/
      config.py          — BotConfig from BOT_* env vars
      api.py             — WillItMerge REST client (httpx)
      trade.py           — Trade sizing math
      strategy.py        — OpenclawStrategy (ML) + RandomStrategy
      loop.py            — Main trading loop
      logger.py          — Colored terminal output
    bot.py               — Bot entry point
  models/
    lgbm_model.pkl       — Trained LightGBM booster
    calibrator.pkl       — Isotonic regression calibrator
    metadata.json        — Feature names, hyperparams, val metrics
  dataset/
    prs.jsonl            — Raw PR data (not committed)
    README.md            — Dataset schema documentation
  analysis/
    findings.md          — Data analysis and EDA findings
  cache/
    llm_features.json    — Cached LLM scores (persists across runs)
```

---

## Feature Groups

| Group | Description | Features |
|---|---|---|
| A | At-creation | Title/body text stats, code change size, file types, author account age, branch prefix, labels, draft/milestone/assignee flags |
| B | Post-creation | CI state & check counts, review states, comment/participant/reaction counts, review threads, greptile inline comment priority counts |
| C | Time-aware | Hours/days open, fraction of deadline elapsed, hours remaining, last activity age |
| D | Author history | Prior PR count, prior accept count, prior accept rate, is-first-PR flag |
| E | LLM | 14 structured quality scores from gpt-5-nano (body structure, clarity, scope, tests, docs, etc.) |

**Greptile features (Group B):** Greptile-apps posts automated code reviews with priority tags
`[P0]`–`[P3]`. PRs with zero P0 issues merge at 14.3%; PRs with any P0 issues merge at 4–5%.
Features: `greptile_p0_count`, `greptile_p1_count`, `greptile_p2_count`, `greptile_inline_count`.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and GITHUB_TOKEN
```

---

## Training

```bash
python src/train.py
# Options:
#   --deadline 14      Merge deadline in days (default: 14)
#   --no-llm           Skip LLM features (faster, lower AUC)
#   --data dataset/prs.jsonl
```

Outputs `models/lgbm_model.pkl`, `models/calibrator.pkl`, `models/metadata.json`.

LLM scores are cached in `cache/llm_features.json` and reused on subsequent runs.

---

## Running the Bot

```bash
# Dry run (default — logs trades without executing)
python src/bot.py

# Live trading
BOT_DRY_RUN=false python src/bot.py

# Random strategy (no GitHub/OpenAI calls, for testing)
BOT_STRATEGY=random python src/bot.py
```

The bot uses `OpenclawStrategy` by default: fetches each PR live from GitHub, runs the ML
pipeline, and trades when the model probability differs from the market price by more than
`BOT_EDGE_THRESHOLD` (default 0.03).

### Key Bot Config (`.env`)

| Variable | Default | Description |
|---|---|---|
| `BOT_API_BASE_URL` | `http://127.0.0.1:8787` | WillItMerge API endpoint |
| `BOT_API_KEY` | — | WillItMerge API key |
| `BOT_STRATEGY` | `openclaw` | `openclaw` or `random` |
| `BOT_DRY_RUN` | `true` | `false` for live trading |
| `BOT_LOOP_MINUTES` | `10` | Cycle interval |
| `BOT_MARKETS_PER_CYCLE` | `10` | Markets evaluated per cycle |
| `BOT_EDGE_THRESHOLD` | `0.03` | Minimum edge to trade |
| `BOT_MAX_FRACTION_PER_MARKET` | `0.02` | Max bankroll fraction per trade |
| `BOT_MAX_DOLLARS_PER_MARKET` | `25` | Hard cap per trade |
| `GITHUB_TOKEN` | — | GitHub PAT for live PR fetching |
| `OPENAI_API_KEY` | — | For LLM features |

---

## Label Notes

- **Fork PRs only** — same-repo maintainer PRs are excluded (86% accept rate vs 21% for forks).
- **Local merges** — ~135 PRs closed on GitHub but cherry-picked to main are relabeled as
  positive using comment-text detection.
- **Deadline** — 14 days from PR creation. Open PRs past deadline are labeled negative.
  PRs not yet past deadline at collection time are excluded from training.
