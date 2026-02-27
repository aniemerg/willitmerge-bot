# Bot V2 Design

Status: **implemented**

---

## Overview

The bot runs a continuous loop, waking every `BOT_LOOP_MINUTES` (default: 10). Each cycle has
three phases: fetch all open markets, score them in parallel, then execute the trades needed to
move every position to its target.

---

## Three-Phase Cycle

```
Phase 1 — Fetch   (fast, ~1s)
  GET /markets?status=open  →  all open markets with pools_seeded=True
  GET /me                   →  current bankroll
  GET /me/positions         →  current token holdings by market

Phase 2 — Score   (parallel, slow)
  ThreadPoolExecutor(max_workers=BOT_MAX_WORKERS)
  for each market: GitHub GraphQL fetch + predict_pr() → target probability

Phase 3 — Trade   (serial)
  sort markets by |edge| descending
  for each market: compute trade plans, execute in order
```

---

## Position Sizing: Trade to Target

The bot does not blindly add to positions each cycle. Instead, it computes a **target position
in dollars** for each market and trades only the delta needed to reach it.

### Target calculation

```
edge          = model_probability - market_price_yes
edge_strength = (|edge| - threshold) / (1 - threshold)   # 0 → 1

target_dollars = min(
    BOT_MAX_DOLLARS_PER_MARKET,
    bankroll × BOT_MAX_FRACTION_PER_MARKET × edge_strength
)
```

If `|edge| < BOT_EDGE_THRESHOLD`, the target is **zero** — exit any existing position.

The target side is YES when `edge > 0`, NO when `edge < 0`.

### Current position value

Current holdings (from `GET /me/positions`) are converted to dollars using the current market
price as an approximation:

```
current_dollars = shares × price_yes          # for YES holdings
current_dollars = shares × (1 - price_yes)   # for NO holdings
```

### Trade decision logic

Given target side, target dollars, and current position, the bot determines what to do:

| Situation | Action |
|---|---|
| Holding wrong side (e.g. have NO but target is YES) | Sell all wrong-side shares, then buy right-side to target |
| Below target on correct side | Buy the delta |
| Above target on correct side | Sell the excess |
| Already at target (delta < `BOT_MIN_DOLLARS_PER_TRADE`) | No trade — show "at target" |
| Edge below threshold | Exit entire position (sell all); target = 0 |
| No position, edge below threshold | No trade |

Wrong-side sells and right-side buys happen **in the same cycle** when the model flips.

### Caps

- **Sells** are never capped — reducing or exiting a position always executes.
- **Buys** are subject to `BOT_MAX_TRADES_PER_CYCLE` and `BOT_MAX_SPEND_PER_CYCLE` in live mode.
- Caps are **not applied in dry-run** so the full opportunity set is visible.

---

## Parallel Scoring

`ThreadPoolExecutor(max_workers=BOT_MAX_WORKERS)` scores all open markets concurrently.
Each worker: GitHub GraphQL fetch → LightGBM + isotonic calibration → TradePlan list.

`BOT_MAX_WORKERS` (default: 20) is the primary rate-limit dial. At 20 workers and a 10-minute
cycle the bot can comfortably handle hundreds of markets within GitHub's API rate limits.

---

## Trade Execution

Trades use the async `POST /markets/:id/trade` endpoint:

1. POST returns `operation_id` and `operation_status: "pending"` immediately
2. Bot polls `GET /markets/:id/trade/ops/:op_id` every 2s until `completed` or `failed`
3. Fill details (`shares`, `cost`) come from the initial POST response
4. Swap tx hash comes from the completed operation state

Each POST includes an `Idempotency-Key` to prevent duplicate trades on retry.

Markets with `pools_seeded=False` are scored and displayed but never traded (the API would
reject the trade).

---

## Error Handling

Errors from the API are parsed as `{ error, error_code, retryable }`:

- `unauthenticated`, `invalid_api_key` — fatal, stop the bot
- `trade_failed` — log and continue to the next market
- `operation_unavailable`, `internal_error` — log and continue (retryable in a future cycle)

A scoring error on one market (GitHub timeout, model error) never blocks other markets.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `BOT_STRATEGY` | `openclaw` | `openclaw` (ML model) or `random` (testing) |
| `BOT_DRY_RUN` | `true` | Set `false` for live trading |
| `BOT_LOOP_MINUTES` | `10` | Minutes between cycles |
| `BOT_MAX_WORKERS` | `20` | Parallel scoring threads |
| `BOT_MAX_TRADES_PER_CYCLE` | `5` | Max buys per cycle (live only) |
| `BOT_MAX_SPEND_PER_CYCLE` | `50` | Max $ spent on buys per cycle (live only) |
| `BOT_EDGE_THRESHOLD` | `0.03` | Min \|edge\| to open or hold a position |
| `BOT_MAX_FRACTION_PER_MARKET` | `0.02` | Max bankroll fraction per market |
| `BOT_MAX_DOLLARS_PER_MARKET` | `25` | Hard dollar cap per position |
| `BOT_MIN_DOLLARS_PER_TRADE` | `0.01` | Minimum trade size |
| `BOT_DEFAULT_BANKROLL` | `1000` | Fallback if `/me` balance fetch fails |
| `GITHUB_TOKEN` or `BOT_GITHUB_TOKEN` | — | GitHub token for live PR fetching |
| `BOT_API_KEY` | — | WillItMerge API bearer token |

---

## Running

```bash
# Dry run (default) — no trades placed, full output shown
python src/bot.py

# Live trading
BOT_DRY_RUN=false python src/bot.py

# Random strategy — no GitHub or OpenAI calls, useful for testing display
BOT_STRATEGY=random python src/bot.py
```
