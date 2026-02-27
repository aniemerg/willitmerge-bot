# Python Bot Port Plan (IMPLEMENTED)

All items below are complete. See `src/bot/` for the implementation.

Port the TypeScript `willitmerge-bot` into `openclaw-models/src/bot/` so the trading bot and
ML model live in one Python codebase with no inter-process communication.

---

## Why This Is Better

- `predict_pr()` becomes a direct function call — no HTTP microservice to run
- Model iteration (retrain → redeploy) is a single repo change
- All ML tooling (LightGBM, scikit-learn, SHAP) stays in the same language
- `httpx` is already a dependency

---

## New File Structure

```
openclaw-models/
  src/
    bot/
      __init__.py        ← empty
      config.py          ← env var loading (mirrors config.ts)
      api.py             ← WillItMerge REST client (mirrors api/client.ts)
      trade.py           ← trade sizing math (mirrors execution/tradeToTarget.ts)
      strategy.py        ← OpenclawStrategy + RandomStrategy
      loop.py            ← main trading loop (mirrors runner/loop.ts)
      logger.py          ← colored terminal output (mirrors logger.ts)
    bot.py               ← entry point: python src/bot.py
    github_live.py       ← GitHub GraphQL fetcher (NEW, needed by strategy)
    predict.py           ← existing inference entry point (unchanged)
    ...
```

---

## Component Mapping: TypeScript → Python

### `config.ts` → `src/bot/config.py`

Same env vars, same defaults. Use `os.environ.get()` with type coercion.
`rng` becomes a `random.Random` instance seeded from `BOT_RANDOM_SEED`.

```python
@dataclass
class BotConfig:
    api_base_url: str          = "http://127.0.0.1:8787"
    api_key: str | None        = None
    github_token: str | None   = None
    strategy: str              = "openclaw"    # default changed from "random"
    loop_minutes: int          = 10
    markets_per_cycle: int     = 10
    edge_threshold: float      = 0.03
    max_fraction_per_market: float = 0.02
    max_dollars_per_market: float  = 25.0
    min_dollars_per_trade: float   = 1.0
    default_bankroll: float    = 1000.0
    dry_run: bool              = True
    random_min_prob: float     = 0.2
    random_max_prob: float     = 0.8
    rng: random.Random         = field(default_factory=random.Random)

def load_config() -> BotConfig: ...
```

No new env vars needed — same `BOT_*` prefix as before.

---

### `api/client.ts` → `src/bot/api.py`

`httpx.Client` (sync), same three endpoints:

```python
class ApiClient:
    def get_open_markets(self) -> list[dict]: ...   # GET /markets?status=open (paginated)
    def get_me(self) -> dict: ...                   # GET /me
    def post_trade(self, market_id, side, max_cost) -> dict: ...  # POST /markets/{id}/trade
```

---

### `execution/tradeToTarget.ts` → `src/bot/trade.py`

Pure math, direct port:

```python
@dataclass
class TradePlan:
    side: str       # "yes" | "no"
    max_cost: float
    edge: float
    note: str

def plan_trade_to_target(
    market: dict, target_prob: float, bankroll: float, config: BotConfig
) -> TradePlan | None:
    edge = target_prob - market["price_yes"]
    if abs(edge) < config.edge_threshold:
        return None
    edge_strength = (abs(edge) - config.edge_threshold) / (1 - config.edge_threshold)
    dollars_by_fraction = bankroll * config.max_fraction_per_market * min(1.0, edge_strength)
    max_cost = min(config.max_dollars_per_market, dollars_by_fraction)
    if max_cost < config.min_dollars_per_trade:
        return None
    return TradePlan(side="yes" if edge > 0 else "no", max_cost=max_cost, edge=edge,
                     note=f"target={target_prob:.3f} price={market['price_yes']:.3f}")
```

---

### `strategies/` → `src/bot/strategy.py`

Two strategies in one file:

**OpenclawStrategy** (primary): calls `predict_pr()` directly.

```python
class OpenclawStrategy:
    name = "openclaw"

    def select_markets(self, markets, config) -> list[dict]:
        open_markets = [m for m in markets if m["status"] == "open"]
        return random.sample(open_markets, min(config.markets_per_cycle, len(open_markets)))

    def target_yes_probability(self, market: dict, config: BotConfig) -> float | None:
        from github_live import fetch_pr
        from predict import predict_pr
        pr = fetch_pr(market["repo_owner"], market["repo_name"],
                      market["pr_number"], config.github_token)
        if pr is None:
            return None
        # Compute deadline_days from market expires_at vs PR createdAt
        expires_dt = datetime.fromisoformat(market["expires_at"].replace("Z", "+00:00"))
        created_dt = datetime.fromisoformat(pr["createdAt"].replace("Z", "+00:00"))
        deadline_days = max(1, round((expires_dt - created_dt).total_seconds() / 86400))
        return predict_pr(pr, prediction_time=datetime.now(timezone.utc),
                          deadline_days=deadline_days)
```

**RandomStrategy** (fallback/testing): uniform random in [min, max].

---

### `runner/loop.ts` → `src/bot/loop.py`

Synchronous with `time.sleep`. Strategy calls are blocking (httpx + OpenAI are sync).

```python
def run_loop(config: BotConfig, api: ApiClient, strategy) -> None:
    cycle = 0
    while True:
        cycle += 1
        markets = api.get_open_markets()
        bankroll = _get_bankroll(api, config)
        selected = strategy.select_markets(markets, config)
        for market in selected:
            prob = strategy.target_yes_probability(market, config)
            if prob is None:
                continue
            plan = plan_trade_to_target(market, prob, bankroll, config)
            if plan is None:
                continue
            if config.dry_run:
                log_success(f"[dry-run] {_fmt(market)} side={plan.side} max_cost=${plan.max_cost:.2f} {plan.note}")
            else:
                result = api.post_trade(market["id"], plan.side, plan.max_cost)
                log_success(f"[trade] {_fmt(market)} filled ...")
        time.sleep(config.loop_minutes * 60)
```

---

### `data/github.ts` → `src/github_live.py`

New file. Fetches full PR dict via GitHub GraphQL matching the `prs.jsonl` schema that
`features.py` expects. This is the only meaningfully new code.

```python
def fetch_pr(owner: str, name: str, number: int, token: str | None) -> dict | None:
    """Fetch a live PR via GitHub GraphQL. Returns prs.jsonl-compatible dict."""
    # GraphQL query covers all fields needed by features.py:
    # title, body, state, merged, mergedAt, createdAt, updatedAt, isDraft, headRefName,
    # isCrossRepository, additions, deletions, changedFiles, author (login/createdAt/
    # followers/repos), labels, commits (totalCount + last CI status rollup),
    # reviews, comments/participants/reactions counts, reviewThreads, reviewRequests,
    # milestone, assignees, autoMergeRequest, closingIssuesReferences
```

---

### `src/index.ts` → `src/bot.py`

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import env  # loads .env
from bot.config import load_config
from bot.api import ApiClient
from bot.strategy import OpenclawStrategy, RandomStrategy
from bot.loop import run_loop

def main():
    config = load_config()
    api = ApiClient(config.api_base_url, config.api_key)
    strategy = OpenclawStrategy() if config.strategy == "openclaw" else RandomStrategy()
    run_loop(config, api, strategy)

if __name__ == "__main__":
    main()
```

---

### `logger.ts` → `src/bot/logger.py`

ANSI color codes, same visual style. No new dependencies needed.

---

## Dependencies to Add to `requirements.txt`

None needed — `httpx` is already present.

---

## What Happens to the TypeScript Bot

The `willitmerge-bot` TypeScript code can be kept as-is or archived — it still works independently.
The Python bot is an independent implementation, not a deletion of the TS one.

---

## Running

```bash
cd openclaw-models

# Normal run (dry-run by default)
python src/bot.py

# Live trading
BOT_DRY_RUN=false python src/bot.py

# Random strategy for testing (no GitHub/OpenAI calls)
BOT_STRATEGY=random python src/bot.py
```

---

## Build Order

1. `src/github_live.py` — new, needed by strategy; test with a known PR number
2. `src/bot/config.py` — env loading
3. `src/bot/api.py` — WillItMerge client; test with `get_open_markets()`
4. `src/bot/trade.py` — math, no I/O
5. `src/bot/logger.py` — cosmetic
6. `src/bot/strategy.py` — integrates github_live + predict_pr
7. `src/bot/loop.py` — wires it all together
8. `src/bot.py` — entry point

---

## Verification

1. `BOT_STRATEGY=random BOT_DRY_RUN=true python src/bot.py` — one cycle completes, shows random trades
2. `BOT_STRATEGY=openclaw BOT_DRY_RUN=true python src/bot.py` — one cycle, shows `openclaw_model` probabilities
3. Confirm `cache/llm_features.json` grows as new PRs are scored
4. Confirm probabilities match `python src/predict.py --pr-number X` for the same PR
