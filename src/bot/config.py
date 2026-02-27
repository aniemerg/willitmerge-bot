"""
Bot configuration — mirrors willitmerge-bot src/config.ts.
All settings come from BOT_* environment variables with sensible defaults.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field


def _str(name: str, fallback: str | None = None) -> str:
    val = os.environ.get(name, "")
    if not val:
        if fallback is None:
            raise ValueError(f"Missing required env var: {name}")
        return fallback
    return val


def _float(name: str, fallback: float) -> float:
    val = os.environ.get(name, "")
    if not val:
        return fallback
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Invalid number for {name}: {val!r}")


def _int(name: str, fallback: int) -> int:
    return int(_float(name, float(fallback)))


def _bool(name: str, fallback: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return fallback
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean for {name}: {val!r}")


def _clamp01(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass
class BotConfig:
    api_base_url: str
    api_key: str | None
    github_token: str | None
    strategy: str
    loop_minutes: int
    max_workers: int               # thread pool size for parallel scoring
    max_trades_per_cycle: int      # max trades placed per cycle
    max_spend_per_cycle: float     # dollar cap per cycle
    edge_threshold: float
    max_fraction_per_market: float
    max_dollars_per_market: float
    min_dollars_per_trade: float
    default_bankroll: float
    dry_run: bool
    random_min_prob: float
    random_max_prob: float
    rng: random.Random = field(default_factory=random.Random, repr=False)


def load_config() -> BotConfig:
    strategy = _str("BOT_STRATEGY", "openclaw")
    if strategy not in ("openclaw", "random"):
        raise ValueError(f"Unsupported BOT_STRATEGY: {strategy!r}. Choose 'openclaw' or 'random'.")

    random_min = _clamp01(_float("BOT_RANDOM_MIN_PROB", 0.2), "BOT_RANDOM_MIN_PROB")
    random_max = _clamp01(_float("BOT_RANDOM_MAX_PROB", 0.8), "BOT_RANDOM_MAX_PROB")
    if random_max <= random_min:
        raise ValueError("BOT_RANDOM_MAX_PROB must be greater than BOT_RANDOM_MIN_PROB")

    seed_raw = os.environ.get("BOT_RANDOM_SEED", "")
    rng = random.Random(seed_raw if seed_raw else None)

    return BotConfig(
        api_base_url=_str("BOT_API_BASE_URL", "http://127.0.0.1:8787"),
        api_key=os.environ.get("BOT_API_KEY") or None,
        github_token=os.environ.get("BOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or None,
        strategy=strategy,
        loop_minutes=max(1, _int("BOT_LOOP_MINUTES", 10)),
        max_workers=max(1, _int("BOT_MAX_WORKERS", 20)),
        max_trades_per_cycle=max(1, _int("BOT_MAX_TRADES_PER_CYCLE", 5)),
        max_spend_per_cycle=max(0.01, _float("BOT_MAX_SPEND_PER_CYCLE", 50.0)),
        edge_threshold=_clamp01(_float("BOT_EDGE_THRESHOLD", 0.03), "BOT_EDGE_THRESHOLD"),
        max_fraction_per_market=_clamp01(
            _float("BOT_MAX_FRACTION_PER_MARKET", 0.02), "BOT_MAX_FRACTION_PER_MARKET"
        ),
        max_dollars_per_market=max(0.01, _float("BOT_MAX_DOLLARS_PER_MARKET", 25.0)),
        min_dollars_per_trade=max(0.01, _float("BOT_MIN_DOLLARS_PER_TRADE", 0.01)),
        default_bankroll=max(1.0, _float("BOT_DEFAULT_BANKROLL", 1000.0)),
        dry_run=_bool("BOT_DRY_RUN", True),
        random_min_prob=random_min,
        random_max_prob=random_max,
        rng=rng,
    )
