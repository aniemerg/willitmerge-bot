"""
WillItMerge trading bot — Python port of willitmerge-bot.

Usage:
    python src/bot.py

Configuration via environment variables (BOT_* prefix).
See .env.example for all available options.

Key env vars:
    BOT_API_BASE_URL           Market API endpoint (default: http://127.0.0.1:8787)
    BOT_API_KEY                Bearer token for market API (optional)
    BOT_GITHUB_TOKEN           GitHub token for fetching live PR data
    BOT_STRATEGY               'openclaw' (default) or 'random'
    BOT_DRY_RUN                true (default) — set to false for live trading
    BOT_LOOP_MINUTES           Minutes between cycles (default: 10)
    BOT_MAX_WORKERS            Parallel scoring threads (default: 20)
    BOT_MAX_TRADES_PER_CYCLE   Max trades placed per cycle (default: 5)
    BOT_MAX_SPEND_PER_CYCLE    Max $ spent per cycle (default: 50)
    BOT_EDGE_THRESHOLD         Min probability edge to trade (default: 0.03)
    BOT_MAX_DOLLARS_PER_MARKET Max $ per single trade (default: 25)
"""

import sys
from pathlib import Path

# Ensure src/ is on the path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

import env  # loads .env before any other imports read os.environ

from bot.api import ApiClient
from bot.config import load_config
from bot.loop import run_loop
from bot.strategy import OpenclawStrategy, RandomStrategy
from bot import display


def main() -> None:
    config = load_config()

    if config.strategy == "openclaw":
        strategy = OpenclawStrategy()
    else:
        strategy = RandomStrategy()

    with ApiClient(config.api_base_url, config.api_key) as api:
        run_loop(config, api, strategy)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        display.console.print("\n[dim]Interrupted — shutting down.[/dim]")
    except Exception as exc:
        display.error(str(exc))
        sys.exit(1)
