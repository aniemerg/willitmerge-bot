"""
Parallel market scoring for bot V2.

Scores all open markets concurrently using a ThreadPoolExecutor.
Each ScoredMarket carries the ordered list of TradePlans needed to reach
the target position from the current one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from bot.config import BotConfig
    from bot.trade import TradePlan


@dataclass
class ScoredMarket:
    market: dict
    target_prob: float | None
    edge: float | None          # signed: model_prob - price_yes
    abs_edge: float | None
    plans: list["TradePlan"] = field(default_factory=list)
    error: str | None = None


def score_all_markets(
    markets: list[dict],
    strategy,
    bankroll: float,
    config: "BotConfig",
    positions: dict[str, dict] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> list[ScoredMarket]:
    """
    Score every market in parallel. Returns all results sorted by abs_edge
    descending (scored markets first, then errors/skips).

    Args:
        markets:      All open markets from the API.
        strategy:     Strategy instance with target_yes_probability().
        bankroll:     Current bankroll for trade sizing.
        config:       Bot config.
        positions:    Dict of market_id → position from GET /me/positions.
        on_progress:  Optional callback(completed, total, errors).
    """
    from bot.trade import plan_trades_to_target

    positions = positions or {}
    results: list[ScoredMarket] = []
    completed_count = 0
    error_count = 0

    def _score_one(market: dict) -> ScoredMarket:
        try:
            target = strategy.target_yes_probability(market, config)

            if target is None:
                return ScoredMarket(
                    market=market,
                    target_prob=None, edge=None, abs_edge=None,
                )

            raw_price = market.get("price_yes")
            if raw_price is None:
                return ScoredMarket(
                    market=market,
                    target_prob=target, edge=None, abs_edge=None,
                    error="price unavailable",
                )

            price = float(raw_price)
            edge  = target - price
            position = positions.get(market.get("id", ""))

            plans = plan_trades_to_target(market, target, bankroll, config, position)

            return ScoredMarket(
                market=market,
                target_prob=target,
                edge=edge,
                abs_edge=abs(edge),
                plans=plans,
            )
        except Exception as exc:
            return ScoredMarket(
                market=market,
                target_prob=None, edge=None, abs_edge=None,
                error=str(exc),
            )

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        future_to_market = {pool.submit(_score_one, m): m for m in markets}

        for future in as_completed(future_to_market):
            result = future.result()
            results.append(result)
            completed_count += 1
            if result.error is not None:
                error_count += 1
            if on_progress:
                on_progress(completed_count, len(markets), error_count)

    # Sort: highest abs_edge first; errors/skips at the end
    results.sort(
        key=lambda r: (r.abs_edge is not None, r.abs_edge or 0.0),
        reverse=True,
    )
    return results
