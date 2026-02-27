"""
Main trading loop — bot V2.

Three-phase cycle:
  1. Fetch   — GET all open markets from the API
  2. Score   — parallel GitHub + ML scoring of every market
  3. Trade   — walk ranked results top-down, trade within caps
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bot import display
from bot.api import ApiError
from bot.scorer import score_all_markets

if TYPE_CHECKING:
    from bot.api import ApiClient
    from bot.config import BotConfig


def run_loop(config: "BotConfig", api: "ApiClient", strategy) -> None:
    loop_secs = config.loop_minutes * 60
    cycle = 0

    display.startup_banner(
        strategy_name=strategy.name,
        loop_minutes=config.loop_minutes,
        dry_run=config.dry_run,
        max_workers=config.max_workers,
        max_trades=config.max_trades_per_cycle,
        max_spend=config.max_spend_per_cycle,
    )

    while True:
        cycle += 1
        started = time.time()
        display.cycle_start(cycle)

        markets: list[dict] = []
        positions: dict[str, dict] = {}
        trades_executed = 0
        spend = 0.0
        bankroll = config.default_bankroll

        try:
            # ── Phase 1: Fetch ────────────────────────────────────────────────
            markets = api.get_open_markets()
            display.console.print(f"  [dim]{len(markets)} open markets fetched (prices live from chain)[/dim]")

            bankroll = config.default_bankroll
            try:
                me = api.get_me()
                bal = float(me.get("balance", 0))
                if bal > 0:
                    bankroll = bal
            except Exception as exc:
                display.warn(
                    f"could not fetch /me balance ({exc}), "
                    f"using default ${config.default_bankroll:.2f}"
                )

            positions = api.get_positions()
            display.console.print(
                f"  Bankroll: [cyan]${bankroll:.2f}[/cyan]  ·  "
                f"[dim]{len(positions)} position{'s' if len(positions) != 1 else ''} held[/dim]\n"
            )

            # ── Phase 2: Score (parallel) ─────────────────────────────────────
            with display.make_scoring_progress() as progress:
                task = progress.add_task(
                    "scoring",
                    total=len(markets),
                    errors=0,
                )

                def _on_progress(completed: int, total: int, errors: int) -> None:
                    progress.update(task, completed=completed, errors=errors)

                results = score_all_markets(
                    markets=markets,
                    strategy=strategy,
                    bankroll=bankroll,
                    config=config,
                    positions=positions,
                    on_progress=_on_progress,
                )

            display.console.print()

            # ── Phase 3: Trade ────────────────────────────────────────────────
            # Show opportunity table (display logic mirrors trade logic below)
            display.opportunity_table(
                results=results,
                positions=positions,
                edge_threshold=config.edge_threshold,
                max_trades=config.max_trades_per_cycle,
                max_spend=config.max_spend_per_cycle,
                dry_run=config.dry_run,
            )

            for result in results:
                # Skip unscoreable markets
                if result.edge is None or not result.plans:
                    continue
                if not result.market.get("pools_seeded"):
                    continue

                for plan in result.plans:
                    # In live mode, buy caps apply (sells always execute)
                    if not config.dry_run and plan.action == "buy":
                        if trades_executed >= config.max_trades_per_cycle:
                            continue
                        if spend + plan.max_cost > config.max_spend_per_cycle:
                            continue

                    if config.dry_run:
                        display.trade_placed(result.market, plan, None, None, dry_run=True)
                        if plan.action == "buy":
                            trades_executed += 1
                            spend += plan.max_cost
                    else:
                        try:
                            init = api.post_trade(result.market["id"], plan)
                            op_id = init.get("operation_id")
                            if not op_id:
                                raise ValueError(f"No operation_id in trade response: {init}")

                            final = api.poll_trade_op(result.market["id"], op_id)

                            if final.get("status") == "failed":
                                raise ApiError(
                                    final.get("error") or "trade failed",
                                    error_code="trade_failed",
                                    retryable=False,
                                    status=200,
                                )

                            shares  = abs(float(init.get("shares", 0)))
                            cost    = abs(float(init.get("cost",   0)))
                            tx_hash = (final.get("tx_hashes") or {}).get("swap")
                            display.trade_placed(
                                result.market, plan,
                                filled_shares=shares,
                                filled_cost=cost,
                                tx_hash=tx_hash,
                                dry_run=False,
                            )
                            if plan.action == "buy":
                                trades_executed += 1
                                spend += cost or plan.max_cost

                        except ApiError as exc:
                            display.trade_failed(result.market, plan, str(exc))
                            if exc.error_code in ("unauthenticated", "invalid_api_key"):
                                raise
                        except Exception as exc:
                            display.trade_failed(result.market, plan, str(exc))

        except ApiError as exc:
            display.error(f"cycle {cycle} API error [{exc.error_code}]: {exc}")
            if not exc.retryable:
                raise  # fatal auth/config errors should stop the bot
        except Exception as exc:
            display.error(f"cycle {cycle} failed: {exc}")

        duration = time.time() - started
        display.cycle_summary(
            cycle=cycle,
            total_markets=len(markets),
            trades_executed=trades_executed,
            spend=spend,
            bankroll=bankroll,
            duration=duration,
            loop_minutes=config.loop_minutes,
            dry_run=config.dry_run,
        )

        time.sleep(loop_secs)
