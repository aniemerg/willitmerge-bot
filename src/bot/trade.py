"""
Trade sizing — position-aware "trade to target" logic.

For each market the bot computes a target position in dollars based on
edge and bankroll, then returns the trades needed to reach it from the
current position (which may be zero, partial, wrong-side, or oversized).

Up to two TradePlans can be returned per market (sell wrong side + buy
right side). The loop executes them in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.api import ApiClient
    from bot.config import BotConfig


@dataclass
class TradePlan:
    action: str      # "buy" | "sell"
    side: str        # "yes" | "no" — which token
    max_cost: float  # buy only: max xDAI to spend
    shares: float    # sell only: number of shares to sell (positive)
    edge: float      # signed edge (for reference / display)
    note: str


@dataclass
class TradeDecision:
    plans: list[TradePlan]
    reference_price: float | None
    edge: float | None


def _budget_cap(bankroll: float, config: "BotConfig") -> float:
    return min(
        config.max_dollars_per_market,
        bankroll * config.max_fraction_per_market,
    )


def _strength(edge: float, threshold: float, max_edge: float) -> float:
    if edge <= threshold or max_edge <= threshold:
        return 0.0
    return min(1.0, (edge - threshold) / (max_edge - threshold))


def plan_trades_to_target(
    market: dict,
    target_prob: float,
    bankroll: float,
    config: "BotConfig",
    api: "ApiClient",
    position: dict | None = None,
) -> TradeDecision:
    """
    Return the trades needed to move from the current position to the target
    position implied by target_prob, plus the reference price used for edge.

    Sells always come before buys so the loop can execute them in order.
    """
    activation_state = market.get("activation_state")
    if activation_state == "activating":
        return TradeDecision(plans=[], reference_price=None, edge=None)
    if activation_state == "unactivated":
        return _plan_activation_trade(market, target_prob, bankroll, config, api)

    raw_price = market.get("price_yes")
    if raw_price is None or market.get("price_status") != "live":
        return TradeDecision(plans=[], reference_price=None, edge=None)
    price_yes = float(raw_price)
    price_no  = 1.0 - price_yes

    edge = target_prob - price_yes

    # ── Determine target ──────────────────────────────────────────────────────
    if abs(edge) < config.edge_threshold:
        target_dollars = 0.0
        target_side    = None
    else:
        edge_strength  = (abs(edge) - config.edge_threshold) / (1.0 - config.edge_threshold)
        target_dollars = min(
            config.max_dollars_per_market,
            bankroll * config.max_fraction_per_market * min(1.0, edge_strength),
        )
        target_side = "yes" if edge > 0 else "no"

    # ── Current position ──────────────────────────────────────────────────────
    yes_shares = float((position or {}).get("yes_shares") or 0)
    no_shares  = float((position or {}).get("no_shares")  or 0)

    current_yes_dollars = yes_shares * price_yes
    current_no_dollars  = no_shares  * price_no

    plans: list[TradePlan] = []

    note_base = (
        f"model={target_prob:.3f} price={price_yes:.3f} "
        f"edge={edge:+.3f} target=${target_dollars:.2f}"
    )

    # ── Exit or reduce wrong-side position ────────────────────────────────────
    # Sell any shares on the side we're NOT targeting (includes "target = None").
    wrong_yes = yes_shares if target_side != "yes" else 0.0
    wrong_no  = no_shares  if target_side != "no"  else 0.0

    if wrong_yes * price_yes >= config.min_dollars_per_trade:
        plans.append(TradePlan(
            action="sell", side="yes",
            max_cost=0, shares=wrong_yes,
            edge=edge,
            note=f"exit YES  {note_base}",
        ))
    if wrong_no * price_no >= config.min_dollars_per_trade:
        plans.append(TradePlan(
            action="sell", side="no",
            max_cost=0, shares=wrong_no,
            edge=edge,
            note=f"exit NO  {note_base}",
        ))

    # ── Trade toward target on the right side ─────────────────────────────────
    if target_side is None:
        return TradeDecision(plans=plans, reference_price=price_yes, edge=edge)

    current_dollars = current_yes_dollars if target_side == "yes" else current_no_dollars
    delta = target_dollars - current_dollars

    if delta >= config.min_dollars_per_trade:
        # Buy more
        plans.append(TradePlan(
            action="buy", side=target_side,
            max_cost=min(delta, config.max_dollars_per_market),
            shares=0,
            edge=edge,
            note=f"buy {target_side.upper()}  {note_base} current=${current_dollars:.2f}",
        ))
    elif delta <= -config.min_dollars_per_trade:
        # Trim position
        price    = price_yes if target_side == "yes" else price_no
        to_sell  = (-delta) / price if price > 0 else 0.0
        if to_sell * price >= config.min_dollars_per_trade:
            plans.append(TradePlan(
                action="sell", side=target_side,
                max_cost=0, shares=to_sell,
                edge=edge,
                note=f"trim {target_side.upper()}  {note_base} current=${current_dollars:.2f}",
            ))

    return TradeDecision(plans=plans, reference_price=price_yes, edge=edge)


def _plan_activation_trade(
    market: dict,
    target_prob: float,
    bankroll: float,
    config: "BotConfig",
    api: "ApiClient",
) -> TradeDecision:
    """
    Opening activation is BUY YES only. We size a candidate notional within the
    configured risk budget, then quote that exact opening trade and compare the
    model probability against the quoted post-open YES price.
    """
    opening_min = float(market.get("activation_min_opening_buy_xdai") or 0)
    budget_cap = _budget_cap(bankroll, config)
    confidence_edge = target_prob - 0.5

    if target_prob <= 0.5 or confidence_edge < config.edge_threshold:
        return TradeDecision(plans=[], reference_price=None, edge=None)
    if opening_min <= 0 or budget_cap < opening_min:
        return TradeDecision(plans=[], reference_price=None, edge=None)

    strength = _strength(confidence_edge, config.edge_threshold, 0.5)
    candidate_spend = max(opening_min, budget_cap * strength)
    spend = min(budget_cap, candidate_spend)

    quote = api.get_buy_quote(
        market_id=market["id"],
        side="yes",
        xdai_in=spend,
        strategy="mint_rebalance",
    )
    quoted_price = float(
        quote.get("yesPriceAfter")
        or quote.get("priceAfter")
        or quote.get("priceBefore")
        or 0
    )
    edge = target_prob - quoted_price
    note_base = (
        f"activation model={target_prob:.3f} quoted={quoted_price:.3f} "
        f"edge={edge:+.3f} spend=${spend:.2f}"
    )

    if edge < config.edge_threshold:
        return TradeDecision(plans=[], reference_price=quoted_price, edge=edge)

    return TradeDecision(
        plans=[
            TradePlan(
                action="buy",
                side="yes",
                max_cost=spend,
                shares=0,
                edge=edge,
                note=f"open YES  {note_base}",
            )
        ],
        reference_price=quoted_price,
        edge=edge,
    )
