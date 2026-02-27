"""
Trading strategies.

OpenclawStrategy (default): calls predict_pr() directly — no HTTP boundary.
RandomStrategy: uniform random target, useful for testing without API keys.

V2 note: select_markets() has been removed. The loop now scores all markets
in parallel and ranks by edge. Strategies are responsible only for producing
a target probability for a single market.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.config import BotConfig


class OpenclawStrategy:
    name = "openclaw"

    def target_yes_probability(
        self, market: dict, config: "BotConfig"
    ) -> float | None:
        """
        Fetch the live PR from GitHub and run the ML model.
        Returns calibrated P(merged within deadline) or None to skip.
        """
        from github_live import fetch_pr
        from predict import predict_pr

        pr = fetch_pr(
            market["repo_owner"],
            market["repo_name"],
            market["pr_number"],
            token=config.github_token,
        )
        if pr is None:
            return None

        # Derive deadline_days from how long the market window is
        try:
            expires_dt = datetime.fromisoformat(
                market["expires_at"].replace("Z", "+00:00")
            )
            created_dt = datetime.fromisoformat(
                pr["createdAt"].replace("Z", "+00:00")
            )
            deadline_days = max(1, round(
                (expires_dt - created_dt).total_seconds() / 86400
            ))
        except Exception:
            deadline_days = None  # predict_pr will use metadata default

        return predict_pr(
            pr,
            prediction_time=datetime.now(timezone.utc),
            deadline_days=deadline_days,
        )


class RandomStrategy:
    name = "random"

    def target_yes_probability(
        self, market: dict, config: "BotConfig"
    ) -> float | None:
        span = config.random_max_prob - config.random_min_prob
        return config.random_min_prob + config.rng.random() * span
