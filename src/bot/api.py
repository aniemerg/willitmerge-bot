"""
WillItMerge REST API client.

Endpoints used:
  GET  /markets?status=open&limit=100&page={n}   → paginated { markets, total, page }
                                                   cached list payload; prices are metadata only
  GET  /markets/state?ids={id,...}               → live/synthetic price state for up to 100 markets
  GET  /markets/{id}/quote                       → quote opening buys (used for unactivated markets)
  GET  /me                                       → account info including balance
  GET  /me/positions                             → current token holdings by market
  POST /markets/{id}/trade                       → start async trade operation
  GET  /markets/{id}/trade/ops/{op_id}           → poll async trade operation status
  GET  /markets/{id}/activation/ops/{op_id}      → poll activation+trade operation status
"""

from __future__ import annotations

import time
import uuid

import httpx


class ApiError(Exception):
    """Raised when the API returns a structured error response."""
    def __init__(self, message: str, error_code: str | None, retryable: bool, status: int):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.status = status


# Error codes that indicate a caller bug or auth problem — do not retry.
_FATAL_ERROR_CODES = {
    "unauthenticated",
    "invalid_api_key",
    "invalid_request",
    "idempotency_conflict",
    "operation_market_mismatch",
    "forbidden",
    "legacy_endpoint_disabled",
}


def _raise_for_status(resp: httpx.Response) -> None:
    """Raise ApiError with structured details if the response is an error."""
    if resp.is_success:
        return
    try:
        body = resp.json()
        message   = body.get("error", resp.text)
        code      = body.get("error_code")
        retryable = body.get("retryable", resp.status_code >= 500)
        if code in _FATAL_ERROR_CODES:
            retryable = False
    except Exception:
        message, code, retryable = resp.text, None, resp.status_code >= 500
    raise ApiError(message, error_code=code, retryable=retryable, status=resp.status_code)


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(headers=headers, timeout=30.0)

    # ------------------------------------------------------------------
    def get_open_markets(self) -> list[dict]:
        """Fetch all open markets and hydrate current state.

        API response shape: { markets: [...], total: N, page: N }
        The list route is cached, so we overlay `/markets/state` to get the current
        activation state and price fields before the bot scores anything.
        """
        markets: list[dict] = []
        page = 1
        while True:
            resp = self._client.get(
                f"{self._base}/markets",
                params={"status": "open", "limit": 100, "page": page},
            )
            _raise_for_status(resp)
            data = resp.json()
            batch: list[dict] = data.get("markets", [])
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        self._hydrate_market_states(markets)
        return markets

    def _hydrate_market_states(self, markets: list[dict]) -> None:
        market_ids = [m.get("id") for m in markets if m.get("id")]
        if not market_ids:
            return
        try:
            states = self.get_market_states(market_ids)
        except Exception:
            return
        for market in markets:
            state = states.get(market.get("id", ""))
            if not state:
                continue
            market.update({
                "activation_state": state.get("activation_state", market.get("activation_state")),
                "price_yes": state.get("price_yes", market.get("price_yes")),
                "price_no": state.get("price_no", market.get("price_no")),
                "price_status": state.get("price_status", market.get("price_status")),
            })

    def get_market_states(self, market_ids: list[str]) -> dict[str, dict]:
        """Fetch live state for up to 100 markets per request."""
        states: dict[str, dict] = {}
        for start in range(0, len(market_ids), 100):
            batch = market_ids[start:start + 100]
            resp = self._client.get(
                f"{self._base}/markets/state",
                params={"ids": ",".join(batch)},
            )
            _raise_for_status(resp)
            data = resp.json()
            for state in data.get("states", []):
                market_id = state.get("market_id")
                if market_id:
                    states[market_id] = state
        return states

    def get_me(self) -> dict:
        """Fetch the authenticated user's account info (balance as formatted ether string)."""
        resp = self._client.get(f"{self._base}/me")
        _raise_for_status(resp)
        return resp.json()

    def get_positions(self) -> dict[str, dict]:
        """
        Fetch all current positions via GET /me/positions.
        Returns a dict keyed by market_id, each value having yes_shares and no_shares
        as formatted ether strings (e.g. "1.500000000000000000").
        Returns empty dict on error so callers can treat it as optional.
        """
        try:
            resp = self._client.get(f"{self._base}/me/positions")
            _raise_for_status(resp)
            data = resp.json()
            return {p["market_id"]: p for p in data.get("positions", []) if p}
        except Exception:
            return {}

    def post_trade(
        self,
        market_id: str,
        plan: "TradePlan",
        idempotency_key: str | None = None,
    ) -> dict:
        """
        Start an async trade operation for a buy or sell plan.

        Returns the initial response including operation_id and operation_status.
        Callers should then poll poll_trade_op() until a terminal status is reached.
        """
        from bot.trade import TradePlan  # local import to avoid circular

        key = idempotency_key or str(uuid.uuid4())

        if plan.action == "buy":
            body: dict = {"side": plan.side, "max_cost": plan.max_cost}
        else:
            # Sell: pass shares as a negative number
            body = {"side": plan.side, "shares": -plan.shares}

        resp = self._client.post(
            f"{self._base}/markets/{market_id}/trade",
            json=body,
            headers={"Idempotency-Key": key},
        )
        _raise_for_status(resp)
        return resp.json()

    def get_buy_quote(
        self,
        market_id: str,
        side: str,
        xdai_in: float,
        strategy: str = "pool_direct",
    ) -> dict:
        """Fetch a buy quote for the given notional size."""
        resp = self._client.get(
            f"{self._base}/markets/{market_id}/quote",
            params={
                "side": side,
                "strategy": strategy,
                "xdaiIn": f"{xdai_in:.6f}",
            },
        )
        _raise_for_status(resp)
        return resp.json()

    def poll_operation(
        self,
        market_id: str,
        operation_id: str,
        operation_kind: str = "trade",
        poll_interval: float = 2.0,
        timeout: float = 60.0,
    ) -> dict:
        """
        Poll the appropriate operation endpoint until the operation reaches a
        terminal state (completed or failed), then return the final state.

        Raises:
            ApiError: on non-retryable API errors or if status stays non-terminal
                      past timeout.
            TimeoutError: if the operation doesn't complete within `timeout` seconds.
        """
        path = (
            f"/markets/{market_id}/activation/ops/{operation_id}"
            if operation_kind == "activation_and_trade"
            else f"/markets/{market_id}/trade/ops/{operation_id}"
        )
        deadline = time.monotonic() + timeout
        while True:
            resp = self._client.get(f"{self._base}{path}")
            _raise_for_status(resp)
            state = resp.json()
            status = state.get("status")
            if status in ("completed", "failed"):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Trade operation {operation_id} still '{status}' after {timeout}s"
                )
            time.sleep(poll_interval)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
