"""
WillItMerge REST API client.

Endpoints used:
  GET  /markets?status=open&limit=100&page={n}  → paginated { markets, total, page }
                                                   includes live on-chain price_yes/price_no
  GET  /me                                       → account info including balance
  GET  /me/positions                             → current token holdings by market
  POST /markets/{id}/trade                       → start async trade operation
  GET  /markets/{id}/trade/ops/{op_id}           → poll async trade operation status
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
        """Fetch all open markets (handles pagination).

        API response shape: { markets: [...], total: N, page: N }
        Only returns markets with pools_seeded=True (untradeble markets skipped).
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
        return markets

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

    def poll_trade_op(
        self,
        market_id: str,
        operation_id: str,
        poll_interval: float = 2.0,
        timeout: float = 60.0,
    ) -> dict:
        """
        Poll GET /markets/{id}/trade/ops/{op_id} until the operation reaches a
        terminal state (completed or failed), then return the final state.

        Raises:
            ApiError: on non-retryable API errors or if status stays non-terminal
                      past timeout.
            TimeoutError: if the operation doesn't complete within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            resp = self._client.get(
                f"{self._base}/markets/{market_id}/trade/ops/{operation_id}"
            )
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
