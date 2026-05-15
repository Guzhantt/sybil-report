"""
Market Finder - Discovers active BTC 5m markets and provides real-time orderbook data.

Enhanced capabilities:
1. Auto-discover current active BTC 5m market from Gamma API
2. Fetch full orderbook depth from CLOB API
3. Calculate orderbook imbalance metrics (bid/ask pressure)
4. Track spread dynamics and liquidity changes
5. Detect large resting orders that signal informed flow

The BTC Up or Down 5m series (slug: btc-up-or-down-5m) creates a new market
every 5 minutes. Resolution uses Chainlink BTC/USD data stream.
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SERIES_SLUG = "btc-up-or-down-5m"
CLOB_BASE_URL = "https://clob.polymarket.com"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class OrderbookLevel:
    """A single price level in the orderbook."""
    price: float
    size: float  # In token units (shares)

    @property
    def notional(self) -> float:
        """USDC value of this level."""
        return self.price * self.size


@dataclass
class Orderbook:
    """Full orderbook for one token (Up or Down)."""
    bids: list[OrderbookLevel] = field(default_factory=list)
    asks: list[OrderbookLevel] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.5

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid."""
        mid = self.mid_price
        if mid <= 0:
            return 0
        return (self.spread / mid) * 10000

    @property
    def total_bid_depth(self) -> float:
        """Total USDC on the bid side (top 10 levels)."""
        return sum(lvl.notional for lvl in self.bids[:10])

    @property
    def total_ask_depth(self) -> float:
        """Total USDC on the ask side (top 10 levels)."""
        return sum(lvl.notional for lvl in self.asks[:10])

    @property
    def imbalance(self) -> float:
        """
        Orderbook imbalance: -1 (all asks) to +1 (all bids).
        Positive = more buying pressure.
        """
        bid_d = self.total_bid_depth
        ask_d = self.total_ask_depth
        total = bid_d + ask_d
        if total == 0:
            return 0.0
        return (bid_d - ask_d) / total

    @property
    def weighted_mid(self) -> float:
        """Volume-weighted mid price (more accurate than simple mid)."""
        if not self.bids or not self.asks:
            return self.mid_price
        bid_vol = self.bids[0].size
        ask_vol = self.asks[0].size
        total = bid_vol + ask_vol
        if total == 0:
            return self.mid_price
        return (self.best_bid * ask_vol + self.best_ask * bid_vol) / total

    def depth_at_price(self, price: float, side: str = "bid") -> float:
        """Get total size available at or better than a given price."""
        levels = self.bids if side == "bid" else self.asks
        total = 0.0
        for lvl in levels:
            if side == "bid" and lvl.price >= price:
                total += lvl.size
            elif side == "ask" and lvl.price <= price:
                total += lvl.size
        return total

    def large_orders(self, threshold: float = 100.0) -> list[OrderbookLevel]:
        """Find large resting orders (potential informed flow)."""
        large = []
        for lvl in self.bids + self.asks:
            if lvl.notional >= threshold:
                large.append(lvl)
        return large


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    # Identity
    market_id: str = ""
    condition_id: str = ""
    question: str = ""
    slug: str = ""

    # Token IDs for CLOB
    up_token_id: str = ""
    down_token_id: str = ""

    # Orderbooks
    book_up: Orderbook = field(default_factory=Orderbook)
    book_down: Orderbook = field(default_factory=Orderbook)

    # Timing
    event_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accepting_orders: bool = False

    # Derived metrics (updated on each refresh)
    up_price: float = 0.5
    down_price: float = 0.5

    @property
    def seconds_until_close(self) -> float:
        now = datetime.now(timezone.utc)
        return max(0, (self.end_time - now).total_seconds())

    @property
    def seconds_since_start(self) -> float:
        now = datetime.now(timezone.utc)
        return max(0, (now - self.event_start).total_seconds())

    @property
    def progress_pct(self) -> float:
        total = (self.end_time - self.event_start).total_seconds()
        if total <= 0:
            return 100.0
        return min(100.0, max(0.0, (self.seconds_since_start / total) * 100.0))

    @property
    def is_tradeable(self) -> bool:
        return self.accepting_orders and self.seconds_until_close > 10

    @property
    def best_bid_up(self) -> float:
        return self.book_up.best_bid

    @property
    def best_ask_up(self) -> float:
        return self.book_up.best_ask

    @property
    def best_bid_down(self) -> float:
        return self.book_down.best_bid

    @property
    def best_ask_down(self) -> float:
        return self.book_down.best_ask

    @property
    def spread_up(self) -> float:
        return self.book_up.spread

    @property
    def spread_down(self) -> float:
        return self.book_down.spread

    @property
    def book_imbalance_up(self) -> float:
        return self.book_up.imbalance

    @property
    def book_imbalance_down(self) -> float:
        return self.book_down.imbalance

    @property
    def total_liquidity(self) -> float:
        """Total USDC liquidity across both books."""
        return (
            self.book_up.total_bid_depth + self.book_up.total_ask_depth +
            self.book_down.total_bid_depth + self.book_down.total_ask_depth
        )


# ─── Market Finder ────────────────────────────────────────────────────────────

class MarketFinder:
    """
    Discovers and tracks the currently active BTC 5m market.
    Provides real-time orderbook data and market microstructure metrics.
    """

    def __init__(self, gamma_api_url: str = "https://gamma-api.polymarket.com"):
        self.gamma_api_url = gamma_api_url
        self._client: Optional[httpx.AsyncClient] = None
        self._current_snapshot: Optional[MarketSnapshot] = None
        self._last_market_fetch: float = 0
        self._last_book_fetch: float = 0
        self._market_cache_ttl: float = 10.0  # Refetch market identity every 10s
        self._book_cache_ttl: float = 1.5     # Refetch orderbook every 1.5s

        # Historical spread tracking
        self._spread_history: list[float] = []
        self._imbalance_history: list[float] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=8.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_snapshot(self, force_refresh: bool = False) -> Optional[MarketSnapshot]:
        """
        Get a complete market snapshot including orderbook.
        This is the main entry point called by the Trader.
        
        Returns None if no active market exists.
        """
        now = time.time()

        # Step 1: Ensure we have the current market identity
        need_market_refresh = (
            force_refresh
            or self._current_snapshot is None
            or not self._current_snapshot.is_tradeable
            or (now - self._last_market_fetch) > self._market_cache_ttl
        )

        if need_market_refresh:
            snapshot = await self._fetch_market()
            if snapshot is None:
                return None
            self._current_snapshot = snapshot
            self._last_market_fetch = now

        # Step 2: Refresh orderbook data (more frequent)
        if (now - self._last_book_fetch) > self._book_cache_ttl:
            await self._refresh_orderbooks()
            self._last_book_fetch = now

        return self._current_snapshot

    async def _fetch_market(self) -> Optional[MarketSnapshot]:
        """Fetch the current active BTC 5m market from Gamma API."""
        client = await self._get_client()

        try:
            # Query by series slug - descending to get NEWEST markets first
            params = {
                "active": "true",
                "closed": "false",
                "series_slug": SERIES_SLUG,
                "limit": "5",
                "order": "endDate",
                "ascending": "false",
            }

            response = await client.get(f"{self.gamma_api_url}/events", params=params)
            response.raise_for_status()
            events = response.json()

            now = datetime.now(timezone.utc)

            for event in events:
                markets = event.get("markets", [])
                for mkt in markets:
                    # Check if market is accepting orders
                    accepting = mkt.get("acceptingOrders", False) or mkt.get("enableOrderBook", False)
                    if not accepting:
                        # Also check if it has orderbook enabled at event level
                        if not event.get("enableOrderBook", False):
                            continue

                    end_str = mkt.get("endDate", "")
                    if not end_str:
                        continue

                    end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    if end_time <= now:
                        continue  # Already expired

                    # Found active market!
                    return self._parse_market(mkt, event)

            # Fallback: direct market search
            return await self._fallback_search()

        except httpx.HTTPStatusError as e:
            logger.error(f"Gamma API HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching market: {e}")
            return None

    async def _fallback_search(self) -> Optional[MarketSnapshot]:
        """Fallback search for BTC 5m markets."""
        client = await self._get_client()

        try:
            params = {
                "active": "true",
                "closed": "false",
                "limit": "20",
                "order": "endDate",
                "ascending": "true",
            }
            response = await client.get(f"{self.gamma_api_url}/markets", params=params)
            response.raise_for_status()
            markets = response.json()

            now = datetime.now(timezone.utc)
            for mkt in markets:
                slug = mkt.get("slug", "")
                if "btc-updown-5m" not in slug:
                    continue
                if not mkt.get("acceptingOrders", False):
                    continue
                end_str = mkt.get("endDate", "")
                if not end_str:
                    continue
                end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_time <= now:
                    continue
                return self._parse_market(mkt)

        except Exception as e:
            logger.error(f"Fallback search error: {e}")

        return None

    def _parse_market(self, mkt: dict, event: Optional[dict] = None) -> MarketSnapshot:
        """Parse raw API data into a MarketSnapshot."""
        # Parse token IDs
        clob_ids_raw = mkt.get("clobTokenIds", "[]")
        if isinstance(clob_ids_raw, str):
            clob_ids = json.loads(clob_ids_raw)
        else:
            clob_ids = clob_ids_raw

        # Parse prices
        prices_raw = mkt.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        up_price = float(prices[0]) if len(prices) > 0 else 0.5
        down_price = float(prices[1]) if len(prices) > 1 else 0.5

        # Parse timing
        end_time = datetime.fromisoformat(mkt["endDate"].replace("Z", "+00:00"))

        event_start_str = mkt.get("eventStartTime", "")
        if not event_start_str and event:
            event_start_str = event.get("startTime", "")
        if event_start_str:
            event_start = datetime.fromisoformat(event_start_str.replace("Z", "+00:00"))
        else:
            event_start = end_time - timedelta(minutes=5)

        return MarketSnapshot(
            market_id=mkt.get("id", ""),
            condition_id=mkt.get("conditionId", ""),
            question=mkt.get("question", ""),
            slug=mkt.get("slug", ""),
            up_token_id=clob_ids[0] if len(clob_ids) > 0 else "",
            down_token_id=clob_ids[1] if len(clob_ids) > 1 else "",
            event_start=event_start,
            end_time=end_time,
            accepting_orders=mkt.get("acceptingOrders", False),
            up_price=up_price,
            down_price=down_price,
        )

    async def _refresh_orderbooks(self):
        """Fetch fresh orderbook data for both tokens."""
        if not self._current_snapshot:
            return

        snapshot = self._current_snapshot

        # Fetch both orderbooks in parallel
        tasks = []
        if snapshot.up_token_id:
            tasks.append(self._fetch_orderbook(snapshot.up_token_id))
        if snapshot.down_token_id:
            tasks.append(self._fetch_orderbook(snapshot.down_token_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        if len(results) >= 1 and not isinstance(results[0], Exception):
            snapshot.book_up = results[0]
            snapshot.up_price = snapshot.book_up.mid_price

        if len(results) >= 2 and not isinstance(results[1], Exception):
            snapshot.book_down = results[1]
            snapshot.down_price = snapshot.book_down.mid_price

        # Track history
        if snapshot.book_up.spread > 0:
            self._spread_history.append(snapshot.book_up.spread)
            if len(self._spread_history) > 100:
                self._spread_history = self._spread_history[-100:]

        self._imbalance_history.append(snapshot.book_up.imbalance)
        if len(self._imbalance_history) > 100:
            self._imbalance_history = self._imbalance_history[-100:]

    async def _fetch_orderbook(self, token_id: str) -> Orderbook:
        """Fetch orderbook for a single token from CLOB API."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{CLOB_BASE_URL}/book",
                params={"token_id": token_id},
            )
            response.raise_for_status()
            data = response.json()

            bids = []
            for level in data.get("bids", []):
                bids.append(OrderbookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                ))

            asks = []
            for level in data.get("asks", []):
                asks.append(OrderbookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                ))

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return Orderbook(bids=bids, asks=asks, timestamp=time.time())

        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id[:20]}...: {e}")
            return Orderbook(timestamp=time.time())

    # ─── Analytics ────────────────────────────────────────────────────────

    @property
    def avg_spread(self) -> float:
        """Average spread over recent history."""
        if not self._spread_history:
            return 0.02
        return sum(self._spread_history) / len(self._spread_history)

    @property
    def spread_percentile(self) -> float:
        """Current spread as percentile of recent history (0-100)."""
        if not self._spread_history or not self._current_snapshot:
            return 50.0
        current = self._current_snapshot.spread_up
        below = sum(1 for s in self._spread_history if s <= current)
        return (below / len(self._spread_history)) * 100.0

    @property
    def imbalance_trend(self) -> float:
        """
        Direction of imbalance change over recent ticks.
        Positive = imbalance increasing (more buyers appearing).
        """
        if len(self._imbalance_history) < 5:
            return 0.0
        recent = self._imbalance_history[-5:]
        older = self._imbalance_history[-10:-5] if len(self._imbalance_history) >= 10 else self._imbalance_history[:5]
        return (sum(recent) / len(recent)) - (sum(older) / len(older))

    def get_microstructure_signals(self) -> dict:
        """
        Extract trading signals from market microstructure.
        Used by the signal fusion engine.
        """
        if not self._current_snapshot:
            return {}

        snap = self._current_snapshot
        return {
            "spread_up": snap.spread_up,
            "spread_down": snap.spread_down,
            "imbalance_up": snap.book_imbalance_up,
            "imbalance_down": snap.book_imbalance_down,
            "bid_depth_up": snap.book_up.total_bid_depth,
            "ask_depth_up": snap.book_up.total_ask_depth,
            "bid_depth_down": snap.book_down.total_bid_depth,
            "ask_depth_down": snap.book_down.total_ask_depth,
            "weighted_mid_up": snap.book_up.weighted_mid,
            "weighted_mid_down": snap.book_down.weighted_mid,
            "avg_spread": self.avg_spread,
            "spread_percentile": self.spread_percentile,
            "imbalance_trend": self.imbalance_trend,
            "total_liquidity": snap.total_liquidity,
            "large_orders_up": len(snap.book_up.large_orders(100)),
            "large_orders_down": len(snap.book_down.large_orders(100)),
        }
