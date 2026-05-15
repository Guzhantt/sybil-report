"""
Price Feed - Derives BTC direction signals purely from Polymarket orderbook data.

NO external data source (no Binance, no Chainlink). Instead we infer everything
from the Polymarket BTC 5m market itself:

  - Up/Down price movements → implied BTC direction
  - Orderbook imbalance changes → buying/selling pressure
  - Spread dynamics → confidence/uncertainty
  - Price velocity → momentum
  - Volume-weighted mid → fair value estimate

Philosophy:
  The Polymarket Up token price IS the market's consensus probability that BTC
  will be higher at window end. If Up price moves from 0.50 → 0.60, the market
  is telling us BTC has moved up. We don't need Binance to confirm this.

  By tracking the RATE and PATTERN of these price changes, we can detect:
  - Momentum (price trending in one direction)
  - Mean reversion opportunities (overshooting)
  - Informed flow (large orders moving the book)
  - Stale pricing (when the book hasn't adjusted yet)
"""

import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PriceTick:
    """A single snapshot of Polymarket market prices."""
    timestamp: float           # Unix seconds
    up_price: float            # Mid price of Up token (0-1)
    down_price: float          # Mid price of Down token (0-1)
    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0
    spread_up: float = 0.0
    spread_down: float = 0.0
    bid_depth_up: float = 0.0  # Total USDC on bid
    ask_depth_up: float = 0.0  # Total USDC on ask
    imbalance_up: float = 0.0  # -1 to +1

    @property
    def implied_direction(self) -> float:
        """Implied BTC direction: >0.5 means market thinks BTC going up."""
        return self.up_price

    @property
    def displacement_from_fair(self) -> float:
        """How far Up price is from 0.50 (fair coin flip)."""
        return self.up_price - 0.50


@dataclass
class PriceFeedState:
    """
    Complete state derived purely from Polymarket orderbook observations.
    Replaces the old Binance-based PriceFeedState.
    """
    # Current market snapshot
    current_up_price: float = 0.5
    current_down_price: float = 0.5
    last_update_time: float = 0.0
    connected: bool = False

    # History of price ticks (sampled every ~1.5s)
    ticks: deque = field(default_factory=lambda: deque(maxlen=300))

    # Reconnect tracking (for compatibility)
    reconnect_count: int = 0

    # ─── Core Properties ──────────────────────────────────────────────

    @property
    def is_stale(self) -> bool:
        """Data considered stale if no update in 10 seconds."""
        return (time.time() - self.last_update_time) > 10.0

    @property
    def current_price(self) -> float:
        """
        'Price' in this context = Up token price.
        This serves as our BTC direction proxy.
        """
        return self.current_up_price

    @property
    def tick_count(self) -> int:
        return len(self.ticks)

    def get_recent_up_prices(self, n: int) -> list[float]:
        """Get last N Up token mid prices."""
        recent = list(self.ticks)[-n:]
        return [t.up_price for t in recent]

    # ─── Momentum ────────────────────────────────────────────────────

    def get_momentum(self, lookback_ticks: int = 10) -> Optional[float]:
        """
        Price momentum: how much Up price has moved over last N ticks.
        Returns change in probability units (e.g., 0.05 = Up price rose 5 cents).
        Positive = bullish momentum.
        """
        prices = self.get_recent_up_prices(lookback_ticks)
        if len(prices) < 3:
            return None
        return prices[-1] - prices[0]

    def get_momentum_pct(self, lookback_ticks: int = 10) -> Optional[float]:
        """Momentum as percentage change (for compatibility with old interface)."""
        mom = self.get_momentum(lookback_ticks)
        if mom is None:
            return None
        # Scale: 0.01 movement ≈ 1% equivalent signal strength
        return mom * 100.0

    def get_velocity(self, lookback_ticks: int = 5) -> Optional[float]:
        """
        Price velocity: rate of change per second.
        Positive = price accelerating upward.
        """
        ticks = list(self.ticks)[-lookback_ticks:]
        if len(ticks) < 3:
            return None
        dt = ticks[-1].timestamp - ticks[0].timestamp
        if dt <= 0:
            return None
        dp = ticks[-1].up_price - ticks[0].up_price
        return dp / dt

    def get_acceleration(self) -> Optional[float]:
        """
        Price acceleration: change in velocity.
        Positive = momentum increasing.
        """
        v_recent = self.get_velocity(3)
        v_older = self.get_velocity(8)
        if v_recent is None or v_older is None:
            return None
        return v_recent - v_older

    # ─── RSI (adapted for probability prices) ────────────────────────

    def get_rsi(self, period: int = 14) -> Optional[float]:
        """
        RSI computed on Up token price changes.
        Works the same as traditional RSI but on probability prices.
        """
        prices = self.get_recent_up_prices(period + 1)
        if len(prices) < period + 1:
            return None

        gains = []
        losses = []
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i - 1]
            gains.append(max(0, delta))
            losses.append(max(0, -delta))

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ─── Spread & Microstructure ─────────────────────────────────────

    def get_avg_spread(self, lookback: int = 20) -> float:
        """Average spread over recent ticks."""
        ticks = list(self.ticks)[-lookback:]
        if not ticks:
            return 0.02
        spreads = [t.spread_up for t in ticks if t.spread_up > 0]
        if not spreads:
            return 0.02
        return sum(spreads) / len(spreads)

    def get_spread_trend(self, lookback: int = 10) -> float:
        """
        Spread trend: widening (positive) or tightening (negative).
        Widening = uncertainty increasing.
        """
        ticks = list(self.ticks)[-lookback:]
        if len(ticks) < 5:
            return 0.0
        recent_spread = sum(t.spread_up for t in ticks[-3:]) / 3
        older_spread = sum(t.spread_up for t in ticks[:3]) / 3
        if older_spread == 0:
            return 0.0
        return (recent_spread - older_spread) / older_spread

    # ─── Imbalance ───────────────────────────────────────────────────

    def get_imbalance_trend(self, lookback: int = 10) -> float:
        """
        Direction of orderbook imbalance change.
        Positive = bids growing relative to asks (bullish pressure building).
        """
        ticks = list(self.ticks)[-lookback:]
        if len(ticks) < 5:
            return 0.0
        recent_imb = sum(t.imbalance_up for t in ticks[-3:]) / 3
        older_imb = sum(t.imbalance_up for t in ticks[:3]) / 3
        return recent_imb - older_imb

    def get_current_imbalance(self) -> float:
        """Current orderbook imbalance."""
        if not self.ticks:
            return 0.0
        return self.ticks[-1].imbalance_up

    # ─── Volatility ──────────────────────────────────────────────────

    def get_realized_volatility(self, lookback: int = 20) -> Optional[float]:
        """
        Realized volatility of Up token price.
        Returns as a percentage-like measure.
        """
        prices = self.get_recent_up_prices(lookback + 1)
        if len(prices) < 5:
            return None

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append(prices[i] - prices[i - 1])

        if len(returns) < 3:
            return None

        import numpy as np
        std = float(np.std(returns))
        # Scale to percentage-like (multiply by 100 and by sqrt of frequency)
        # Ticks are ~1.5s apart, so ~200 ticks per 5 min
        return std * math.sqrt(200) * 100.0

    def get_volatility_regime(self) -> VolatilityRegime:
        """Classify current volatility."""
        vol = self.get_realized_volatility(20)
        if vol is None:
            return VolatilityRegime.NORMAL
        # Calibrated for probability price movements
        if vol < 5.0:
            return VolatilityRegime.LOW
        elif vol > 15.0:
            return VolatilityRegime.HIGH
        return VolatilityRegime.NORMAL

    # ─── Volume / Flow Detection ─────────────────────────────────────

    def detect_large_move(self, threshold: float = 0.03) -> Optional[str]:
        """
        Detect if a large sudden move happened (potential informed trade).
        Returns 'UP', 'DOWN', or None.
        """
        if len(self.ticks) < 3:
            return None
        recent = list(self.ticks)[-3:]
        move = recent[-1].up_price - recent[0].up_price
        if move > threshold:
            return "UP"
        elif move < -threshold:
            return "DOWN"
        return None

    def get_depth_imbalance_velocity(self, lookback: int = 5) -> float:
        """
        How fast the depth imbalance is changing.
        Positive = bids filling in faster than asks (bullish flow).
        """
        ticks = list(self.ticks)[-lookback:]
        if len(ticks) < 3:
            return 0.0
        imbalances = [t.imbalance_up for t in ticks]
        # Simple slope
        if len(imbalances) < 2:
            return 0.0
        return (imbalances[-1] - imbalances[0]) / max(1, len(imbalances) - 1)

    # ─── Composite Signals ───────────────────────────────────────────

    def get_all_signals(self) -> dict:
        """
        Package all signals for the signal fusion engine.
        This is the main interface used by signals.py and the strategy.
        
        Replaces the old Binance-based signals with Polymarket-native ones.
        """
        mom_short = self.get_momentum(5)
        mom_mid = self.get_momentum(10)
        mom_long = self.get_momentum(20)

        return {
            # Momentum (map to old interface names)
            "momentum_1m": mom_short * 100 if mom_short else None,
            "momentum_3m": mom_mid * 100 if mom_mid else None,
            "momentum_5m": mom_long * 100 if mom_long else None,
            "momentum_accel": self.get_acceleration(),

            # Mean reversion
            "rsi_14": self.get_rsi(14),
            "vwap_deviation": self._get_price_vs_moving_avg(),

            # Volume/Flow (derived from orderbook dynamics)
            "obv_slope": self._get_imbalance_slope(),
            "large_trade_bias": self._get_large_move_bias(),
            "trade_flow_imbalance": self.get_current_imbalance(),
            "volume_surge": self._detect_volume_surge(),

            # Volatility
            "realized_vol": self.get_realized_volatility(20),
            "volatility_regime": self.get_volatility_regime().value,
            "intrabar_vol": self.get_realized_volatility(5),

            # Microstructure
            "trade_frequency": 0.0,  # Not applicable without trade stream
            "avg_trade_size": 0.0,

            # Current state
            "current_price": self.current_up_price,
            "vwap": self._get_moving_avg(20),
        }

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _get_moving_avg(self, lookback: int = 20) -> Optional[float]:
        """Simple moving average of Up price."""
        prices = self.get_recent_up_prices(lookback)
        if not prices:
            return None
        return sum(prices) / len(prices)

    def _get_price_vs_moving_avg(self) -> Optional[float]:
        """Current price deviation from moving average (like VWAP deviation)."""
        ma = self._get_moving_avg(20)
        if ma is None or ma == 0:
            return None
        return ((self.current_up_price - ma) / ma) * 100.0

    def _get_imbalance_slope(self) -> Optional[float]:
        """Slope of imbalance over time (like OBV slope)."""
        ticks = list(self.ticks)[-15:]
        if len(ticks) < 5:
            return None
        imbalances = [t.imbalance_up for t in ticks]
        # Linear slope normalized to [-1, 1]
        n = len(imbalances)
        x_mean = (n - 1) / 2.0
        y_mean = sum(imbalances) / n
        num = sum((i - x_mean) * (imbalances[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope = num / den
        return max(-1.0, min(1.0, slope * 10))  # Scale up

    def _get_large_move_bias(self) -> float:
        """Detect recent large moves and their direction."""
        move = self.detect_large_move(0.02)
        if move == "UP":
            return 0.7
        elif move == "DOWN":
            return -0.7
        return 0.0

    def _detect_volume_surge(self) -> bool:
        """
        Detect 'volume surge' equivalent: rapid price movement suggests
        large orders hitting the book.
        """
        if len(self.ticks) < 5:
            return False
        recent = list(self.ticks)[-5:]
        move = abs(recent[-1].up_price - recent[0].up_price)
        # If price moved > 3 cents in 5 ticks (~7.5s), that's a surge
        return move > 0.03

    # ─── Feed Updates ─────────────────────────────────────────────────

    def update(self, tick: PriceTick):
        """
        Called by the Trader/MarketFinder when new orderbook data arrives.
        This replaces the old WebSocket-based feed.
        """
        self.ticks.append(tick)
        self.current_up_price = tick.up_price
        self.current_down_price = tick.down_price
        self.last_update_time = tick.timestamp
        self.connected = True


# ─── PriceFeed Manager (simplified) ──────────────────────────────────────────

class PriceFeed:
    """
    Simplified PriceFeed that doesn't need any external WebSocket.
    Data is pushed into it by the Trader from MarketFinder orderbook refreshes.
    
    This is a thin wrapper around PriceFeedState for interface compatibility.
    """

    def __init__(self):
        self.state = PriceFeedState()

    @property
    def price(self) -> float:
        """Current Up token price (our BTC direction proxy)."""
        return self.state.current_up_price

    @property
    def connected(self) -> bool:
        return self.state.connected

    async def start(self):
        """No-op: data is pushed by the Trader."""
        logger.info("Price feed ready (Polymarket-native, no external source)")

    async def stop(self):
        """No-op."""
        self.state.connected = False
        logger.info("Price feed stopped")

    async def wait_for_price(self, timeout: float = 30.0) -> bool:
        """Wait until we receive at least one market data update."""
        start = time.time()
        while self.state.tick_count == 0:
            if time.time() - start > timeout:
                return False
            import asyncio
            await asyncio.sleep(0.2)
        return True

    def push_tick(self, tick: PriceTick):
        """Push a new price tick from the MarketFinder."""
        self.state.update(tick)
