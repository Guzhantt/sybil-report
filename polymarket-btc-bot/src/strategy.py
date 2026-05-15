"""
Strategy - Three-mode trading engine for BTC 5-minute binary markets.

PURE POLYMARKET: No external price feed. All decisions derived from:
  - Up/Down token price movements (= market's implied BTC direction)
  - Orderbook imbalance and depth
  - Price velocity and acceleration
  - Time remaining in window

Modes:
  1. MARKET_MAKING (10-180s): Provide liquidity on both sides, earn spread
  2. MOMENTUM (60-240s): Detect strong directional conviction, take bets
  3. ENDGAME_ARB (240-285s): Exploit mispriced markets near expiry

Key insight:
  Up token price IS the market's consensus P(BTC finishes higher).
  If Up moves from 0.50 → 0.65 with 30s left, BTC almost certainly moved up
  and will likely stay there. If the price hasn't caught up fully, that's our edge.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class StrategyMode(Enum):
    MARKET_MAKING = "MARKET_MAKING"
    MOMENTUM = "MOMENTUM"
    ENDGAME_ARB = "ENDGAME_ARB"
    WAIT = "WAIT"


class ActionType(Enum):
    PLACE_BID = "PLACE_BID"
    PLACE_ASK = "PLACE_ASK"
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    CANCEL_ALL = "CANCEL_ALL"
    CANCEL_SIDE = "CANCEL_SIDE"
    HOLD = "HOLD"


class TokenSide(Enum):
    UP = "UP"
    DOWN = "DOWN"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Action:
    """A single action the strategy wants to execute."""
    action_type: ActionType
    token_side: TokenSide = TokenSide.UP
    price: float = 0.0
    size: float = 0.0
    urgency: float = 0.0  # 0-1, how quickly this should execute
    reason: str = ""
    mode: StrategyMode = StrategyMode.WAIT

    @property
    def is_passive(self) -> bool:
        return self.action_type in (ActionType.PLACE_BID, ActionType.PLACE_ASK)


@dataclass
class MarketState:
    """
    Snapshot of current market conditions passed to strategy.
    ALL data comes from Polymarket orderbook - no external BTC price needed.
    """
    # Up/Down token prices (mid)
    up_price: float = 0.5
    down_price: float = 0.5

    # Orderbook
    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0
    spread_up: float = 0.0
    spread_down: float = 0.0

    # Orderbook imbalance (-1 to +1, positive = more bids = bullish)
    book_imbalance_up: float = 0.0
    book_imbalance_down: float = 0.0
    bid_depth_up: float = 0.0
    ask_depth_up: float = 0.0

    # Timing
    seconds_elapsed: float = 0.0
    seconds_remaining: float = 300.0
    progress_pct: float = 0.0  # 0-100

    # Displacement: how far Up price has moved from 0.50 (window start fair value)
    # This REPLACES the old btc_displacement_pct
    up_displacement: float = 0.0  # up_price - 0.50, positive = bullish

    # Signals from signal engine (derived from Polymarket data)
    composite_signal: float = 0.0   # -1 (strong down) to +1 (strong up)
    signal_confidence: float = 0.0  # 0-1

    # Additional context
    momentum_score: float = 0.0
    volatility_regime: str = "normal"
    volume_surge: bool = False
    spread_trend: float = 0.0     # Widening or tightening
    imbalance_velocity: float = 0.0  # How fast imbalance is changing


@dataclass
class StrategyConfig:
    """Configuration for the strategy engine."""
    # Market Making
    mm_spread: float = 0.03
    mm_size: float = 10.0
    mm_max_inventory: float = 30.0
    mm_skew_factor: float = 0.5

    # Momentum
    momentum_threshold: float = 0.6
    momentum_size: float = 15.0
    momentum_max_size: float = 40.0

    # Endgame Arbitrage
    arb_min_displacement: float = 0.08  # Min Up price deviation from 0.50
    arb_prob_threshold: float = 0.08    # Min edge vs market price
    arb_size: float = 25.0
    arb_max_size: float = 60.0

    # Timing thresholds (seconds from start)
    warmup_period: float = 10.0
    mm_start: float = 10.0
    mm_end: float = 180.0
    momentum_start: float = 60.0
    momentum_end: float = 240.0
    endgame_start: float = 240.0
    endgame_cutoff: float = 285.0

    # Risk
    edge_threshold: float = 0.04
    max_loss_per_window: float = 20.0


# ─── Strategy Engine ──────────────────────────────────────────────────────────

class Strategy:
    """
    Three-mode strategy engine using ONLY Polymarket orderbook data.

    Timeline:
      0-10s:   WAIT (let market establish)
      10-180s: MARKET_MAKING (earn spread, gather information)
      60-240s: MOMENTUM (directional bets on strong signal convergence)
      240-285s: ENDGAME_ARB (exploit stale pricing near expiry)
      285-300s: WAIT (too close to expiry)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._current_mode = StrategyMode.WAIT
        self._window_pnl: float = 0.0
        self._net_inventory: float = 0.0
        self._mm_fills: int = 0
        self._last_actions: list = []
        self._window_start_up_price: float = 0.5  # Track where Up was at window start

    @property
    def current_mode(self) -> StrategyMode:
        return self._current_mode

    def reset_window(self):
        """Reset state for a new 5-minute window."""
        self._window_pnl = 0.0
        self._net_inventory = 0.0
        self._mm_fills = 0
        self._current_mode = StrategyMode.WAIT
        self._last_actions = []
        self._window_start_up_price = 0.5

    def set_window_start_price(self, up_price: float):
        """Record the Up token price at the start of the window."""
        self._window_start_up_price = up_price

    def evaluate(self, state: MarketState) -> list:
        """
        Main evaluation function. Called every tick (~1.5 seconds).
        Returns a list of Action objects to execute.
        """
        actions = []

        # Stop conditions
        if self._should_stop(state):
            actions.append(Action(
                action_type=ActionType.CANCEL_ALL,
                reason="Window stop condition",
                mode=StrategyMode.WAIT,
            ))
            return actions

        # Determine active modes
        active_modes = self._get_active_modes(state)

        if not active_modes:
            self._current_mode = StrategyMode.WAIT
            return [Action(action_type=ActionType.HOLD, reason="Waiting", mode=StrategyMode.WAIT)]

        # Evaluate each active mode
        for mode in active_modes:
            if mode == StrategyMode.MARKET_MAKING:
                actions.extend(self._evaluate_market_making(state))
            elif mode == StrategyMode.MOMENTUM:
                actions.extend(self._evaluate_momentum(state))
            elif mode == StrategyMode.ENDGAME_ARB:
                actions.extend(self._evaluate_endgame(state))

        if actions:
            self._current_mode = actions[0].mode

        self._last_actions = actions
        return actions if actions else [Action(action_type=ActionType.HOLD, mode=StrategyMode.WAIT)]

    # ─── Mode Selection ───────────────────────────────────────────────

    def _get_active_modes(self, state: MarketState) -> list:
        elapsed = state.seconds_elapsed
        cfg = self.config
        modes = []

        if elapsed < cfg.warmup_period:
            return []
        if elapsed >= cfg.endgame_cutoff:
            return []

        if cfg.mm_start <= elapsed <= cfg.mm_end:
            modes.append(StrategyMode.MARKET_MAKING)
        if cfg.momentum_start <= elapsed <= cfg.momentum_end:
            modes.append(StrategyMode.MOMENTUM)
        if elapsed >= cfg.endgame_start:
            modes = [StrategyMode.ENDGAME_ARB]

        return modes

    def _should_stop(self, state: MarketState) -> bool:
        if state.seconds_remaining < 10:
            return True
        if self._window_pnl < -self.config.max_loss_per_window:
            return True
        return False

    # ─── Market Making ────────────────────────────────────────────────

    def _evaluate_market_making(self, state: MarketState) -> list:
        """
        Market Making using ONLY Polymarket data.
        Quote around the current mid, skewed by signal and inventory.
        """
        cfg = self.config
        actions = []

        # Adjust spread based on volatility
        base_spread = cfg.mm_spread
        if state.volatility_regime == "high":
            base_spread *= 1.5
        elif state.volatility_regime == "low":
            base_spread *= 0.7

        # Fair value = current Up mid, adjusted by composite signal
        fair_value_up = state.up_price + (state.composite_signal * 0.005)
        fair_value_up = max(0.05, min(0.95, fair_value_up))

        # Inventory skew
        inventory_skew = (self._net_inventory / cfg.mm_max_inventory) * cfg.mm_skew_factor * base_spread

        # Calculate bid/ask for Up token
        bid_up = fair_value_up - base_spread - inventory_skew
        ask_up = fair_value_up + base_spread - inventory_skew

        # Size adjustment
        bid_size = cfg.mm_size
        ask_size = cfg.mm_size

        if self._net_inventory > cfg.mm_max_inventory * 0.7:
            bid_size *= 0.3
            ask_size *= 1.5
        elif self._net_inventory < -cfg.mm_max_inventory * 0.7:
            bid_size *= 1.5
            ask_size *= 0.3

        bid_up = max(0.01, min(0.98, round(bid_up, 2)))
        ask_up = max(0.02, min(0.99, round(ask_up, 2)))

        if ask_up - bid_up >= 0.02:
            actions.append(Action(
                action_type=ActionType.PLACE_BID, token_side=TokenSide.UP,
                price=bid_up, size=round(bid_size, 2),
                reason=f"MM bid Up@{bid_up:.2f} fair={fair_value_up:.3f} inv={self._net_inventory:.1f}",
                mode=StrategyMode.MARKET_MAKING,
            ))
            actions.append(Action(
                action_type=ActionType.PLACE_ASK, token_side=TokenSide.UP,
                price=ask_up, size=round(ask_size, 2),
                reason=f"MM ask Up@{ask_up:.2f}",
                mode=StrategyMode.MARKET_MAKING,
            ))

        # Mirror on Down token
        fair_value_down = 1.0 - fair_value_up
        bid_down = fair_value_down - base_spread + inventory_skew
        ask_down = fair_value_down + base_spread + inventory_skew
        bid_down = max(0.01, min(0.98, round(bid_down, 2)))
        ask_down = max(0.02, min(0.99, round(ask_down, 2)))

        if ask_down - bid_down >= 0.02:
            actions.append(Action(
                action_type=ActionType.PLACE_BID, token_side=TokenSide.DOWN,
                price=bid_down, size=round(bid_size, 2),
                reason=f"MM bid Down@{bid_down:.2f}",
                mode=StrategyMode.MARKET_MAKING,
            ))
            actions.append(Action(
                action_type=ActionType.PLACE_ASK, token_side=TokenSide.DOWN,
                price=ask_down, size=round(ask_size, 2),
                reason=f"MM ask Down@{ask_down:.2f}",
                mode=StrategyMode.MARKET_MAKING,
            ))

        return actions

    # ─── Momentum ─────────────────────────────────────────────────────

    def _evaluate_momentum(self, state: MarketState) -> list:
        """
        Momentum: when composite signal is strong AND orderbook confirms,
        take a directional position.
        
        Trigger conditions (ALL must be met):
        1. Composite signal > threshold
        2. Up price has moved in the signal direction (confirmation)
        3. Orderbook imbalance supports the move
        4. Not in high-vol chop
        """
        cfg = self.config
        signal = state.composite_signal
        confidence = state.signal_confidence

        if state.volatility_regime == "high" and confidence < 0.7:
            return []

        if abs(signal) < cfg.momentum_threshold:
            return []

        # Confirm with price movement direction
        displacement = state.up_displacement
        if signal > 0 and displacement < 0.005:
            return []  # Signal says up but price hasn't moved
        if signal < 0 and displacement > -0.005:
            return []

        # Confirm with orderbook
        if signal > 0 and state.book_imbalance_up < -0.3:
            return []
        if signal < 0 and state.book_imbalance_up > 0.3:
            return []

        # Calculate edge: our estimated fair value vs current market
        estimated_fair = self._estimate_fair_value(state)
        
        if signal > 0:
            edge = estimated_fair - state.up_price
            if edge < cfg.edge_threshold:
                return []
            size = self._calc_momentum_size(edge, confidence)
            price = min(state.best_ask_up, state.up_price + 0.01)
            price = max(0.01, min(0.99, round(price, 2)))
            return [Action(
                action_type=ActionType.MARKET_BUY, token_side=TokenSide.UP,
                price=price, size=size, urgency=min(1.0, abs(signal)),
                reason=f"Momentum UP: sig={signal:.2f} edge={edge:.3f} conf={confidence:.2f}",
                mode=StrategyMode.MOMENTUM,
            )]
        else:
            edge = (1.0 - estimated_fair) - state.down_price
            if edge < cfg.edge_threshold:
                return []
            size = self._calc_momentum_size(edge, confidence)
            price = min(state.best_ask_down, state.down_price + 0.01)
            price = max(0.01, min(0.99, round(price, 2)))
            return [Action(
                action_type=ActionType.MARKET_BUY, token_side=TokenSide.DOWN,
                price=price, size=size, urgency=min(1.0, abs(signal)),
                reason=f"Momentum DOWN: sig={signal:.2f} edge={edge:.3f} conf={confidence:.2f}",
                mode=StrategyMode.MOMENTUM,
            )]

    def _calc_momentum_size(self, edge: float, confidence: float) -> float:
        cfg = self.config
        edge_mult = min(3.0, edge / cfg.edge_threshold)
        size = cfg.momentum_size * edge_mult * confidence
        return round(max(5.0, min(cfg.momentum_max_size, size)), 2)

    # ─── Endgame Arbitrage ────────────────────────────────────────────

    def _evaluate_endgame(self, state: MarketState) -> list:
        """
        Endgame Arbitrage: Near expiry, if Up/Down price has moved significantly
        from 0.50, the outcome is increasingly certain.
        
        Key insight: With 30s left and Up price at 0.65, BTC is almost certainly
        above the open price. The theoretical fair value should be much higher
        (approaching 0.85-0.95). If market is still at 0.65, that's free money.
        
        We use the TIME-ADJUSTED probability: as time → 0, if price is still
        displaced, certainty → 1.0.
        """
        cfg = self.config
        displacement = state.up_displacement  # How far from 0.50
        seconds_left = state.seconds_remaining

        # Need meaningful displacement
        if abs(displacement) < cfg.arb_min_displacement:
            return []

        # Estimate "true" probability given displacement and time remaining
        # With little time left, current displacement is very informative
        true_prob = self._estimate_endgame_probability(displacement, seconds_left)

        # Calculate edge
        if displacement > 0:
            # Market thinks Up, check if Up token is underpriced
            edge = true_prob - state.up_price
            if edge < cfg.arb_prob_threshold:
                return []

            size = self._calc_endgame_size(edge, seconds_left)
            price = min(state.best_ask_up + 0.01, true_prob - 0.02)
            price = max(0.01, min(0.99, round(price, 2)))

            logger.info(
                f"ENDGAME: Up disp={displacement:+.3f}, P_true={true_prob:.3f}, "
                f"mkt={state.up_price:.3f}, edge={edge:.3f}, {seconds_left:.0f}s left"
            )
            return [Action(
                action_type=ActionType.MARKET_BUY, token_side=TokenSide.UP,
                price=price, size=size, urgency=0.9,
                reason=f"Endgame Up: P={true_prob:.3f} vs mkt={state.up_price:.3f} edge={edge:.3f}",
                mode=StrategyMode.ENDGAME_ARB,
            )]
        else:
            # Market thinks Down
            true_prob_down = 1.0 - true_prob
            edge = true_prob_down - state.down_price
            if edge < cfg.arb_prob_threshold:
                return []

            size = self._calc_endgame_size(edge, seconds_left)
            price = min(state.best_ask_down + 0.01, true_prob_down - 0.02)
            price = max(0.01, min(0.99, round(price, 2)))

            logger.info(
                f"ENDGAME: Down disp={displacement:+.3f}, P_true_down={true_prob_down:.3f}, "
                f"mkt={state.down_price:.3f}, edge={edge:.3f}, {seconds_left:.0f}s left"
            )
            return [Action(
                action_type=ActionType.MARKET_BUY, token_side=TokenSide.DOWN,
                price=price, size=size, urgency=0.9,
                reason=f"Endgame Down: P={true_prob_down:.3f} vs mkt={state.down_price:.3f} edge={edge:.3f}",
                mode=StrategyMode.ENDGAME_ARB,
            )]

    def _estimate_endgame_probability(self, displacement: float, seconds_left: float) -> float:
        """
        Estimate true P(Up) given current Up token displacement and time remaining.
        
        Model: The further Up price is from 0.50, AND the less time remains,
        the more certain the outcome. We model this as:
        
        P(Up) = Φ(displacement * time_sharpening_factor)
        
        Where time_sharpening makes displacement more decisive as time → 0.
        """
        # Time sharpening: as seconds_left → 0, small displacements become conclusive
        # At 60s: moderate sharpening. At 10s: extreme sharpening.
        time_factor = max(1.0, 60.0 / max(5.0, seconds_left))
        
        # Scale displacement (0.10 = 10 cent move should be very decisive with little time)
        scaled = displacement * time_factor * 8.0  # Calibration factor
        
        # Sigmoid to get probability
        prob = 0.5 + 0.5 * math.tanh(scaled)
        
        # Additional certainty boost for extreme displacements
        if abs(displacement) > 0.15:
            # If Up price is > 0.65 or < 0.35, it's very likely correct
            certainty_boost = (abs(displacement) - 0.15) * 0.5
            if displacement > 0:
                prob = min(0.99, prob + certainty_boost)
            else:
                prob = max(0.01, prob - certainty_boost)

        return max(0.02, min(0.98, prob))

    def _calc_endgame_size(self, edge: float, seconds_left: float) -> float:
        cfg = self.config
        edge_mult = min(3.0, edge / cfg.arb_prob_threshold)
        # More aggressive as time runs out (less risk of reversal)
        time_mult = 0.5 + 0.5 * (1.0 - seconds_left / 60.0)
        time_mult = max(0.5, min(1.5, time_mult))
        size = cfg.arb_size * edge_mult * time_mult
        return round(max(5.0, min(cfg.arb_max_size, size)), 2)

    # ─── Fair Value Estimation ────────────────────────────────────────

    def _estimate_fair_value(self, state: MarketState) -> float:
        """
        Estimate the fair value of the Up token based on all available signals.
        This is used by Momentum mode to detect when market is mispriced.
        
        Combines:
        - Current Up price (base)
        - Composite signal direction
        - Orderbook imbalance
        - Spread dynamics
        """
        base = state.up_price
        
        # Signal-based adjustment
        signal_adj = state.composite_signal * 0.03 * state.signal_confidence
        
        # Imbalance adjustment (strong imbalance suggests price should move)
        imb_adj = state.book_imbalance_up * 0.015
        
        # If spread is tightening while imbalance grows, stronger signal
        if state.spread_trend < 0 and abs(state.book_imbalance_up) > 0.3:
            imb_adj *= 1.5
        
        fair = base + signal_adj + imb_adj
        return max(0.02, min(0.98, fair))

    # ─── Inventory Management ─────────────────────────────────────────

    def update_inventory(self, token_side: TokenSide, size: float, is_buy: bool):
        """Update net inventory after a fill."""
        delta = size if is_buy else -size
        if token_side == TokenSide.UP:
            self._net_inventory += delta
        else:
            self._net_inventory -= delta
        self._mm_fills += 1

    def update_pnl(self, realized_pnl: float):
        self._window_pnl += realized_pnl

    def get_state(self) -> dict:
        return {
            "mode": self._current_mode.value,
            "net_inventory": round(self._net_inventory, 2),
            "window_pnl": round(self._window_pnl, 2),
            "mm_fills": self._mm_fills,
        }
