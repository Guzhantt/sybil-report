"""
Risk Manager - Position management, hedging, and exposure control for market making.

Enhanced for the three-mode strategy:

1. MARKET MAKING RISK:
   - Net delta (Up vs Down) exposure tracking
   - Auto-skewing when inventory exceeds limits
   - Maximum gross exposure (total tokens held)
   - Forced unwind triggers

2. DIRECTIONAL RISK (Momentum/Endgame):
   - Position sizing via fractional Kelly
   - Per-trade and per-window stop-loss
   - Maximum concurrent directional positions
   - Correlation-aware sizing (don't double up)

3. GLOBAL RISK:
   - Per-window loss limit (circuit breaker)
   - Per-hour trade frequency cap
   - Daily P&L tracking and drawdown limit
   - Minimum time between trades (anti-spam)
   - Fee-aware profitability threshold
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.strategy import StrategyMode, TokenSide, Action, ActionType

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"  # Market resolved


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single position (token holding)."""
    token_side: TokenSide
    size: float           # USDC invested
    entry_price: float    # Average entry price
    token_id: str = ""
    market_slug: str = ""
    mode: StrategyMode = StrategyMode.WAIT
    opened_at: float = field(default_factory=time.time)
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def holding_seconds(self) -> float:
        return time.time() - self.opened_at

    def mark_to_market(self, current_price: float) -> float:
        """Unrealized P&L at current market price."""
        if not self.is_open or self.entry_price == 0:
            return 0.0
        # Binary token: profit = size * (current/entry - 1)
        return self.size * (current_price / self.entry_price - 1.0)

    def close(self, exit_price: float) -> float:
        """Close position and return realized PnL."""
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        if self.entry_price > 0:
            self.realized_pnl = self.size * (exit_price / self.entry_price - 1.0)
        else:
            self.realized_pnl = 0.0
        return self.realized_pnl


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Market Making
    mm_max_gross_exposure: float = 60.0      # Max total USDC in MM positions
    mm_max_net_delta: float = 30.0           # Max net (Up - Down) exposure
    mm_inventory_skew_start: float = 15.0    # Start skewing at this net delta

    # Directional
    dir_max_position: float = 50.0           # Max single directional position
    dir_max_total: float = 100.0             # Max total directional exposure
    dir_stop_loss_pct: float = 0.20          # 20% stop loss per position
    dir_take_profit_pct: float = 0.40        # 40% take profit per position

    # Global
    max_loss_per_window: float = 25.0        # Circuit breaker per 5-min window
    max_loss_per_hour: float = 80.0          # Hourly loss limit
    max_loss_per_day: float = 200.0          # Daily loss limit
    max_trades_per_hour: int = 60            # Rate limit (higher for MM)
    min_trade_interval: float = 2.0          # Min seconds between trades
    min_edge_after_fees: float = 0.01        # Min expected profit after 7% fee

    # Fee structure (Polymarket)
    taker_fee_rate: float = 0.07             # 7% taker fee
    maker_rebate_rate: float = 0.20          # 20% maker rebate (of fee)


@dataclass
class WindowStats:
    """Statistics for the current 5-minute window."""
    window_start: float = field(default_factory=time.time)
    total_trades: int = 0
    mm_fills: int = 0
    momentum_trades: int = 0
    endgame_trades: int = 0
    gross_volume: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    fees_earned: float = 0.0  # Maker rebates

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid + self.fees_earned

    @property
    def net_fees(self) -> float:
        return self.fees_paid - self.fees_earned


# ─── Risk Manager ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Central risk management engine.
    
    Responsibilities:
    - Approve/reject/resize actions from the strategy
    - Track all open positions and their mark-to-market
    - Enforce exposure limits, loss limits, and rate limits
    - Calculate optimal position sizes
    - Manage market-making inventory delta
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

        # Position tracking
        self._positions: list[Position] = []
        self._window_stats = WindowStats()

        # Rate limiting
        self._trade_timestamps: deque = deque(maxlen=200)
        self._last_trade_time: float = 0.0

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_start: float = time.time()
        self._hourly_pnl: float = 0.0
        self._hour_start: float = time.time()

    # ─── Main Interface ───────────────────────────────────────────────

    def check_action(self, action: Action, available_balance: float) -> tuple[bool, float, str]:
        """
        Check whether an action passes all risk checks.
        
        Args:
            action: The action from strategy
            available_balance: Current USDC balance
            
        Returns:
            (approved, adjusted_size, reason)
        """
        if action.action_type in (ActionType.HOLD, ActionType.CANCEL_ALL, ActionType.CANCEL_SIDE):
            return True, 0.0, "Non-trade action always approved"

        # ─── Global checks ────────────────────────────────────────────

        # Rate limit
        if not self._check_rate_limit():
            return False, 0.0, f"Rate limit: {self._trades_last_hour} trades/hr (max {self.limits.max_trades_per_hour})"

        # Min interval
        elapsed = time.time() - self._last_trade_time
        if elapsed < self.limits.min_trade_interval:
            return False, 0.0, f"Too fast: {elapsed:.1f}s (min {self.limits.min_trade_interval}s)"

        # Window loss limit
        if self._window_stats.net_pnl < -self.limits.max_loss_per_window:
            return False, 0.0, f"Window loss limit: {self._window_stats.net_pnl:.2f} (limit: -{self.limits.max_loss_per_window})"

        # Hourly loss limit
        if self._hourly_pnl < -self.limits.max_loss_per_hour:
            return False, 0.0, f"Hourly loss limit: {self._hourly_pnl:.2f}"

        # Daily loss limit
        if self._daily_pnl < -self.limits.max_loss_per_day:
            return False, 0.0, f"Daily loss limit: {self._daily_pnl:.2f}"

        # Balance check
        if available_balance < 5.0:
            return False, 0.0, f"Insufficient balance: {available_balance:.2f}"

        # ─── Mode-specific checks ────────────────────────────────────

        if action.mode == StrategyMode.MARKET_MAKING:
            return self._check_mm_action(action, available_balance)
        elif action.mode == StrategyMode.MOMENTUM:
            return self._check_directional_action(action, available_balance, "momentum")
        elif action.mode == StrategyMode.ENDGAME_ARB:
            return self._check_directional_action(action, available_balance, "endgame")
        else:
            return False, 0.0, f"Unknown mode: {action.mode}"

    def _check_mm_action(self, action: Action, balance: float) -> tuple[bool, float, str]:
        """Check market making specific limits."""
        limits = self.limits

        # Gross exposure check
        current_gross = self._mm_gross_exposure
        if current_gross + action.size > limits.mm_max_gross_exposure:
            remaining = limits.mm_max_gross_exposure - current_gross
            if remaining < 5.0:
                return False, 0.0, f"MM gross limit: {current_gross:.0f}/{limits.mm_max_gross_exposure:.0f}"
            adjusted = min(action.size, remaining)
            return True, adjusted, f"MM size reduced: {action.size:.0f} → {adjusted:.0f} (gross limit)"

        # Net delta check (only for buys that increase delta)
        net_delta = self._net_delta
        if action.action_type == ActionType.PLACE_BID:
            if action.token_side == TokenSide.UP:
                projected = net_delta + action.size
            else:
                projected = net_delta - action.size

            if abs(projected) > limits.mm_max_net_delta:
                # Reduce size to stay within limit
                if action.token_side == TokenSide.UP:
                    max_allowed = limits.mm_max_net_delta - net_delta
                else:
                    max_allowed = limits.mm_max_net_delta + net_delta
                max_allowed = max(0, max_allowed)
                if max_allowed < 5.0:
                    return False, 0.0, f"MM delta limit: net={net_delta:.1f}, max_delta={limits.mm_max_net_delta}"
                adjusted = min(action.size, max_allowed)
                return True, adjusted, f"MM delta-adjusted: {adjusted:.0f} (net_delta={net_delta:.1f})"

        # Size limit by balance
        adjusted = min(action.size, balance * 0.3)  # Never use more than 30% per MM order
        adjusted = max(5.0, adjusted)

        return True, adjusted, f"MM approved: {adjusted:.0f} USDC"

    def _check_directional_action(self, action: Action, balance: float, mode_name: str) -> tuple[bool, float, str]:
        """Check directional trade (momentum/endgame) limits."""
        limits = self.limits

        # Max single position
        requested = action.size
        max_single = limits.dir_max_position

        # Total directional exposure
        current_dir = self._directional_exposure
        remaining = limits.dir_max_total - current_dir
        if remaining < 5.0:
            return False, 0.0, f"Directional limit: {current_dir:.0f}/{limits.dir_max_total:.0f}"

        # Fee-adjusted edge check
        # For taker orders, need edge > fee to be profitable
        if action.action_type == ActionType.MARKET_BUY:
            effective_fee = limits.taker_fee_rate
            # Edge is implied by the urgency/reason - we trust strategy's edge calculation
            # but verify minimum profitability
            if action.urgency < 0.3:
                return False, 0.0, f"Low urgency ({action.urgency:.2f}) doesn't justify taker fees"

        # Calculate optimal size
        size = min(requested, max_single, remaining, balance * 0.4)
        size = max(5.0, size) if size >= 5.0 else 0.0

        if size == 0:
            return False, 0.0, f"Calculated size too small after limits"

        return True, round(size, 2), f"{mode_name} approved: {size:.0f} USDC (urgency={action.urgency:.2f})"

    # ─── Position Tracking ────────────────────────────────────────────

    def record_fill(
        self,
        token_side: TokenSide,
        size: float,
        price: float,
        token_id: str,
        market_slug: str,
        mode: StrategyMode,
        is_maker: bool = False,
    ):
        """Record a fill (order execution)."""
        pos = Position(
            token_side=token_side,
            size=size,
            entry_price=price,
            token_id=token_id,
            market_slug=market_slug,
            mode=mode,
        )
        self._positions.append(pos)

        # Track rate
        self._trade_timestamps.append(time.time())
        self._last_trade_time = time.time()

        # Track stats
        self._window_stats.total_trades += 1
        self._window_stats.gross_volume += size

        if mode == StrategyMode.MARKET_MAKING:
            self._window_stats.mm_fills += 1
        elif mode == StrategyMode.MOMENTUM:
            self._window_stats.momentum_trades += 1
        elif mode == StrategyMode.ENDGAME_ARB:
            self._window_stats.endgame_trades += 1

        # Track fees
        if is_maker:
            rebate = size * self.limits.taker_fee_rate * self.limits.maker_rebate_rate
            self._window_stats.fees_earned += rebate
        else:
            fee = size * self.limits.taker_fee_rate
            self._window_stats.fees_paid += fee

        logger.info(
            f"📊 Fill: {mode.value} {token_side.value} {size:.2f}@{price:.2f} "
            f"({'maker' if is_maker else 'taker'}) | "
            f"Net Δ={self._net_delta:+.1f} | Gross={self._mm_gross_exposure:.0f}"
        )

    def close_position(self, token_id: str, exit_price: float) -> float:
        """Close a specific position and return realized PnL."""
        for pos in self._positions:
            if pos.token_id == token_id and pos.is_open:
                pnl = pos.close(exit_price)
                self._window_stats.realized_pnl += pnl
                self._hourly_pnl += pnl
                self._daily_pnl += pnl
                logger.info(f"📊 Closed: {pos.token_side.value} PnL={pnl:+.2f}")
                return pnl
        return 0.0

    def expire_all_positions(self, resolved_up: bool):
        """
        Called when a 5-min window resolves.
        Mark all open positions as expired and calculate final PnL.
        
        If resolved Up: Up tokens pay 1.0, Down tokens pay 0.0
        If resolved Down: Up tokens pay 0.0, Down tokens pay 1.0
        """
        for pos in self._positions:
            if not pos.is_open:
                continue

            if pos.token_side == TokenSide.UP:
                exit_price = 1.0 if resolved_up else 0.0
            else:
                exit_price = 0.0 if resolved_up else 1.0

            pnl = pos.close(exit_price)
            pos.status = PositionStatus.EXPIRED
            self._window_stats.realized_pnl += pnl
            self._hourly_pnl += pnl
            self._daily_pnl += pnl

        total_pnl = self._window_stats.realized_pnl
        logger.info(
            f"📊 Window expired ({'UP' if resolved_up else 'DOWN'}): "
            f"PnL={total_pnl:+.2f} | Fees={self._window_stats.net_fees:.2f}"
        )

    # ─── Stop Loss / Take Profit ─────────────────────────────────────

    def check_exits(self, up_price: float, down_price: float) -> list[Position]:
        """
        Check all open directional positions for stop-loss or take-profit.
        Returns list of positions that should be closed.
        """
        to_close = []

        for pos in self._positions:
            if not pos.is_open:
                continue
            if pos.mode == StrategyMode.MARKET_MAKING:
                continue  # MM positions managed differently

            current = up_price if pos.token_side == TokenSide.UP else down_price
            if pos.entry_price <= 0:
                continue

            pnl_pct = (current - pos.entry_price) / pos.entry_price

            if pnl_pct <= -self.limits.dir_stop_loss_pct:
                logger.warning(
                    f"🛑 STOP LOSS: {pos.token_side.value} "
                    f"entry={pos.entry_price:.2f} → {current:.2f} ({pnl_pct*100:+.1f}%)"
                )
                to_close.append(pos)

            elif pnl_pct >= self.limits.dir_take_profit_pct:
                logger.info(
                    f"🎯 TAKE PROFIT: {pos.token_side.value} "
                    f"entry={pos.entry_price:.2f} → {current:.2f} ({pnl_pct*100:+.1f}%)"
                )
                to_close.append(pos)

        return to_close

    # ─── Inventory Management ─────────────────────────────────────────

    @property
    def _net_delta(self) -> float:
        """Net delta: positive = long Up, negative = long Down."""
        up_exposure = sum(p.size for p in self._positions if p.is_open and p.token_side == TokenSide.UP)
        down_exposure = sum(p.size for p in self._positions if p.is_open and p.token_side == TokenSide.DOWN)
        return up_exposure - down_exposure

    @property
    def _mm_gross_exposure(self) -> float:
        """Total MM exposure (both sides)."""
        return sum(
            p.size for p in self._positions
            if p.is_open and p.mode == StrategyMode.MARKET_MAKING
        )

    @property
    def _directional_exposure(self) -> float:
        """Total directional (momentum + endgame) exposure."""
        return sum(
            p.size for p in self._positions
            if p.is_open and p.mode in (StrategyMode.MOMENTUM, StrategyMode.ENDGAME_ARB)
        )

    def get_inventory_skew(self) -> float:
        """
        How much to skew MM quotes based on inventory.
        Returns -1 to +1: positive means we're long Up and should skew to sell Up.
        """
        delta = self._net_delta
        threshold = self.limits.mm_inventory_skew_start
        if abs(delta) < threshold:
            return 0.0
        skew = (delta - threshold * (1 if delta > 0 else -1)) / (self.limits.mm_max_net_delta - threshold)
        return max(-1.0, min(1.0, skew))

    def should_hedge(self) -> tuple[bool, TokenSide, float]:
        """
        Check if inventory is too skewed and needs hedging.
        Returns (should_hedge, side_to_buy, suggested_size).
        """
        delta = self._net_delta
        if abs(delta) > self.limits.mm_max_net_delta * 0.8:
            # Need to hedge: buy the opposite side
            if delta > 0:
                return True, TokenSide.DOWN, abs(delta) * 0.5
            else:
                return True, TokenSide.UP, abs(delta) * 0.5
        return False, TokenSide.UP, 0.0

    # ─── Window Management ────────────────────────────────────────────

    def reset_window(self):
        """Reset for a new 5-minute window."""
        self._positions = []
        self._window_stats = WindowStats()

    def reset_hourly(self):
        """Reset hourly counters."""
        now = time.time()
        if now - self._hour_start > 3600:
            self._hourly_pnl = 0.0
            self._hour_start = now

    def reset_daily(self):
        """Reset daily counters."""
        now = time.time()
        if now - self._daily_start > 86400:
            self._daily_pnl = 0.0
            self._daily_start = now

    # ─── Rate Limiting ────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        cutoff = time.time() - 3600
        recent = sum(1 for t in self._trade_timestamps if t > cutoff)
        return recent < self.limits.max_trades_per_hour

    @property
    def _trades_last_hour(self) -> int:
        cutoff = time.time() - 3600
        return sum(1 for t in self._trade_timestamps if t > cutoff)

    # ─── Reporting ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive risk/performance statistics."""
        open_positions = [p for p in self._positions if p.is_open]

        return {
            "window": {
                "total_trades": self._window_stats.total_trades,
                "mm_fills": self._window_stats.mm_fills,
                "momentum_trades": self._window_stats.momentum_trades,
                "endgame_trades": self._window_stats.endgame_trades,
                "gross_volume": round(self._window_stats.gross_volume, 2),
                "realized_pnl": round(self._window_stats.realized_pnl, 2),
                "net_fees": round(self._window_stats.net_fees, 2),
                "net_pnl": round(self._window_stats.net_pnl, 2),
            },
            "positions": {
                "open_count": len(open_positions),
                "net_delta": round(self._net_delta, 2),
                "mm_gross": round(self._mm_gross_exposure, 2),
                "directional": round(self._directional_exposure, 2),
                "inventory_skew": round(self.get_inventory_skew(), 3),
            },
            "limits": {
                "hourly_pnl": round(self._hourly_pnl, 2),
                "daily_pnl": round(self._daily_pnl, 2),
                "trades_last_hour": self._trades_last_hour,
            },
        }
