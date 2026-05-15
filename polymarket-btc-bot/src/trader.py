"""
Trader - Orchestrates the full 5-minute window lifecycle.

PURE POLYMARKET: No external price feed. The MarketFinder provides
all data (orderbook, prices), which is pushed into PriceFeed for
signal computation. The trader loop is:

  Every 1.5s:
    1. Refresh orderbook from Polymarket CLOB API
    2. Push new tick into PriceFeed (for signal history)
    3. Get composite signal from SignalEngine
    4. Build MarketState
    5. Evaluate Strategy → Actions
    6. Risk-check and execute Actions
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.market_finder import MarketFinder, MarketSnapshot
from src.price_feed import PriceFeed, PriceTick
from src.signals import SignalEngine
from src.strategy import (
    Strategy, StrategyMode, Action, ActionType, TokenSide, MarketState,
)
from src.risk import RiskManager, RiskLimits
from src.polymarket_client import (
    PolymarketClient, OrderSide, FillEvent, OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowSession:
    """State for a single 5-minute trading window."""
    market: MarketSnapshot
    window_start_up_price: float = 0.5
    started_at: float = field(default_factory=time.time)
    is_active: bool = True
    actions_executed: int = 0
    fills_count: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


@dataclass
class TraderStats:
    windows_traded: int = 0
    windows_profitable: int = 0
    total_pnl: float = 0.0
    total_fills: int = 0
    best_window_pnl: float = 0.0
    worst_window_pnl: float = 0.0


class Trader:
    """Main trading engine. Pure Polymarket, no external dependencies."""

    def __init__(
        self,
        market_finder: MarketFinder,
        price_feed: PriceFeed,
        signal_engine: SignalEngine,
        strategy: Strategy,
        risk_manager: RiskManager,
        client: PolymarketClient,
        dry_run: bool = True,
    ):
        self.market_finder = market_finder
        self.price_feed = price_feed
        self.signal_engine = signal_engine
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.client = client
        self.dry_run = dry_run

        self._running = False
        self._session: Optional[WindowSession] = None
        self._stats = TraderStats()
        self._tick_interval = 1.5

        self.client.on_fill(self._on_fill)

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self):
        self._running = True
        await self.client.initialize()
        logger.info("Trader engine started (Polymarket-native)")

    async def stop(self):
        self._running = False
        if self._session and self._session.is_active:
            await self._end_window("Trader stopping")
        await self.client.cancel_all()
        logger.info("Trader engine stopped")

    async def run_window(self) -> bool:
        """Run one complete 5-minute window. Returns True if executed."""
        if not self._running:
            return False

        # Find active market
        snapshot = await self.market_finder.get_snapshot(force_refresh=True)
        if not snapshot or not snapshot.is_tradeable:
            return False

        # Initialize window
        self.strategy.reset_window()
        self.risk_manager.reset_window()
        self.risk_manager.reset_hourly()
        self.risk_manager.reset_daily()

        session = WindowSession(
            market=snapshot,
            window_start_up_price=snapshot.up_price,
        )
        self._session = session
        self.strategy.set_window_start_price(snapshot.up_price)

        # Push initial tick
        self._push_tick(snapshot)

        logger.info(
            f"\n{'='*70}\n"
            f"  NEW WINDOW: {snapshot.question}\n"
            f"  {snapshot.event_start.strftime('%H:%M:%S')} -> {snapshot.end_time.strftime('%H:%M:%S')} UTC\n"
            f"  Up={snapshot.up_price:.3f} Down={snapshot.down_price:.3f} "
            f"Spread={snapshot.spread_up:.3f}\n"
            f"{'='*70}"
        )

        try:
            await self._trading_loop(session)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
        finally:
            await self._end_window("Window complete")

        return True

    async def _trading_loop(self, session: WindowSession):
        """Core loop: refresh orderbook → signals → strategy → execute."""
        while self._running and session.is_active:
            market = session.market

            if market.seconds_until_close <= 5:
                break

            # ─── Step 1: Refresh market data from Polymarket ──────────
            refreshed = await self.market_finder.get_snapshot()
            if refreshed and refreshed.slug == market.slug:
                session.market = refreshed
                market = refreshed
                # Push new tick into price feed
                self._push_tick(market)

            if self.price_feed.state.is_stale:
                await asyncio.sleep(0.5)
                continue

            # ─── Step 2: Build MarketState ────────────────────────────
            state = self._build_market_state(session)

            # ─── Step 3: Evaluate strategy ────────────────────────────
            actions = self.strategy.evaluate(state)

            # ─── Step 4: Execute actions ──────────────────────────────
            for action in actions:
                await self._execute_action(action, session)

            # ─── Step 5: Check exits ──────────────────────────────────
            positions_to_close = self.risk_manager.check_exits(
                market.up_price, market.down_price
            )
            for pos in positions_to_close:
                await self._close_position(pos, session)

            # ─── Step 6: Hedging ──────────────────────────────────────
            should_hedge, hedge_side, hedge_size = self.risk_manager.should_hedge()
            if should_hedge:
                await self._execute_hedge(hedge_side, hedge_size, session)

            # ─── Step 7: Simulate fills (paper) ───────────────────────
            if self.dry_run:
                self.client.simulate_passive_fills(market.up_price, market.down_price)

            await asyncio.sleep(self._tick_interval)

    def _push_tick(self, market: MarketSnapshot):
        """Push current market state as a price tick."""
        tick = PriceTick(
            timestamp=time.time(),
            up_price=market.up_price,
            down_price=market.down_price,
            best_bid_up=market.best_bid_up,
            best_ask_up=market.best_ask_up,
            best_bid_down=market.best_bid_down,
            best_ask_down=market.best_ask_down,
            spread_up=market.spread_up,
            spread_down=market.spread_down,
            bid_depth_up=market.book_up.total_bid_depth,
            ask_depth_up=market.book_up.total_ask_depth,
            imbalance_up=market.book_imbalance_up,
        )
        self.price_feed.push_tick(tick)

    def _build_market_state(self, session: WindowSession) -> MarketState:
        """Build MarketState from Polymarket data only."""
        market = session.market
        ps = self.price_feed.state

        # Get signals from fusion engine
        price_signals = ps.get_all_signals()
        micro_signals = self.market_finder.get_microstructure_signals()
        composite, confidence = self.signal_engine.compute(
            price_signals, micro_signals, market.seconds_since_start
        )

        # Displacement from window start
        up_displacement = market.up_price - session.window_start_up_price

        return MarketState(
            up_price=market.up_price,
            down_price=market.down_price,
            best_bid_up=market.best_bid_up,
            best_ask_up=market.best_ask_up,
            best_bid_down=market.best_bid_down,
            best_ask_down=market.best_ask_down,
            spread_up=market.spread_up,
            spread_down=market.spread_down,
            book_imbalance_up=market.book_imbalance_up,
            book_imbalance_down=market.book_imbalance_down,
            bid_depth_up=market.book_up.total_bid_depth,
            ask_depth_up=market.book_up.total_ask_depth,
            seconds_elapsed=market.seconds_since_start,
            seconds_remaining=market.seconds_until_close,
            progress_pct=market.progress_pct,
            up_displacement=up_displacement,
            composite_signal=composite,
            signal_confidence=confidence,
            momentum_score=composite,
            volatility_regime=price_signals.get("volatility_regime", "normal"),
            volume_surge=price_signals.get("volume_surge", False) or False,
            spread_trend=ps.get_spread_trend(),
            imbalance_velocity=ps.get_depth_imbalance_velocity(),
        )

    async def _execute_action(self, action: Action, session: WindowSession):
        market = session.market

        if action.action_type == ActionType.HOLD:
            return
        if action.action_type == ActionType.CANCEL_ALL:
            await self.client.cancel_all()
            return
        if action.action_type == ActionType.CANCEL_SIDE:
            token_id = market.up_token_id if action.token_side == TokenSide.UP else market.down_token_id
            await self.client.cancel_token_orders(token_id)
            return

        balance = await self.client.get_balance()
        approved, adjusted_size, reason = self.risk_manager.check_action(action, balance)

        if not approved:
            logger.debug(f"Rejected: {reason}")
            return

        token_id = market.up_token_id if action.token_side == TokenSide.UP else market.down_token_id
        price = action.price
        size = adjusted_size

        order_type = OrderType.GTC
        side = OrderSide.BUY

        if action.action_type == ActionType.PLACE_BID:
            side = OrderSide.BUY
        elif action.action_type == ActionType.PLACE_ASK:
            side = OrderSide.SELL
        elif action.action_type == ActionType.MARKET_BUY:
            side = OrderSide.BUY
            order_type = OrderType.IOC
        elif action.action_type == ActionType.MARKET_SELL:
            side = OrderSide.SELL
            order_type = OrderType.IOC
        else:
            return

        order = await self.client.place_order(
            token_id=token_id, side=side, price=price,
            size=size, condition_id=market.condition_id,
            order_type=order_type,
        )

        if order and (order.is_active or order.status.value == "FILLED"):
            session.actions_executed += 1
            if action.mode == StrategyMode.ENDGAME_ARB:
                logger.info(f"ENDGAME: {action.reason}")
            elif action.mode == StrategyMode.MOMENTUM:
                logger.info(f"MOMENTUM: {action.reason}")

    async def _close_position(self, pos, session: WindowSession):
        market = session.market
        if pos.token_side == TokenSide.UP:
            token_id = market.up_token_id
            exit_price = market.best_bid_up
        else:
            token_id = market.down_token_id
            exit_price = market.best_bid_down

        if exit_price <= 0:
            return

        await self.client.place_order(
            token_id=token_id, side=OrderSide.SELL,
            price=exit_price, size=pos.size,
            condition_id=market.condition_id, order_type=OrderType.IOC,
        )
        self.risk_manager.close_position(token_id, exit_price)

    async def _execute_hedge(self, hedge_side: TokenSide, size: float, session: WindowSession):
        market = session.market
        if hedge_side == TokenSide.UP:
            token_id = market.up_token_id
            price = market.best_ask_up
        else:
            token_id = market.down_token_id
            price = market.best_ask_down

        if price <= 0 or size < 5:
            return

        await self.client.place_order(
            token_id=token_id, side=OrderSide.BUY,
            price=price, size=min(size, 30.0),
            condition_id=market.condition_id, order_type=OrderType.IOC,
        )

    def _on_fill(self, fill: FillEvent):
        if self._session:
            self._session.fills_count += 1

        token_side = TokenSide.UP
        if self._session:
            if fill.token_id == self._session.market.down_token_id:
                token_side = TokenSide.DOWN

        self.risk_manager.record_fill(
            token_side=token_side, size=fill.size, price=fill.price,
            token_id=fill.token_id,
            market_slug=self._session.market.slug if self._session else "",
            mode=self.strategy.current_mode, is_maker=fill.is_maker,
        )
        self.strategy.update_inventory(token_side, fill.size, fill.side == OrderSide.BUY)

    async def _end_window(self, reason: str):
        if not self._session:
            return

        session = self._session
        session.is_active = False
        await self.client.cancel_all()

        stats = self.risk_manager.get_stats()
        window_pnl = stats["window"]["net_pnl"]

        self._stats.windows_traded += 1
        self._stats.total_pnl += window_pnl
        self._stats.total_fills += session.fills_count
        if window_pnl > 0:
            self._stats.windows_profitable += 1
        self._stats.best_window_pnl = max(self._stats.best_window_pnl, window_pnl)
        self._stats.worst_window_pnl = min(self._stats.worst_window_pnl, window_pnl)

        win_rate = (self._stats.windows_profitable / max(1, self._stats.windows_traded) * 100)

        logger.info(
            f"\n{'─'*70}\n"
            f"  WINDOW CLOSED ({reason})\n"
            f"  Duration: {session.elapsed:.0f}s | Actions: {session.actions_executed} | "
            f"Fills: {session.fills_count}\n"
            f"  MM: {stats['window']['mm_fills']} | Mom: {stats['window']['momentum_trades']} | "
            f"Endgame: {stats['window']['endgame_trades']}\n"
            f"  Window PnL: ${window_pnl:+.2f} | Cumulative: ${self._stats.total_pnl:+.2f} | "
            f"WR: {win_rate:.0f}%\n"
            f"{'─'*70}"
        )
        self._session = None

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "mode": self.strategy.current_mode.value,
            "session": {
                "active": self._session is not None and self._session.is_active,
                "elapsed": self._session.elapsed if self._session else 0,
                "fills": self._session.fills_count if self._session else 0,
            },
            "cumulative": {
                "windows": self._stats.windows_traded,
                "win_rate": (self._stats.windows_profitable / max(1, self._stats.windows_traded) * 100),
                "total_pnl": round(self._stats.total_pnl, 2),
                "total_fills": self._stats.total_fills,
                "best_window": round(self._stats.best_window_pnl, 2),
                "worst_window": round(self._stats.worst_window_pnl, 2),
            },
            "risk": self.risk_manager.get_stats(),
            "client": self.client.get_stats(),
        }
