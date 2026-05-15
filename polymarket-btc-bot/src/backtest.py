"""
Backtest Engine - Simulate strategy performance on historical BTC data.

Simulates the full trading lifecycle on historical 1-minute BTC candles:
1. Fetches historical klines from Binance API
2. Replays candles to build PriceFeedState indicators
3. Simulates 5-minute windows with synthetic market pricing
4. Runs the Strategy + SignalEngine against each window
5. Models fills with realistic spread/slippage assumptions
6. Tracks cumulative P&L, win rate, and key metrics

Usage:
    python -m src.backtest --days 7 --start 2026-05-01

Output:
    Per-window results + aggregate statistics + equity curve data
"""

import asyncio
import math
import time
import logging
import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import httpx

from src.strategy import Strategy, StrategyConfig, MarketState, StrategyMode, ActionType, TokenSide
from src.signals import SignalEngine, SignalConfig
from src.price_feed import PriceFeedState, Candle, VolatilityRegime

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Backtest parameters."""
    # Data
    start_date: str = "2026-05-08"       # YYYY-MM-DD
    days: int = 7                         # Number of days to simulate
    symbol: str = "BTCUSDT"

    # Market simulation
    base_spread: float = 0.02             # Simulated market spread (2 cents)
    taker_fee: float = 0.07              # 7% taker fee
    maker_rebate: float = 0.014          # 20% of 7% = 1.4% rebate
    fill_probability_passive: float = 0.6 # Probability that passive orders fill
    slippage_bps: float = 10             # Slippage in basis points for aggressive orders

    # Strategy (use defaults)
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)
    signal_config: SignalConfig = field(default_factory=SignalConfig)

    # Capital
    starting_balance: float = 1000.0


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Result of a single 5-minute window backtest."""
    window_start: datetime
    window_end: datetime
    open_price: float
    close_price: float
    resolved_up: bool          # True if close >= open
    actions_generated: int
    trades_executed: int
    mm_fills: int
    momentum_trades: int
    endgame_trades: int
    gross_pnl: float           # Before fees
    fees_paid: float
    fees_rebated: float
    net_pnl: float             # After fees
    final_delta: float         # Net inventory at resolution
    max_drawdown: float        # Peak-to-trough within window

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestResult:
    """Aggregate backtest results."""
    config: BacktestConfig
    windows: list[WindowResult] = field(default_factory=list)
    total_windows: int = 0
    profitable_windows: int = 0
    total_net_pnl: float = 0.0
    total_fees: float = 0.0
    total_rebates: float = 0.0
    total_volume: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return (self.profitable_windows / self.total_windows) * 100.0

    @property
    def avg_pnl_per_window(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_net_pnl / self.total_windows

    @property
    def profit_factor(self) -> float:
        wins = sum(w.net_pnl for w in self.windows if w.net_pnl > 0)
        losses = abs(sum(w.net_pnl for w in self.windows if w.net_pnl < 0))
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        return wins / losses

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe (assuming 288 windows per day)."""
        if len(self.windows) < 10:
            return 0.0
        returns = [w.net_pnl for w in self.windows]
        avg = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        # Annualize: 288 windows/day * 365 days
        return float(avg / std * math.sqrt(288 * 365))

    def summary(self) -> str:
        """Generate a formatted summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Period:          {self.config.start_date} → +{self.config.days} days{' '*22}║
║  Windows Tested:  {self.total_windows:<46}║
║  Win Rate:        {self.win_rate:.1f}%{' '*42}║
║  Net PnL:         ${self.total_net_pnl:+.2f}{' '*36}║
║  Avg PnL/Window:  ${self.avg_pnl_per_window:+.4f}{' '*36}║
║  Profit Factor:   {self.profit_factor:.2f}{' '*42}║
║  Sharpe Ratio:    {self.sharpe_ratio:.2f}{' '*42}║
║  Max Drawdown:    ${self.max_drawdown:.2f}{' '*36}║
║  Total Fees:      ${self.total_fees:.2f}{' '*36}║
║  Total Rebates:   ${self.total_rebates:.2f}{' '*36}║
║  Total Volume:    ${self.total_volume:.2f}{' '*34}║
╠══════════════════════════════════════════════════════════════════╣
║  By Mode:                                                        ║
║    MM Fills:      {sum(w.mm_fills for w in self.windows):<46}║
║    Momentum:      {sum(w.momentum_trades for w in self.windows):<46}║
║    Endgame:       {sum(w.endgame_trades for w in self.windows):<46}║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─── Historical Data Fetcher ──────────────────────────────────────────────────

async def fetch_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_date: str = "2026-05-08",
    days: int = 7,
) -> list[Candle]:
    """
    Fetch historical 1-minute candles from Binance API.
    Returns list of Candle objects.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    candles = []
    current_start = start_ms

    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }

            response = await client.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
            )

            if response.status_code != 200:
                logger.error(f"Binance API error: {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            for k in data:
                candle = Candle(
                    open_time=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    quote_volume=float(k[7]),
                    trades=int(k[8]),
                    close_time=int(k[6]),
                    is_closed=True,
                )
                candles.append(candle)

            # Move to next batch
            current_start = int(data[-1][6]) + 1  # close_time + 1ms

            logger.info(f"Fetched {len(candles)} candles so far...")

            # Rate limiting
            await asyncio.sleep(0.2)

    logger.info(f"Total candles fetched: {len(candles)}")
    return candles


# ─── Backtest Engine ──────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Replays historical data through the strategy engine to measure performance.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy = Strategy(config.strategy_config)
        self.signal_engine = SignalEngine(config.signal_config)

        # Simulated state
        self._balance = config.starting_balance
        self._price_state = PriceFeedState()

    async def run(self) -> BacktestResult:
        """Run the full backtest."""
        logger.info(f"Starting backtest: {self.config.start_date} +{self.config.days} days")

        # Fetch data
        candles = await fetch_historical_klines(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            days=self.config.days,
        )

        if len(candles) < 10:
            logger.error("Not enough candle data for backtest")
            return BacktestResult(config=self.config)

        # Group candles into 5-minute windows
        windows = self._create_windows(candles)
        logger.info(f"Created {len(windows)} 5-minute windows")

        # Run each window
        result = BacktestResult(config=self.config)
        equity = self.config.starting_balance
        peak_equity = equity

        for i, window_candles in enumerate(windows):
            if len(window_candles) < 3:
                continue

            # Build price state from preceding candles (warmup)
            warmup_start = max(0, i * 5 - 60)  # 60 candles before window
            warmup_candles = candles[warmup_start:i * 5]
            self._warmup_price_state(warmup_candles)

            # Run window simulation
            window_result = self._simulate_window(window_candles)
            result.windows.append(window_result)
            result.total_windows += 1

            if window_result.is_profitable:
                result.profitable_windows += 1

            result.total_net_pnl += window_result.net_pnl
            result.total_fees += window_result.fees_paid
            result.total_rebates += window_result.fees_rebated
            result.total_volume += window_result.trades_executed * 10  # Approx volume

            # Equity curve
            equity += window_result.net_pnl
            result.equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            drawdown = peak_equity - equity
            result.max_drawdown = max(result.max_drawdown, drawdown)

            # Progress
            if (i + 1) % 100 == 0:
                logger.info(
                    f"  Window {i+1}/{len(windows)}: "
                    f"PnL=${result.total_net_pnl:+.2f} | "
                    f"WR={result.win_rate:.1f}%"
                )

        result.peak_equity = peak_equity
        return result

    def _create_windows(self, candles: list[Candle]) -> list[list[Candle]]:
        """Group 1-minute candles into consecutive 5-minute windows."""
        windows = []
        for i in range(0, len(candles) - 4, 5):
            window = candles[i:i + 5]
            if len(window) == 5:
                windows.append(window)
        return windows

    def _warmup_price_state(self, candles: list[Candle]):
        """Feed historical candles into price state for indicator warmup."""
        self._price_state = PriceFeedState()
        for candle in candles:
            self._price_state.candles.append(candle)
            self._price_state.current_price = candle.close
            self._price_state.last_update_time = candle.close_time / 1000.0

    def _simulate_window(self, window_candles: list[Candle]) -> WindowResult:
        """
        Simulate one 5-minute window through the strategy.
        
        Approach:
        - Use the first candle's open as the window open price
        - Simulate strategy evaluation every ~30 seconds (10 ticks)
        - Model market prices using the strategy's probability estimate
        - Apply fills with fee/slippage assumptions
        """
        open_price = window_candles[0].open
        close_price = window_candles[-1].close
        resolved_up = close_price >= open_price

        # Strategy state
        self.strategy.reset_window()
        cfg = self.config

        actions_count = 0
        trades_count = 0
        mm_fills = 0
        momentum_trades = 0
        endgame_trades = 0
        gross_pnl = 0.0
        fees_paid = 0.0
        fees_rebated = 0.0
        positions_up = 0.0   # USDC in Up tokens
        positions_down = 0.0  # USDC in Down tokens
        avg_entry_up = 0.0
        avg_entry_down = 0.0

        # Simulate 10 evaluation points across the 5-minute window
        total_seconds = 300.0
        eval_points = 10
        interval = total_seconds / eval_points

        for tick in range(eval_points):
            elapsed = (tick + 1) * interval
            remaining = total_seconds - elapsed

            # Interpolate BTC price at this point in the window
            progress = elapsed / total_seconds
            candle_idx = min(int(progress * 5), 4)
            btc_price = window_candles[candle_idx].close

            # Update price state
            self._price_state.current_price = btc_price
            self._price_state.last_update_time = time.time()

            # Feed the candle if we've crossed a boundary
            if candle_idx < len(window_candles):
                c = window_candles[candle_idx]
                if c not in list(self._price_state.candles)[-5:]:
                    self._price_state.candles.append(c)

            # Calculate displacement
            displacement = (btc_price - open_price) / open_price if open_price > 0 else 0

            # Get signals
            price_signals = self._price_state.get_all_signals()
            # Simulate minimal microstructure signals
            micro_signals = {
                "imbalance_up": displacement * 2,  # Simplified: price up → more bids
                "imbalance_trend": displacement,
                "spread_up": cfg.base_spread,
                "avg_spread": cfg.base_spread,
            }

            composite, confidence = self.signal_engine.compute(
                price_signals, micro_signals, elapsed
            )

            # Simulated market prices (based on "true" probability + noise)
            # True prob at this point (simplified model)
            vol_per_sec = 0.00009
            remaining_std = vol_per_sec * math.sqrt(max(1, remaining))
            if remaining_std > 0:
                z = displacement / remaining_std
                true_prob_up = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            else:
                true_prob_up = 1.0 if displacement >= 0 else 0.0

            # Market price = true prob + some noise (inefficiency)
            noise = np.random.normal(0, 0.015)  # Market is slightly noisy
            market_up = max(0.05, min(0.95, true_prob_up + noise))
            market_down = 1.0 - market_up

            # Build MarketState
            state = MarketState(
                up_price=market_up,
                down_price=market_down,
                best_bid_up=market_up - cfg.base_spread / 2,
                best_ask_up=market_up + cfg.base_spread / 2,
                best_bid_down=market_down - cfg.base_spread / 2,
                best_ask_down=market_down + cfg.base_spread / 2,
                spread_up=cfg.base_spread,
                spread_down=cfg.base_spread,
                book_imbalance_up=displacement * 1.5,
                book_imbalance_down=-displacement * 1.5,
                bid_depth_up=500.0,
                ask_depth_up=500.0,
                seconds_elapsed=elapsed,
                seconds_remaining=remaining,
                progress_pct=(elapsed / total_seconds) * 100,
                btc_price=btc_price,
                window_open_price=open_price,
                btc_displacement_pct=displacement,
                momentum_score=composite,
                volatility_regime=price_signals.get("volatility_regime", "normal"),
                volume_surge=False,
                vwap_deviation=price_signals.get("vwap_deviation", 0) or 0,
                large_trade_bias=0.0,
                composite_signal=composite,
                signal_confidence=confidence,
            )

            # Evaluate strategy
            actions = self.strategy.evaluate(state)
            actions_count += len(actions)

            # Process actions (simplified fill simulation)
            for action in actions:
                if action.action_type in (ActionType.HOLD, ActionType.CANCEL_ALL, ActionType.CANCEL_SIDE):
                    continue

                # Determine if the order fills
                is_passive = action.is_passive
                fills = False

                if is_passive:
                    # Passive (limit) orders fill with probability
                    fills = np.random.random() < cfg.fill_probability_passive
                else:
                    # Aggressive orders always fill
                    fills = True

                if not fills:
                    continue

                trades_count += 1
                size = min(action.size, 20.0)  # Cap for backtest

                # Track by mode
                if action.mode == StrategyMode.MARKET_MAKING:
                    mm_fills += 1
                elif action.mode == StrategyMode.MOMENTUM:
                    momentum_trades += 1
                elif action.mode == StrategyMode.ENDGAME_ARB:
                    endgame_trades += 1

                # Record position
                entry_price = action.price
                if action.action_type in (ActionType.PLACE_BID, ActionType.MARKET_BUY):
                    if action.token_side == TokenSide.UP:
                        positions_up += size
                        avg_entry_up = entry_price if avg_entry_up == 0 else (avg_entry_up + entry_price) / 2
                    else:
                        positions_down += size
                        avg_entry_down = entry_price if avg_entry_down == 0 else (avg_entry_down + entry_price) / 2

                # Fees
                if is_passive:
                    rebate = size * cfg.maker_rebate
                    fees_rebated += rebate
                else:
                    fee = size * cfg.taker_fee
                    fees_paid += fee

        # ─── Resolution ───────────────────────────────────────────────
        # Calculate P&L based on market resolution
        if resolved_up:
            # Up tokens pay 1.0 per share, Down tokens pay 0.0
            if positions_up > 0 and avg_entry_up > 0:
                shares_up = positions_up / avg_entry_up
                payout_up = shares_up * 1.0  # Each share pays $1
                gross_pnl += payout_up - positions_up
            if positions_down > 0:
                gross_pnl -= positions_down  # Total loss on Down
        else:
            # Down tokens pay 1.0, Up tokens pay 0.0
            if positions_down > 0 and avg_entry_down > 0:
                shares_down = positions_down / avg_entry_down
                payout_down = shares_down * 1.0
                gross_pnl += payout_down - positions_down
            if positions_up > 0:
                gross_pnl -= positions_up  # Total loss on Up

        net_pnl = gross_pnl - fees_paid + fees_rebated
        final_delta = positions_up - positions_down

        return WindowResult(
            window_start=datetime.fromtimestamp(window_candles[0].open_time / 1000, tz=timezone.utc),
            window_end=datetime.fromtimestamp(window_candles[-1].close_time / 1000, tz=timezone.utc),
            open_price=open_price,
            close_price=close_price,
            resolved_up=resolved_up,
            actions_generated=actions_count,
            trades_executed=trades_count,
            mm_fills=mm_fills,
            momentum_trades=momentum_trades,
            endgame_trades=endgame_trades,
            gross_pnl=round(gross_pnl, 4),
            fees_paid=round(fees_paid, 4),
            fees_rebated=round(fees_rebated, 4),
            net_pnl=round(net_pnl, 4),
            final_delta=round(final_delta, 2),
            max_drawdown=0.0,
        )


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

async def run_backtest(config: BacktestConfig):
    """Run backtest and print results."""
    engine = BacktestEngine(config)
    result = await engine.run()

    print(result.summary())

    # Print top/bottom windows
    if result.windows:
        sorted_windows = sorted(result.windows, key=lambda w: w.net_pnl, reverse=True)

        print("\n🏆 Best 5 Windows:")
        for w in sorted_windows[:5]:
            print(
                f"  {w.window_start.strftime('%m/%d %H:%M')} | "
                f"PnL=${w.net_pnl:+.4f} | Resolved={'UP' if w.resolved_up else 'DOWN'} | "
                f"Trades={w.trades_executed}"
            )

        print("\n💀 Worst 5 Windows:")
        for w in sorted_windows[-5:]:
            print(
                f"  {w.window_start.strftime('%m/%d %H:%M')} | "
                f"PnL=${w.net_pnl:+.4f} | Resolved={'UP' if w.resolved_up else 'DOWN'} | "
                f"Trades={w.trades_executed}"
            )

    # Save equity curve
    if result.equity_curve:
        with open("backtest_equity.json", "w") as f:
            json.dump({
                "equity_curve": result.equity_curve,
                "config": {
                    "start_date": config.start_date,
                    "days": config.days,
                    "starting_balance": config.starting_balance,
                },
                "summary": {
                    "total_pnl": result.total_net_pnl,
                    "win_rate": result.win_rate,
                    "sharpe": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "profit_factor": result.profit_factor,
                    "total_windows": result.total_windows,
                },
            }, f, indent=2)
        print(f"\n📈 Equity curve saved to backtest_equity.json")

    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest BTC 5m trading strategy")
    parser.add_argument("--start", default="2026-05-08", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of days")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance")
    parser.add_argument("--spread", type=float, default=0.02, help="Simulated market spread")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    config = BacktestConfig(
        start_date=args.start,
        days=args.days,
        starting_balance=args.balance,
        base_spread=args.spread,
    )

    asyncio.run(run_backtest(config))


if __name__ == "__main__":
    main()
