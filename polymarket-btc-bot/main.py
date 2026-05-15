"""
Polymarket BTC 5-Minute Trading Bot - Main Entry Point

PURE POLYMARKET: No external price feed (no Binance). All signals derived
from Polymarket orderbook data (Up/Down prices, depth, spread dynamics).

Three-mode strategy: Market Making + Momentum + Endgame Arbitrage

Usage:
    python main.py              # Uses DRY_RUN from .env
    python main.py --dry-run    # Force paper trading
    python main.py --live       # Force live trading
    python main.py --status     # Show config and exit
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime, timezone

from config import config, BotConfig
from src.market_finder import MarketFinder
from src.price_feed import PriceFeed
from src.signals import SignalEngine
from src.strategy import Strategy
from src.polymarket_client import PolymarketClient
from src.risk import RiskManager
from src.trader import Trader


def setup_logging(level: str = "INFO"):
    log_format = "%(asctime)s | %(levelname)-5s | %(name)-18s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", mode="a", encoding="utf-8"),
        ],
    )
    for lib in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


class Bot:
    """Main bot orchestrator."""

    def __init__(self, cfg: BotConfig):
        self.config = cfg
        self._running = False

        self.market_finder = MarketFinder(gamma_api_url=cfg.network.gamma_api_url)
        self.price_feed = PriceFeed()
        self.signal_engine = SignalEngine(cfg.to_signal_config())
        self.strategy = Strategy(cfg.to_strategy_config())
        self.risk_manager = RiskManager(cfg.to_risk_limits())
        self.client = PolymarketClient(
            private_key=cfg.wallet.private_key,
            api_key=cfg.wallet.api_key,
            api_secret=cfg.wallet.api_secret,
            api_passphrase=cfg.wallet.api_passphrase,
            chain_id=cfg.wallet.chain_id,
            clob_url=cfg.network.clob_api_url,
            dry_run=cfg.dry_run,
        )
        self.trader = Trader(
            market_finder=self.market_finder,
            price_feed=self.price_feed,
            signal_engine=self.signal_engine,
            strategy=self.strategy,
            risk_manager=self.risk_manager,
            client=self.client,
            dry_run=cfg.dry_run,
        )

    async def run(self):
        self._running = True

        errors = self.config.validate()
        if errors:
            for e in errors:
                logging.error(f"Config error: {e}")
            sys.exit(1)

        self._print_banner()

        # Start components (no external WebSocket needed!)
        await self.price_feed.start()
        await self.trader.start()

        # Instead of waiting for Binance, just verify we can reach Polymarket
        logging.info("Checking Polymarket API connection...")
        snapshot = await self.market_finder.get_snapshot(force_refresh=True)
        if snapshot:
            logging.info(f"Connected! Found market: {snapshot.question}")
            logging.info(f"  Up={snapshot.up_price:.3f} Down={snapshot.down_price:.3f}")
            # Push first tick so price_feed is ready
            from src.price_feed import PriceTick
            import time
            self.price_feed.push_tick(PriceTick(
                timestamp=time.time(),
                up_price=snapshot.up_price,
                down_price=snapshot.down_price,
                best_bid_up=snapshot.best_bid_up,
                best_ask_up=snapshot.best_ask_up,
                spread_up=snapshot.spread_up,
                imbalance_up=snapshot.book_imbalance_up,
            ))
        else:
            logging.warning("No active BTC 5m market found (may be between windows)")
            logging.info("Will keep polling until one appears...")

        logging.info("Entering main trading loop...\n")

        consecutive_misses = 0
        try:
            while self._running:
                try:
                    executed = await self.trader.run_window()
                    if executed:
                        consecutive_misses = 0
                        await asyncio.sleep(2.0)
                    else:
                        consecutive_misses += 1
                        if consecutive_misses >= 20:
                            logging.warning("No active market for 60s")
                            consecutive_misses = 0
                        await asyncio.sleep(3.0)
                except asyncio.CancelledError:
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.error(f"Main loop error: {e}", exc_info=True)
                    await asyncio.sleep(5.0)
        finally:
            await self.shutdown()

    async def shutdown(self):
        logging.info("\nShutting down...")
        self._running = False
        await self.trader.stop()
        await self.price_feed.stop()
        await self.market_finder.close()
        self._print_summary()
        logging.info("Goodbye!")

    def _print_banner(self):
        mode = "PAPER" if self.config.dry_run else "LIVE"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cfg = self.config
        print(f"""
{'='*70}
  Polymarket BTC 5-Minute Trading Bot v2.1
  Pure Polymarket - No External Price Feed
{'='*70}
  Mode:          [{mode}] {'Paper Trading' if cfg.dry_run else 'LIVE TRADING'}
  Started:       {now}
  Data Source:   Polymarket Orderbook (CLOB + Gamma API)
{'─'*70}
  Market Making: spread={cfg.market_making.spread:.2f} size={cfg.market_making.size:.0f}
  Momentum:      threshold={cfg.momentum.threshold:.1f} size={cfg.momentum.size:.0f}-{cfg.momentum.max_size:.0f}
  Endgame Arb:   edge>{cfg.endgame.prob_threshold:.2f} size={cfg.endgame.size:.0f}-{cfg.endgame.max_size:.0f}
{'─'*70}
  Risk: max_pos={cfg.risk.dir_max_position:.0f} SL={cfg.risk.dir_stop_loss_pct*100:.0f}% TP={cfg.risk.dir_take_profit_pct*100:.0f}%
  Fees: taker={cfg.risk.taker_fee_rate*100:.0f}% rebate={cfg.risk.maker_rebate_rate*100:.0f}%
{'='*70}
""")

    def _print_summary(self):
        status = self.trader.get_status()
        cum = status.get("cumulative", {})
        print(f"""
{'─'*70}
  SESSION SUMMARY
  Windows: {cum.get('windows', 0)} | Win Rate: {cum.get('win_rate', 0):.0f}%
  PnL: ${cum.get('total_pnl', 0):+.2f} | Fills: {cum.get('total_fills', 0)}
  Best: ${cum.get('best_window', 0):+.2f} | Worst: ${cum.get('worst_window', 0):+.2f}
{'─'*70}
""")


def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5m Bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Paper trading")
    group.add_argument("--live", action="store_true", help="Live trading")
    group.add_argument("--status", action="store_true", help="Show config")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def async_main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.live:
        config.dry_run = False
    elif args.dry_run:
        config.dry_run = True

    if args.status:
        errors = config.validate()
        print(f"Mode: {'PAPER' if config.dry_run else 'LIVE'}")
        print(f"Validation: {'OK' if not errors else errors}")
        return

    bot = Bot(config)

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: setattr(bot, '_running', False))
        loop.add_signal_handler(signal.SIGTERM, lambda: setattr(bot, '_running', False))
    except NotImplementedError:
        pass

    await bot.run()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
