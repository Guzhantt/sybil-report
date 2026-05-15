"""
Configuration for the Polymarket BTC 5m Trading Bot.

All settings loaded from .env file with sensible defaults.
Provides typed dataclasses for each subsystem.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ─── Wallet & Auth ────────────────────────────────────────────────────────────

@dataclass
class WalletConfig:
    private_key: str = field(default_factory=lambda: _env("PRIVATE_KEY"))
    api_key: str = field(default_factory=lambda: _env("POLYMARKET_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("POLYMARKET_API_SECRET"))
    api_passphrase: str = field(default_factory=lambda: _env("POLYMARKET_API_PASSPHRASE"))
    chain_id: int = field(default_factory=lambda: _env_int("CHAIN_ID", 137))


# ─── Network ─────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    clob_api_url: str = field(default_factory=lambda: _env("CLOB_API_URL", "https://clob.polymarket.com"))
    gamma_api_url: str = field(default_factory=lambda: _env("GAMMA_API_URL", "https://gamma-api.polymarket.com"))


# ─── Strategy: Market Making ─────────────────────────────────────────────────

@dataclass
class MarketMakingConfig:
    """Market making mode parameters."""
    spread: float = field(default_factory=lambda: _env_float("MM_SPREAD", 0.03))
    size: float = field(default_factory=lambda: _env_float("MM_SIZE", 10.0))
    max_inventory: float = field(default_factory=lambda: _env_float("MM_MAX_INVENTORY", 30.0))
    skew_factor: float = field(default_factory=lambda: _env_float("MM_SKEW_FACTOR", 0.5))
    start_seconds: float = field(default_factory=lambda: _env_float("MM_START_SECONDS", 10.0))
    end_seconds: float = field(default_factory=lambda: _env_float("MM_END_SECONDS", 180.0))


# ─── Strategy: Momentum ──────────────────────────────────────────────────────

@dataclass
class MomentumConfig:
    """Momentum trading mode parameters."""
    threshold: float = field(default_factory=lambda: _env_float("MOM_THRESHOLD", 0.6))
    size: float = field(default_factory=lambda: _env_float("MOM_SIZE", 15.0))
    max_size: float = field(default_factory=lambda: _env_float("MOM_MAX_SIZE", 40.0))
    start_seconds: float = field(default_factory=lambda: _env_float("MOM_START_SECONDS", 60.0))
    end_seconds: float = field(default_factory=lambda: _env_float("MOM_END_SECONDS", 240.0))


# ─── Strategy: Endgame Arbitrage ─────────────────────────────────────────────

@dataclass
class EndgameConfig:
    """Endgame arbitrage mode parameters."""
    min_displacement: float = field(default_factory=lambda: _env_float("ARB_MIN_DISPLACEMENT", 0.003))
    prob_threshold: float = field(default_factory=lambda: _env_float("ARB_PROB_THRESHOLD", 0.08))
    size: float = field(default_factory=lambda: _env_float("ARB_SIZE", 25.0))
    max_size: float = field(default_factory=lambda: _env_float("ARB_MAX_SIZE", 60.0))
    start_seconds: float = field(default_factory=lambda: _env_float("ARB_START_SECONDS", 240.0))
    cutoff_seconds: float = field(default_factory=lambda: _env_float("ARB_CUTOFF_SECONDS", 285.0))


# ─── Strategy: Global ────────────────────────────────────────────────────────

@dataclass
class StrategyGlobalConfig:
    """Strategy-level global parameters."""
    edge_threshold: float = field(default_factory=lambda: _env_float("EDGE_THRESHOLD", 0.04))
    warmup_period: float = field(default_factory=lambda: _env_float("WARMUP_PERIOD", 10.0))
    max_loss_per_window: float = field(default_factory=lambda: _env_float("MAX_LOSS_PER_WINDOW", 20.0))


# ─── Risk Management ─────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Market Making
    mm_max_gross_exposure: float = field(default_factory=lambda: _env_float("MM_MAX_GROSS_EXPOSURE", 60.0))
    mm_max_net_delta: float = field(default_factory=lambda: _env_float("MM_MAX_NET_DELTA", 30.0))
    mm_inventory_skew_start: float = field(default_factory=lambda: _env_float("MM_INVENTORY_SKEW_START", 15.0))

    # Directional
    dir_max_position: float = field(default_factory=lambda: _env_float("DIR_MAX_POSITION", 20.0))
    dir_max_total: float = field(default_factory=lambda: _env_float("DIR_MAX_TOTAL", 50.0))
    dir_stop_loss_pct: float = field(default_factory=lambda: _env_float("DIR_STOP_LOSS_PCT", 0.20))
    dir_take_profit_pct: float = field(default_factory=lambda: _env_float("DIR_TAKE_PROFIT_PCT", 0.40))

    # Global
    max_loss_per_window: float = field(default_factory=lambda: _env_float("MAX_LOSS_PER_WINDOW", 25.0))
    max_loss_per_hour: float = field(default_factory=lambda: _env_float("MAX_LOSS_PER_HOUR", 80.0))
    max_loss_per_day: float = field(default_factory=lambda: _env_float("MAX_LOSS_PER_DAY", 200.0))
    max_trades_per_hour: int = field(default_factory=lambda: _env_int("MAX_TRADES_PER_HOUR", 60))
    min_trade_interval: float = field(default_factory=lambda: _env_float("MIN_TRADE_INTERVAL", 2.0))
    min_edge_after_fees: float = field(default_factory=lambda: _env_float("MIN_EDGE_AFTER_FEES", 0.01))

    # Fee structure
    taker_fee_rate: float = field(default_factory=lambda: _env_float("TAKER_FEE_RATE", 0.07))
    maker_rebate_rate: float = field(default_factory=lambda: _env_float("MAKER_REBATE_RATE", 0.20))


# ─── Signal Engine ────────────────────────────────────────────────────────────

@dataclass
class SignalConfig:
    """Signal fusion engine parameters."""
    trend_weight: float = field(default_factory=lambda: _env_float("SIG_TREND_WEIGHT", 0.30))
    flow_weight: float = field(default_factory=lambda: _env_float("SIG_FLOW_WEIGHT", 0.30))
    micro_weight: float = field(default_factory=lambda: _env_float("SIG_MICRO_WEIGHT", 0.25))
    reversion_weight: float = field(default_factory=lambda: _env_float("SIG_REVERSION_WEIGHT", 0.15))
    convergence_min_signals: int = field(default_factory=lambda: _env_int("SIG_CONVERGENCE_MIN", 3))
    convergence_threshold: float = field(default_factory=lambda: _env_float("SIG_CONVERGENCE_THRESHOLD", 0.5))


# ─── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestEnvConfig:
    """Backtest parameters from env."""
    start_date: str = field(default_factory=lambda: _env("BT_START_DATE", "2026-05-08"))
    days: int = field(default_factory=lambda: _env_int("BT_DAYS", 7))
    starting_balance: float = field(default_factory=lambda: _env_float("BT_BALANCE", 1000.0))
    base_spread: float = field(default_factory=lambda: _env_float("BT_BASE_SPREAD", 0.02))


# ─── Master Config ────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    """Root configuration combining all subsystems."""
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))
    wallet: WalletConfig = field(default_factory=WalletConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    market_making: MarketMakingConfig = field(default_factory=MarketMakingConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    endgame: EndgameConfig = field(default_factory=EndgameConfig)
    strategy: StrategyGlobalConfig = field(default_factory=StrategyGlobalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestEnvConfig = field(default_factory=BacktestEnvConfig)

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.dry_run:
            if not self.wallet.private_key or self.wallet.private_key == "0xYOUR_PRIVATE_KEY_HERE":
                errors.append("PRIVATE_KEY required for live trading")
            if not self.wallet.api_key:
                errors.append("POLYMARKET_API_KEY required for live trading")
            if not self.wallet.api_secret:
                errors.append("POLYMARKET_API_SECRET required for live trading")
            if not self.wallet.api_passphrase:
                errors.append("POLYMARKET_API_PASSPHRASE required for live trading")
        if self.strategy.edge_threshold <= 0 or self.strategy.edge_threshold >= 1:
            errors.append("EDGE_THRESHOLD must be between 0 and 1")
        if self.risk.taker_fee_rate < 0 or self.risk.taker_fee_rate > 0.5:
            errors.append("TAKER_FEE_RATE must be 0-0.5")
        return errors

    def to_strategy_config(self):
        """Convert to the StrategyConfig dataclass used by strategy.py."""
        from src.strategy import StrategyConfig as SC
        return SC(
            mm_spread=self.market_making.spread,
            mm_size=self.market_making.size,
            mm_max_inventory=self.market_making.max_inventory,
            mm_skew_factor=self.market_making.skew_factor,
            momentum_threshold=self.momentum.threshold,
            momentum_size=self.momentum.size,
            momentum_max_size=self.momentum.max_size,
            arb_min_displacement=self.endgame.min_displacement,
            arb_prob_threshold=self.endgame.prob_threshold,
            arb_size=self.endgame.size,
            arb_max_size=self.endgame.max_size,
            warmup_period=self.strategy.warmup_period,
            mm_start=self.market_making.start_seconds,
            mm_end=self.market_making.end_seconds,
            momentum_start=self.momentum.start_seconds,
            momentum_end=self.momentum.end_seconds,
            endgame_start=self.endgame.start_seconds,
            endgame_cutoff=self.endgame.cutoff_seconds,
            edge_threshold=self.strategy.edge_threshold,
            max_loss_per_window=self.strategy.max_loss_per_window,
        )

    def to_risk_limits(self):
        """Convert to RiskLimits dataclass used by risk.py."""
        from src.risk import RiskLimits
        return RiskLimits(
            mm_max_gross_exposure=self.risk.mm_max_gross_exposure,
            mm_max_net_delta=self.risk.mm_max_net_delta,
            mm_inventory_skew_start=self.risk.mm_inventory_skew_start,
            dir_max_position=self.risk.dir_max_position,
            dir_max_total=self.risk.dir_max_total,
            dir_stop_loss_pct=self.risk.dir_stop_loss_pct,
            dir_take_profit_pct=self.risk.dir_take_profit_pct,
            max_loss_per_window=self.risk.max_loss_per_window,
            max_loss_per_hour=self.risk.max_loss_per_hour,
            max_loss_per_day=self.risk.max_loss_per_day,
            max_trades_per_hour=self.risk.max_trades_per_hour,
            min_trade_interval=self.risk.min_trade_interval,
            min_edge_after_fees=self.risk.min_edge_after_fees,
            taker_fee_rate=self.risk.taker_fee_rate,
            maker_rebate_rate=self.risk.maker_rebate_rate,
        )

    def to_signal_config(self):
        """Convert to SignalConfig dataclass used by signals.py."""
        from src.signals import SignalConfig as SigCfg
        return SigCfg(
            trend_weight=self.signal.trend_weight,
            flow_weight=self.signal.flow_weight,
            micro_weight=self.signal.micro_weight,
            reversion_weight=self.signal.reversion_weight,
            convergence_min_signals=self.signal.convergence_min_signals,
            convergence_threshold=self.signal.convergence_threshold,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

config = BotConfig()
