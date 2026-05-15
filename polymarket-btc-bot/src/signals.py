"""
Signal Fusion Engine - Polymarket Native.

ALL signals derived from Polymarket orderbook dynamics:
  - TREND: Up token price momentum (short/mid/long), acceleration
  - FLOW: Orderbook imbalance, imbalance velocity, large moves
  - MICRO: Spread dynamics, depth ratio, spread trend
  - REVERSION: RSI on Up token price, deviation from moving average
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalComponent:
    name: str
    raw_value: float = 0.0
    normalized: float = 0.0
    weight: float = 0.0
    category: str = ""
    is_valid: bool = True

    @property
    def weighted(self) -> float:
        return self.normalized * self.weight if self.is_valid else 0.0


@dataclass
class SignalConfig:
    trend_weight: float = 0.30
    flow_weight: float = 0.30
    micro_weight: float = 0.25
    reversion_weight: float = 0.15
    weights: dict = field(default_factory=lambda: {
        "momentum_5m": 0.35, "momentum_3m": 0.25, "momentum_1m": 0.15,
        "vwap_deviation": 0.15, "obv_slope": 0.10,
        "large_trade_bias": 0.40, "trade_flow_imbalance": 0.35, "volume_surge_boost": 0.25,
        "book_imbalance": 0.45, "imbalance_trend": 0.30, "spread_signal": 0.25,
        "rsi_contrarian": 1.0,
    })
    convergence_min_signals: int = 3
    convergence_threshold: float = 0.5
    time_decay_start: float = 200.0
    time_decay_rate: float = 0.5
    high_vol_trend_discount: float = 0.5
    low_vol_trend_boost: float = 1.3


class SignalEngine:
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self._components: list = []
        self._history: list = []

    def compute(self, price_signals: dict, micro_signals: dict, elapsed_seconds: float = 0.0) -> tuple:
        self._components = []
        cfg = self.config
        vol_regime = price_signals.get("volatility_regime", "normal")

        trend = self._process_trend(price_signals, vol_regime)
        flow = self._process_flow(price_signals)
        micro = self._process_micro(micro_signals)
        reversion = self._process_reversion(price_signals)

        trend_score = self._cat_score(trend)
        flow_score = self._cat_score(flow)
        micro_score = self._cat_score(micro)
        rev_score = self._cat_score(reversion)

        raw = (trend_score * cfg.trend_weight + flow_score * cfg.flow_weight +
               micro_score * cfg.micro_weight + rev_score * cfg.reversion_weight)

        if elapsed_seconds > cfg.time_decay_start:
            decay = cfg.time_decay_rate + (1 - cfg.time_decay_rate) * (
                1 - (elapsed_seconds - cfg.time_decay_start) / (300 - cfg.time_decay_start))
            decay = max(cfg.time_decay_rate, min(1.0, decay))
            raw = (trend_score * cfg.trend_weight * decay + flow_score * cfg.flow_weight +
                   micro_score * cfg.micro_weight + rev_score * cfg.reversion_weight)

        composite = max(-1.0, min(1.0, raw))
        confidence = self._calc_confidence(trend_score, flow_score, micro_score, rev_score, vol_regime, elapsed_seconds)

        self._history.append((composite, confidence))
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return composite, confidence

    def _process_trend(self, s: dict, vol_regime: str) -> list:
        cfg = self.config
        c = []
        mult = cfg.high_vol_trend_discount if vol_regime == "high" else (cfg.low_vol_trend_boost if vol_regime == "low" else 1.0)

        for key, lookback_name in [("momentum_5m", "momentum_5m"), ("momentum_3m", "momentum_3m"), ("momentum_1m", "momentum_1m")]:
            val = s.get(key)
            if val is not None:
                n = self._sigmoid(val, 0.8) * mult  # Scaled for probability-price momentum
                c.append(SignalComponent(name=key, raw_value=val, normalized=n, weight=cfg.weights[key], category="trend", is_valid=True))
            else:
                c.append(SignalComponent(name=key, is_valid=False, category="trend"))

        vwap = s.get("vwap_deviation")
        if vwap is not None:
            c.append(SignalComponent(name="vwap_deviation", raw_value=vwap, normalized=self._sigmoid(vwap, 1.0), weight=cfg.weights["vwap_deviation"], category="trend"))

        obv = s.get("obv_slope")
        if obv is not None:
            c.append(SignalComponent(name="obv_slope", raw_value=obv, normalized=max(-1, min(1, obv)), weight=cfg.weights["obv_slope"], category="trend"))

        self._components.extend(c)
        return c

    def _process_flow(self, s: dict) -> list:
        cfg = self.config
        c = []

        ltb = s.get("large_trade_bias", 0)
        if ltb != 0:
            c.append(SignalComponent(name="large_trade_bias", raw_value=ltb, normalized=ltb, weight=cfg.weights["large_trade_bias"], category="flow"))
        else:
            c.append(SignalComponent(name="large_trade_bias", is_valid=False, category="flow"))

        tfi = s.get("trade_flow_imbalance")
        if tfi is not None:
            c.append(SignalComponent(name="trade_flow_imbalance", raw_value=tfi, normalized=tfi, weight=cfg.weights["trade_flow_imbalance"], category="flow"))

        surge = s.get("volume_surge", False)
        if surge:
            flow_dir = tfi if tfi else 0
            sig = 0.5 * (1 if flow_dir > 0 else -1 if flow_dir < 0 else 0)
            c.append(SignalComponent(name="volume_surge_boost", raw_value=1.0, normalized=sig, weight=cfg.weights["volume_surge_boost"], category="flow"))

        self._components.extend(c)
        return c

    def _process_micro(self, s: dict) -> list:
        cfg = self.config
        c = []

        imb = s.get("imbalance_up", 0)
        if imb != 0:
            c.append(SignalComponent(name="book_imbalance", raw_value=imb, normalized=imb, weight=cfg.weights["book_imbalance"], category="micro"))

        imb_trend = s.get("imbalance_trend", 0)
        if imb_trend != 0:
            c.append(SignalComponent(name="imbalance_trend", raw_value=imb_trend, normalized=self._sigmoid(imb_trend, 0.3), weight=cfg.weights["imbalance_trend"], category="micro"))

        spread = s.get("spread_up", 0.02)
        avg_spread = s.get("avg_spread", 0.02)
        if avg_spread > 0 and spread > 0:
            ratio = spread / avg_spread
            if ratio < 0.8:
                sig = imb * 0.3
            elif ratio > 1.5:
                sig = 0.0
            else:
                sig = 0.0
            c.append(SignalComponent(name="spread_signal", raw_value=ratio, normalized=sig, weight=cfg.weights["spread_signal"], category="micro"))

        self._components.extend(c)
        return c

    def _process_reversion(self, s: dict) -> list:
        cfg = self.config
        c = []
        rsi = s.get("rsi_14")
        if rsi is not None:
            if rsi > 75:
                n = -((rsi - 75) / 25.0) * 0.6
            elif rsi < 25:
                n = ((25 - rsi) / 25.0) * 0.6
            else:
                n = 0.0
            c.append(SignalComponent(name="rsi_contrarian", raw_value=rsi, normalized=n, weight=cfg.weights["rsi_contrarian"], category="reversion"))
        else:
            c.append(SignalComponent(name="rsi_contrarian", is_valid=False, category="reversion"))
        self._components.extend(c)
        return c

    def _calc_confidence(self, trend, flow, micro, rev, vol_regime, elapsed) -> float:
        scores = [s for s in [trend, flow, micro, rev] if abs(s) > 0.05]
        if not scores:
            return 0.0

        pos = sum(1 for s in scores if s > 0.1)
        neg = sum(1 for s in scores if s < -0.1)
        agreement = max(pos, neg) / len(scores)
        strength = min(1.0, sum(abs(s) for s in scores) / len(scores) * 2)

        regime_mult = 0.6 if vol_regime == "high" else (1.2 if vol_regime == "low" else 1.0)
        time_mult = min(1.0, elapsed / 30.0) if elapsed < 30 else (0.5 if elapsed > 270 else 1.0)

        bonus = 0.2 if max(pos, neg) >= self.config.convergence_min_signals else 0.0

        conf = (agreement * 0.4 + strength * 0.4 + bonus) * regime_mult * time_mult
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _sigmoid(value: float, scale: float = 0.1) -> float:
        if scale <= 0:
            return 0.0
        return 2.0 / (1.0 + math.exp(-value / scale)) - 1.0

    @staticmethod
    def _cat_score(components: list) -> float:
        valid = [c for c in components if c.is_valid]
        if not valid:
            return 0.0
        tw = sum(c.weight for c in valid)
        return sum(c.weighted for c in valid) / tw if tw > 0 else 0.0

    def get_components(self) -> list:
        return [{"name": c.name, "normalized": round(c.normalized, 3), "category": c.category, "valid": c.is_valid} for c in self._components]
