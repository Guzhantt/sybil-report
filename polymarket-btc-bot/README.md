# Polymarket BTC 5-Minute Trading Bot v2.0

一个专业级的 Polymarket **"BTC Up or Down 5m"** 系列二元市场自动交易机器人。

采用三模式混合策略：**做市 + 动量突破 + 末尾套利**，旨在通过多维信号融合和严格风控实现稳定盈利。

## 市场说明

Polymarket 每 5 分钟自动生成一个 BTC 二元市场：
- **结算规则**: 5 分钟窗口结束时 BTC ≥ 开始价格 → "Up" 赢，否则 "Down" 赢
- **数据源**: [Chainlink BTC/USD](https://data.chain.link/streams/btc-usd)
- **最小下单**: 5 USDC | **Tick Size**: 0.01
- **费率**: Taker 7%，Maker Rebate 20%（净 5.6%）
- **系列**: `btc-up-or-down-5m`（每天 288 个窗口）

## 策略架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-MINUTE WINDOW LIFECYCLE                      │
├─────────┬──────────────────┬─────────────────┬─────────────────┤
│  WARMUP │  MARKET MAKING   │    MOMENTUM     │  ENDGAME ARB    │
│  0-10s  │    10-180s       │    60-240s      │   240-285s      │
│         │  Earn spread     │  Strong signals │  Exploit stale  │
│  Wait   │  Both sides      │  Directional    │  prices at      │
│         │  Inventory mgmt  │  Multi-signal   │  near-certain   │
│         │                  │  convergence    │  resolution     │
├─────────┴──────────────────┴─────────────────┴─────────────────┤
│                        285-300s: NO TRADING                       │
└─────────────────────────────────────────────────────────────────┘
```

### Mode 1: Market Making (做市)

在窗口早期同时挂买卖单，赚取 bid-ask spread：
- 双边挂单，根据信号偏向性微调价格
- 库存管理：净头寸过大时自动 skew 报价
- 利用 Maker Rebate（20%）降低实际费用
- 波动率自适应：高波加宽 spread，低波收紧

### Mode 2: Momentum (动量突破)

当多个独立信号**同时确认**同一方向时入场：
- 4 类信号融合：趋势 + 资金流 + 微结构 + 均值回归
- 信号置信度门槛：只在高确信度时出手
- 使用 IOC 订单快速执行
- Kelly 仓位计算（保守 25%）

### Mode 3: Endgame Arbitrage (末尾套利)

在窗口最后 60 秒，如果 BTC 已大幅偏离开盘价，结果接近确定：
- 用正态分布 CDF + 时间衰减计算理论概率
- 如果市场价与理论概率差距 > 8%，入场
- 这是**最高胜率**模式（接近确定性交易）
- 风险：流动性可能枯竭、报价可能已经调整

## 信号融合引擎

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL FUSION ENGINE                           │
├─────────────────┬───────────────────────────────────────────────┤
│  TREND (30%)    │  5m/3m/1m momentum, VWAP deviation, OBV      │
│  FLOW (30%)     │  Large trade bias, trade flow imbalance       │
│  MICRO (25%)    │  Orderbook imbalance, spread dynamics         │
│  REVERSION(15%) │  RSI contrarian (extreme overbought/sold)     │
├─────────────────┴───────────────────────────────────────────────┤
│  → Sigmoid normalization → Weighted aggregation → Time decay    │
│  → Convergence detection → Confidence score                     │
└─────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
polymarket-btc-bot/
├── main.py                    # 主入口 (trading / backtest / status)
├── config.py                  # 统一配置 (从 .env 加载)
├── requirements.txt           # Python 依赖
├── .env.example               # 配置模板 (~50 个参数)
├── .gitignore
└── src/
    ├── __init__.py
    ├── market_finder.py       # 市场发现 + 实时 Orderbook 深度
    ├── price_feed.py          # Binance WS (kline + aggTrade 双流)
    ├── signals.py             # 多维信号融合引擎
    ├── strategy.py            # 三模式策略 (MM / Momentum / Endgame)
    ├── risk.py                # 风控 (delta管理 / 止损 / 限频 / 断路器)
    ├── polymarket_client.py   # CLOB 客户端 (批量/改单/成交追踪)
    ├── trader.py              # 交易编排器 (窗口生命周期管理)
    └── backtest.py            # 回测引擎 (历史验证)
```

## 快速开始

### 1. 安装

```bash
cd polymarket-btc-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env - 默认 DRY_RUN=true，无需钱包即可运行
```

### 3. 运行

```bash
# Paper Trading（推荐先用这个）
python main.py --dry-run

# 回测历史数据
python main.py --backtest

# Live Trading（需要完整钱包配置）
python main.py --live

# 查看配置状态
python main.py --status

# Debug 模式
python main.py --dry-run --log-level DEBUG
```

### 4. 回测

```bash
# 使用 .env 中的回测参数
python main.py --backtest

# 或直接运行回测模块
python -m src.backtest --start 2026-05-01 --days 14 --balance 1000
```

## 风险控制层

```
┌─────────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT                               │
├─────────────────────────────────────────────────────────────────┤
│  Per-Trade:                                                      │
│    • Max single position: 50 USDC                                │
│    • Stop-loss: -20% | Take-profit: +40%                         │
│    • Min edge after fees: 1%                                     │
├─────────────────────────────────────────────────────────────────┤
│  Market Making:                                                   │
│    • Max gross exposure: 60 USDC                                 │
│    • Max net delta: 30 USDC                                      │
│    • Auto-skew at 15 USDC imbalance                              │
│    • Auto-hedge at 80% of max delta                              │
├─────────────────────────────────────────────────────────────────┤
│  Global:                                                          │
│    • Window loss limit: -25 USDC (circuit breaker)               │
│    • Hourly loss limit: -80 USDC                                 │
│    • Daily loss limit: -200 USDC                                 │
│    • Max 60 trades/hour                                          │
│    • Min 2s between trades                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 盈利逻辑分析

| 模式 | 边际来源 | 预期胜率 | 频率 |
|------|----------|----------|------|
| 做市 | Bid-Ask spread + Maker rebate | ~50% (靠spread赚) | 高 |
| 动量 | 多信号融合的信息优势 | ~53-55% | 中 |
| 末尾套利 | 窗口末尾的确定性溢价 | ~65-75% | 低 |

**关键洞察**: 
- 做市模式不依赖预测方向，赚的是spread（即使50%胜率也可能盈利，因为有maker rebate）
- 末尾套利只在高确信度时出手，虽然频率低但胜率极高
- 动量交易作为补充，只在多信号强收敛时才触发

## 参数调优指南

### 保守设置（推荐新手）
```env
MM_SPREAD=0.04
MOM_THRESHOLD=0.7
ARB_MIN_DISPLACEMENT=0.005
ARB_PROB_THRESHOLD=0.12
DIR_MAX_POSITION=30
```

### 激进设置（高流动性时期）
```env
MM_SPREAD=0.02
MOM_THRESHOLD=0.5
ARB_MIN_DISPLACEMENT=0.002
ARB_PROB_THRESHOLD=0.06
DIR_MAX_POSITION=80
```

## 注意事项

**风险提示**:
- 加密市场波动剧烈，二元市场有归零风险
- 做市存在库存风险（市场单边走时可能亏损）
- 策略回测 ≠ 实盘表现（滑点、延迟、流动性）
- 7% taker fee 是巨大的摩擦成本
- **永远不要投入超过你能承受损失的资金**

**安全**:
- 私钥不要提交 git（.gitignore 已排除 .env）
- 使用独立的交易钱包
- 先用 DRY_RUN=true 跑至少 24 小时观察

## 依赖

- Python 3.11+
- [py-clob-client](https://github.com/Polymarket/py-clob-client) - Polymarket CLOB
- [websockets](https://websockets.readthedocs.io/) - Binance 实时数据
- [httpx](https://www.python-httpx.org/) - 异步 HTTP
- [numpy](https://numpy.org/) - 数值计算
- [python-dotenv](https://github.com/theskumar/python-dotenv) - 配置管理

## License

MIT
