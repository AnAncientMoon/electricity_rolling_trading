# electricity_rolling_trading

浙江电力滚动撮合交易强化学习实验项目。

本仓库基于历史滚动撮合聚合成交数据和预测电价文件，构建一个可训练的本地交易环境，让强化学习模型学习输出：

```text
方向: buy / sell / hold
报价: quote price
电量: order quantity
```

当前代码分为两条实验线：

```text
PPO: 早期基于 Stable-Baselines3 PPO 的训练版本
SAC: 后续更贴合月末滚搓买卖相抵约束的 Lagrangian SAC 版本
```

## 项目结构

```text
electricity_rolling_trading/
  README.md

  PPO/
    environment_feedback.py      PPO 使用的交易环境
    train_ppo.py                 PPO 训练入口

  SAC/
    trading_env.py               SAC 使用的交易环境
    train_lagrangian_sac.py      Strict Lagrangian SAC 训练入口
    baseline_strategy.py         规则 baseline 策略
```

## 数据要求

代码默认读取本地数据，不随仓库一起提交。

### DATA 文件夹

`DATA` 文件夹中应包含按日期整理的 Excel 文件。每个 Excel 至少需要以下列：

| 列名 | 含义 |
|---|---|
| `标的日开始日期` | 标的日日期 |
| `分时段类型` | 小时时段，例如 `08:00-09:00` |
| `总交易量` | 该时段成交总量 |
| `最高价` | 最高成交价 |
| `最低价` | 最低成交价 |
| `加权价格` | 加权平均成交价 |
| `中位数价格` | 中位数成交价 |

### 预测电价文件

预测电价文件应为 CSV，至少包含：

| 列名 | 含义 |
|---|---|
| `time` | 小时级时间戳 |
| `predicted_value` | 预测电价，作为模型观测输入 |
| `true_value` | 真实出清价，用于 reward 结算 |

示例：

```csv
time,predicted_value,true_value
2025-07-01 00:00:00,320.5,318.2
2025-07-01 01:00:00,315.0,309.8
```

## 环境建模思路

当前没有真实逐笔订单簿，因此环境使用聚合成交统计重建近似订单簿。

每个标的小时有：

```text
V      = 总交易量
p_min  = 最低价
p_max  = 最高价
p_w    = 加权价格
p_med  = 中位数价格
```

环境用 Beta 分布在 `[p_min, p_max]` 上重建价格-数量曲线：

```text
DATA 聚合统计
-> Beta 分布拟合价格分布
-> 切分为多个价格层
-> 每个价格层分配成交量
-> 得到近似 price-volume order book
```

模型可见的信息包括：

```text
预测价 predicted_value
最近成交价 last_trade_price
售方最低 3 档价格和电量
购方最高 3 档价格和电量
月内进度
当前滚搓净量
上一时段成交反馈
```

撮合时则使用完整重建订单簿：

```text
买单: 从低价吃到报价
卖单: 从高价吃到报价
成交量: 逐价格层撮合，直到订单量或可成交量耗尽
成交均价: 成交价格层的加权平均价
```

reward 使用真实出清价 `true_value` 结算：

```text
买入收益 = (true_value - 成交均价) * 成交量
卖出收益 = (成交均价 - true_value) * 成交量
```

一个 episode 对应一个自然月，一个 step 对应该月中的一个标的小时。

## PPO 版本

PPO 代码位于：

```text
PPO/
```

核心文件：

```text
PPO/environment_feedback.py
PPO/train_ppo.py
```

PPO 使用 Stable-Baselines3，action 为连续三维：

```text
action = [side_signal, price_signal, quantity_signal]
```

含义：

| 维度 | 含义 |
|---|---|
| `side_signal` | 决定买、卖、不交易 |
| `price_signal` | 映射为具体报价 |
| `quantity_signal` | 映射为具体报量 |

运行示例：

```bash
python PPO/train_ppo.py \
  --data-root /path/to/DATA \
  --forecast-file /path/to/predict_price.csv \
  --model-dir PPO/ppo_artifacts \
  --episodes 1500
```

PPO 阶段主要用于验证：

```text
环境是否能跑通
连续 action 是否可行
盘口信息是否能作为观测输入
true_value 结算 reward 是否合理
```

但在滚搓交易任务中，PPO 暴露出一个问题：

```text
月末买卖相抵是强约束，不只是普通 reward 惩罚。
```

如果只靠 terminal penalty 或手写 reward shaping，模型容易出现：

```text
赚钱但不平仓
平仓但过度保守
训练不稳定
```

因此后续主线转向 Lagrangian SAC。

## SAC 版本

SAC 代码位于：

```text
SAC/
```

核心文件：

```text
SAC/trading_env.py
SAC/train_lagrangian_sac.py
SAC/baseline_strategy.py
```

SAC 版本的目标写成约束优化问题：

```text
maximize:
    trade_pnl

constraint:
    episode 结束时 abs(net_volume) = 0
```

也就是：

```text
模型要赚钱
但月末滚搓买入量和卖出量必须相抵
```

### Strict Lagrangian SAC

普通 SAC 的 replay buffer 会把 reward 固定存进去。

但 Lagrangian SAC 中，约束权重 `lambda` 会随训练变化。同一条历史经验在不同 `lambda` 下应有不同 reward。

因此本项目没有直接调用普通 SB3 SAC，而是在 `SAC/train_lagrangian_sac.py` 中实现了一个自定义版本：

```text
ReplayBuffer 存:
  objective_reward = trade_pnl / reward_scale
  constraint_cost = terminal_abs_net_volume / constraint_scale

Critic 更新时动态计算:
  reward_for_update = objective_reward - current_lambda * constraint_cost
```

这样旧样本参与训练时，会按照当前最新 `lambda` 重新计算约束惩罚。

### Balance corridor

环境支持 `enforce_balance_corridor`。

它用于避免模型走进未来已经无法配平的位置：

```text
如果当前 net_volume 为正，未来必须有足够卖出容量才能继续买
如果当前 net_volume 为负，未来必须有足够买入容量才能继续卖
如果某个动作会导致未来一定无法配平，则截断该动作可执行电量
```

这比单纯在月末加惩罚更符合真实滚搓交易约束。

运行示例：

```bash
python SAC/train_lagrangian_sac.py \
  --data-root /path/to/DATA \
  --forecast-file /path/to/predict_price.csv \
  --model-dir SAC/constrained_sac_artifacts \
  --episodes 1500 \
  --reward-norm-mode episode_volume \
  --constraint-norm-mode episode_volume \
  --lambda-lr 500 \
  --lambda-max 50000 \
  --enforce-balance-corridor
```

## Baseline 策略

`SAC/baseline_strategy.py` 是一个规则策略，用于给强化学习模型提供对照。

它的大致逻辑是：

```text
如果预测价明显高于卖一价:
    买入

如果预测价明显低于买一价:
    卖出

月末后段:
    优先把 net_volume 往 0 收敛
```

运行示例：

```bash
python SAC/baseline_strategy.py \
  --data-root /path/to/DATA \
  --forecast-file /path/to/predict_price.csv \
  --months 2026-01 \
  --enforce-balance-corridor
```

这个 baseline 不是最终目标，但它很重要：

```text
如果强化学习模型打不过 baseline，说明模型还没有学到比简单预测价差套利更强的策略。
```

## 推荐依赖

建议使用 Python 3.10+。

基础环境需要：

```bash
pip install numpy pandas scipy openpyxl gymnasium torch tensorboard
```

如果运行 PPO，还需要：

```bash
pip install stable-baselines3
```

## 快速检查

语法检查：

```bash
python -m py_compile \
  PPO/environment_feedback.py \
  PPO/train_ppo.py \
  SAC/trading_env.py \
  SAC/train_lagrangian_sac.py \
  SAC/baseline_strategy.py
```

环境 smoke test：

```bash
python SAC/trading_env.py
```

注意：smoke test 需要本地存在 DATA 和预测价格文件。

## 当前研究结论

项目路线可以概括为：

```text
聚合成交统计
-> Beta 重建近似订单簿
-> 暴露政策可见三档盘口
-> 连续 action 输出方向、报价、电量
-> 按重建订单簿撮合
-> 用 true_value 结算收益
-> PPO 早期探索
-> Lagrangian SAC 处理月末相抵约束
-> baseline 对照策略评估
```

阶段性判断：

```text
PPO 适合早期验证环境和 action 设计。
Lagrangian SAC 更适合当前带月末相抵约束的滚搓交易任务。
baseline 是强参照线，后续模型应重点学习比 baseline 更细的报价和报量调整。
```
