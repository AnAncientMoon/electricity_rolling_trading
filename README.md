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


## 推荐依赖

建议使用 Python 3.10+。

基础环境需要：

```bash
pip install numpy pandas scipy openpyxl gymnasium torch tensorboard
```

如果运行 PPO版本，还需要：

```bash
pip install stable-baselines3
```
