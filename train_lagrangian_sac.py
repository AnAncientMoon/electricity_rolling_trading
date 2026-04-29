from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 torch/tensorboard。") from exc

from trading_env import DataDrivenTradingEnv, EnvConfig, get_episode_months


SAC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SAC_DIR / "constrained_sac_artifacts"
DEFAULT_DATA_ROOT = Path("/Users/Master/白马湖/DATA")
DEFAULT_FORECAST_FILE = "/Users/Master/白马湖/Trade/predict_price.csv"
TRAIN_MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]
VALIDATION_MONTHS = ["2025-12"]
TEST_MONTHS = ["2026-01"]
DEFAULT_START_DATE = "2025-07-01"
DEFAULT_END_DATE = "2026-01-31"
EPS = 1e-6


@dataclass
class LagrangeState:
    value: float = 0.0
    lr: float = 0.05
    target: float = 0.0
    max_value: float = 1000.0

    def update(self, normalized_constraint_cost: float) -> float:
        delta = self.lr * (normalized_constraint_cost - self.target)
        self.value = float(np.clip(self.value + delta, 0.0, self.max_value))
        return self.value


@dataclass
class TrainConfig:
    data_root: Path
    forecast_file: Path
    model_dir: Path
    train_months: list[str]
    validation_months: list[str]
    test_months: list[str]
    start_date: str
    end_date: str
    seed: int
    episodes: int
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    gamma: float
    tau: float
    hidden_size: int
    gradient_steps: int
    random_steps: int
    trade_reward_scale: float
    constraint_norm: float
    reward_norm_mode: str
    constraint_norm_mode: str
    initial_lambda: float
    lambda_lr: float
    lambda_max: float
    constraint_target: float
    best_constraint_tolerance: float
    enforce_balance_corridor: bool
    log_every: int
    step_log_every: int
    eval_every: int
    checkpoint_every: int
    skip_final_eval: bool


class RunningNormalizer:
    def __init__(self, shape: tuple[int, ...], clip: float = 10.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = float(clip)

    def update(self, value: np.ndarray) -> None:
        x = np.asarray(value, dtype=np.float64)
        batch_mean = x
        batch_var = np.zeros_like(x)
        batch_count = 1.0

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, EPS)
        self.count = total_count

    def normalize(self, value: np.ndarray) -> np.ndarray:
        x = (np.asarray(value, dtype=np.float64) - self.mean) / np.sqrt(self.var + EPS)
        return np.clip(x, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "clip": float(self.clip),
        }


class ComponentReplayBuffer:
    """
    Stores reward components instead of a pre-computed Lagrangian reward.

    During SAC updates we recompute:
        reward = objective_reward - current_lambda * constraint_cost

    This avoids the ordinary-SAC bug where old samples keep rewards produced by
    old lambda values.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.objective_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.constraint_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        objective_reward: float,
        constraint_cost: float,
    ) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.objective_rewards[self.pos] = float(objective_reward)
        self.constraint_costs[self.pos] = float(constraint_cost)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
            "objective_rewards": torch.as_tensor(self.objective_rewards[indices], device=device),
            "constraint_costs": torch.as_tensor(self.constraint_costs[indices], device=device),
        }


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20.0, 2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + EPS)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self(obs_t)
            if deterministic:
                action = torch.tanh(mean)
            else:
                action, _ = self.sample(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ConstrainedSACAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: TrainConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.actor = SquashedGaussianActor(obs_dim, action_dim, cfg.hidden_size).to(device)
        self.q1 = QNetwork(obs_dim, action_dim, cfg.hidden_size).to(device)
        self.q2 = QNetwork(obs_dim, action_dim, cfg.hidden_size).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim, cfg.hidden_size).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim, cfg.hidden_size).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.learning_rate)
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.act(obs, self.device, deterministic)

    def update(self, batch: dict[str, torch.Tensor], lambda_value: float) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        objective_rewards = batch["objective_rewards"]
        constraint_costs = batch["constraint_costs"]

        lambda_t = torch.as_tensor(lambda_value, dtype=torch.float32, device=self.device)
        rewards = objective_rewards - lambda_t * constraint_costs

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            next_q = torch.min(self.q1_target(next_obs, next_actions), self.q2_target(next_obs, next_actions))
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * (next_q - self.alpha.detach() * next_log_probs)

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        sampled_actions, log_probs = self.actor.sample(obs)
        min_q = torch.min(self.q1(obs, sampled_actions), self.q2(obs, sampled_actions))
        actor_loss = (self.alpha.detach() * log_probs - min_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * param.data)

        return {
            "critic_loss": float(q_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "mean_dynamic_reward": float(rewards.mean().detach().cpu()),
            "mean_objective_reward": float(objective_rewards.mean().detach().cpu()),
            "mean_constraint_cost": float(constraint_costs.mean().detach().cpu()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.q_opt.state_dict(),
            "alpha_optimizer": self.alpha_opt.state_dict(),
            "target_entropy": self.target_entropy,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jsonable_config(cfg: TrainConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["data_root"] = str(cfg.data_root)
    data["forecast_file"] = str(cfg.forecast_file)
    data["model_dir"] = str(cfg.model_dir)
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def log_metric_group(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def parse_months(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def episode_market_volume(env: DataDrivenTradingEnv) -> float:
    return float(max(sum(slot.total_volume for slot in env._slots()), EPS))


def reward_scale_for_episode(env: DataDrivenTradingEnv, cfg: TrainConfig) -> float:
    if cfg.reward_norm_mode == "episode_volume":
        return episode_market_volume(env)
    return float(max(cfg.trade_reward_scale, EPS))


def constraint_scale_for_episode(env: DataDrivenTradingEnv, cfg: TrainConfig) -> float:
    if cfg.constraint_norm_mode == "episode_volume":
        return episode_market_volume(env)
    return float(max(cfg.constraint_norm, EPS))


def save_checkpoint(
    path: Path,
    agent: ConstrainedSACAgent,
    normalizer: RunningNormalizer,
    lagrange: LagrangeState,
    cfg: TrainConfig,
    obs_dim: int,
    action_dim: int,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "normalizer": normalizer.state_dict(),
            "lagrange": asdict(lagrange),
            "train_config": jsonable_config(cfg),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "metadata": metadata,
        },
        path,
    )


def select_episode_indices(episode_months: list[str], target_months: list[str]) -> list[int]:
    month_to_index = {month: index for index, month in enumerate(episode_months)}
    missing = [month for month in target_months if month not in month_to_index]
    if missing:
        raise ValueError(f"缺少指定月份 episode: {missing}")
    return [month_to_index[month] for month in target_months]


def make_env_config(cfg: TrainConfig, seed: int, random_reset: bool) -> EnvConfig:
    return EnvConfig(
        data_root=cfg.data_root,
        forecast_file=cfg.forecast_file,
        seed=seed,
        random_reset=random_reset,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        enforce_balance_corridor=cfg.enforce_balance_corridor,
        balance_feasibility_penalty=0.0,
        close_reward_scale=0.0,
        inventory_penalty_scale=0.0,
        expansion_penalty_scale=0.0,
        late_penalty_scale=0.0,
        terminal_penalty_scale=0.0,
    )


def run_evaluation(
    agent: ConstrainedSACAgent,
    normalizer: RunningNormalizer,
    env_cfg: EnvConfig,
    episode_indices: list[int],
    cfg: TrainConfig,
) -> dict[str, float]:
    env = DataDrivenTradingEnv(cfg=env_cfg, episode_indices=episode_indices)
    results: list[dict[str, float]] = []
    for _ in episode_indices:
        obs_raw, info = env.reset()
        reward_scale = reward_scale_for_episode(env, cfg)
        constraint_scale = constraint_scale_for_episode(env, cfg)
        done = False
        trade_pnl = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
        final_net = 0.0
        steps = 0
        while not done:
            obs = normalizer.normalize(obs_raw)
            action = agent.act(obs, deterministic=True)
            next_obs_raw, _base_reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            trade_pnl += float(step_info.get("trade_pnl", 0.0))
            buy_volume = float(step_info.get("rolling_buy_volume", 0.0))
            sell_volume = float(step_info.get("rolling_sell_volume", 0.0))
            final_net = float(step_info.get("net_rolling_volume", 0.0))
            obs_raw = next_obs_raw
            steps += 1
        constraint_cost = abs(final_net)
        results.append(
            {
                "trade_pnl": trade_pnl,
                "objective_reward": trade_pnl / reward_scale,
                "constraint_cost": constraint_cost,
                "constraint_cost_normalized": constraint_cost / constraint_scale,
                "final_net": final_net,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "steps": float(steps),
                "reward_scale": reward_scale,
                "constraint_scale": constraint_scale,
            }
        )
    env.close()
    return {
        f"{key}_mean": float(np.mean([row[key] for row in results]))
        for key in results[0]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict Lagrangian SAC with dynamic reward recomputation.")
    parser.add_argument("--config-file", default=None, help="可选 JSON 配置文件；命令行参数会覆盖其中的同名字段。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--forecast-file", default=str(DEFAULT_FORECAST_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--run-name", default=None, help="输出子文件夹名称；不指定时自动使用时间戳。")
    parser.add_argument("--no-run-subdir", action="store_true", help="直接写入 --model-dir，不自动创建 run 子文件夹。")
    parser.add_argument("--train-months", default=",".join(TRAIN_MONTHS), help="逗号分隔的训练月份。")
    parser.add_argument("--validation-months", default=",".join(VALIDATION_MONTHS), help="逗号分隔的验证月份。")
    parser.add_argument("--test-months", default=",".join(TEST_MONTHS), help="逗号分隔的测试月份。")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--random-steps", type=int, default=1_000)
    parser.add_argument("--trade-reward-scale", type=float, default=100.0)
    parser.add_argument("--constraint-norm", type=float, default=100.0)
    parser.add_argument("--reward-norm-mode", choices=["fixed", "episode_volume"], default="fixed")
    parser.add_argument("--constraint-norm-mode", choices=["fixed", "episode_volume"], default="fixed")
    parser.add_argument("--initial-lambda", type=float, default=0.0)
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=10_000.0)
    parser.add_argument("--constraint-target", type=float, default=0.0)
    parser.add_argument("--best-constraint-tolerance", type=float, default=100.0)
    parser.add_argument("--enforce-balance-corridor", action="store_true")
    parser.add_argument("--log-every", type=int, default=1, help="每多少个 episode 打印一次训练摘要。")
    parser.add_argument("--step-log-every", type=int, default=500, help="每多少个 step 打印一次 heartbeat；设为 0 可关闭。")
    parser.add_argument("--eval-every", type=int, default=50, help="每多少个 episode 做一次验证/测试评估；设为 0 可关闭。")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="每多少个 episode 保存一次 checkpoint；设为 0 可关闭。")
    parser.add_argument("--skip-final-eval", action="store_true", help="跳过训练结束后的最终验证/测试评估，主要用于快速 smoke test。")
    pre_args, _ = parser.parse_known_args()
    if pre_args.config_file:
        config_path = Path(pre_args.config_file)
        with config_path.open("r", encoding="utf-8") as file:
            config_values = json.load(file)
        valid_keys = {action.dest for action in parser._actions}
        unknown = sorted(set(config_values) - valid_keys)
        if unknown:
            raise ValueError(f"配置文件包含未知字段: {unknown}")
        parser.set_defaults(**config_values)
    return parser.parse_args()


def _safe_run_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return safe or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_model_dir(args: argparse.Namespace) -> Path:
    root = Path(args.model_dir)
    if args.no_run_subdir:
        return root
    run_name = args.run_name
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}_ep{args.episodes}"
    return root / _safe_run_name(run_name)


def next_tensorboard_run_dir(artifact_root: Path) -> Path:
    tensorboard_root = artifact_root / "tensorboard"
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for path in tensorboard_root.iterdir():
        if not path.is_dir() or not path.name.startswith("SAC_"):
            continue
        suffix = path.name.removeprefix("SAC_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return tensorboard_root / f"SAC_{max_index + 1}"


def namespace_to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_root=Path(args.data_root),
        forecast_file=Path(args.forecast_file),
        model_dir=resolve_model_dir(args),
        train_months=parse_months(args.train_months),
        validation_months=parse_months(args.validation_months),
        test_months=parse_months(args.test_months),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        seed=args.seed,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        hidden_size=args.hidden_size,
        gradient_steps=args.gradient_steps,
        random_steps=args.random_steps,
        trade_reward_scale=args.trade_reward_scale,
        constraint_norm=args.constraint_norm,
        reward_norm_mode=args.reward_norm_mode,
        constraint_norm_mode=args.constraint_norm_mode,
        initial_lambda=args.initial_lambda,
        lambda_lr=args.lambda_lr,
        lambda_max=args.lambda_max,
        constraint_target=args.constraint_target,
        best_constraint_tolerance=args.best_constraint_tolerance,
        enforce_balance_corridor=args.enforce_balance_corridor,
        log_every=max(int(args.log_every), 1),
        step_log_every=max(int(args.step_log_every), 0),
        eval_every=max(int(args.eval_every), 0),
        checkpoint_every=max(int(args.checkpoint_every), 0),
        skip_final_eval=args.skip_final_eval,
    )


def main() -> None:
    args = parse_args()
    cfg = namespace_to_config(args)
    started_at = time.time()
    if not cfg.forecast_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {cfg.forecast_file}")
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_run_subdir:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.model_dir) / "latest_run.txt").write_text(str(cfg.model_dir), encoding="utf-8")
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[start] strict Lagrangian SAC | episodes={cfg.episodes} | device={device} | "
        f"model_dir={cfg.model_dir}",
        flush=True,
    )
    if not args.no_run_subdir:
        print(f"[output] artifact_root={Path(args.model_dir)} | run_dir={cfg.model_dir}", flush=True)
    print(f"[data] loading DATA from {cfg.data_root}", flush=True)
    train_env_cfg = make_env_config(cfg, seed=cfg.seed, random_reset=True)
    probe_env = DataDrivenTradingEnv(cfg=train_env_cfg)
    episode_months = get_episode_months(probe_env.episodes)
    train_indices = select_episode_indices(episode_months, cfg.train_months)
    validation_indices = select_episode_indices(episode_months, cfg.validation_months)
    test_indices = select_episode_indices(episode_months, cfg.test_months)
    train_lengths = [len(probe_env.episodes[idx]) for idx in train_indices]
    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "Strict Lagrangian SAC",
        "train_months": cfg.train_months,
        "validation_months": cfg.validation_months,
        "test_months": cfg.test_months,
        "all_loaded_months": episode_months,
        "train_episode_steps": train_lengths,
    }
    write_json(
        cfg.model_dir / "config.json",
        {
            "config": jsonable_config(cfg),
            "metadata": run_metadata,
        },
    )
    print(
        f"[data] loaded months={episode_months} | train_months={cfg.train_months} | "
        f"train_episode_steps={train_lengths}",
        flush=True,
    )
    del probe_env

    print("[init] building environment, replay buffer, actor and critics", flush=True)
    env = DataDrivenTradingEnv(cfg=train_env_cfg, episode_indices=train_indices)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    normalizer = RunningNormalizer((obs_dim,))
    replay = ComponentReplayBuffer(cfg.buffer_size, obs_dim, action_dim, cfg.seed)
    agent = ConstrainedSACAgent(obs_dim, action_dim, cfg, device)
    lagrange = LagrangeState(
        value=cfg.initial_lambda,
        lr=cfg.lambda_lr,
        target=cfg.constraint_target,
        max_value=cfg.lambda_max,
    )

    artifact_root = Path(args.model_dir) if not args.no_run_subdir else cfg.model_dir
    tensorboard_dir = next_tensorboard_run_dir(artifact_root)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / "tensorboard_logdir.txt").write_text(str(tensorboard_dir), encoding="utf-8")
    (artifact_root / "latest_tensorboard.txt").write_text(str(tensorboard_dir), encoding="utf-8")
    print(f"[tensorboard] run_logdir={tensorboard_dir}", flush=True)
    writer = SummaryWriter(str(tensorboard_dir))
    writer.add_text("run/config", f"```json\n{json.dumps(jsonable_config(cfg), ensure_ascii=False, indent=2)}\n```", 0)
    writer.add_text("run/metadata", f"```json\n{json.dumps(run_metadata, ensure_ascii=False, indent=2)}\n```", 0)
    metrics_path = cfg.model_dir / "episode_metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "episode",
            "steps",
            "lambda_before",
            "lambda_after",
            "objective_reward",
            "lagrangian_reward_before",
            "lagrangian_reward_after",
            "trade_pnl",
            "constraint_cost",
            "constraint_cost_normalized",
            "final_net",
            "buy_volume",
            "sell_volume",
            "reward_scale",
            "constraint_scale",
            "month",
        ],
    )
    metrics_writer.writeheader()
    eval_metric_keys = [
        "trade_pnl_mean",
        "objective_reward_mean",
        "constraint_cost_mean",
        "constraint_cost_normalized_mean",
        "final_net_mean",
        "buy_volume_mean",
        "sell_volume_mean",
        "steps_mean",
        "reward_scale_mean",
        "constraint_scale_mean",
    ]
    eval_history_path = cfg.model_dir / "eval_history.csv"
    eval_history_fields = (
        ["episode", "is_best", "best_reason"]
        + [f"validation_{key}" for key in eval_metric_keys]
        + [f"test_{key}" for key in eval_metric_keys]
    )
    with eval_history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=eval_history_fields).writeheader()
    best_eval: dict[str, Any] | None = None
    last_recorded_eval_episode: int | None = None

    def current_eval_is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> tuple[bool, str]:
        if best is None:
            return True, "first_eval"
        candidate_abs = float(candidate["validation"]["constraint_cost_mean"])
        best_abs = float(best["validation"]["constraint_cost_mean"])
        candidate_pnl = float(candidate["validation"]["trade_pnl_mean"])
        best_pnl = float(best["validation"]["trade_pnl_mean"])
        candidate_feasible = candidate_abs <= cfg.best_constraint_tolerance
        best_feasible = best_abs <= cfg.best_constraint_tolerance
        if candidate_feasible and not best_feasible:
            return True, "first_feasible_validation_constraint"
        if candidate_feasible and best_feasible and candidate_pnl > best_pnl:
            return True, "higher_validation_pnl_under_constraint"
        if (not candidate_feasible) and (not best_feasible) and candidate_abs < best_abs:
            return True, "lower_validation_constraint_violation"
        return False, "not_better"

    def record_eval(episode_value: int, validation_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
        nonlocal best_eval, last_recorded_eval_episode
        if last_recorded_eval_episode == episode_value:
            return
        last_recorded_eval_episode = episode_value
        candidate = {
            "episode": episode_value,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        is_best, reason = current_eval_is_better(candidate, best_eval)
        row = {
            "episode": episode_value,
            "is_best": int(is_best),
            "best_reason": reason,
        }
        for key in eval_metric_keys:
            row[f"validation_{key}"] = validation_metrics.get(key, "")
            row[f"test_{key}"] = test_metrics.get(key, "")
        with eval_history_path.open("a", newline="", encoding="utf-8") as file:
            writer_obj = csv.DictWriter(file, fieldnames=eval_history_fields)
            writer_obj.writerow(row)
        if is_best:
            best_metadata = {
                "episode": episode_value,
                "reason": reason,
                "best_constraint_tolerance": cfg.best_constraint_tolerance,
                "validation": validation_metrics,
                "test": test_metrics,
            }
            best_eval = best_metadata
            save_checkpoint(
                cfg.model_dir / "best_model.pt",
                agent,
                normalizer,
                lagrange,
                cfg,
                obs_dim,
                action_dim,
                best_metadata,
            )
            write_json(cfg.model_dir / "best_model_metadata.json", best_metadata)
            writer.add_scalar("best/validation_trade_pnl_mean", validation_metrics["trade_pnl_mean"], episode_value)
            writer.add_scalar("best/validation_constraint_cost_mean", validation_metrics["constraint_cost_mean"], episode_value)
            print(
                f"[best] episode={episode_value} reason={reason} "
                f"val_pnl={validation_metrics['trade_pnl_mean']:.2f} "
                f"val_abs_net={validation_metrics['constraint_cost_mean']:.2f}",
                flush=True,
            )

    obs_raw, reset_info = env.reset()
    normalizer.update(obs_raw)
    episode_reward_scale = reward_scale_for_episode(env, cfg)
    episode_constraint_scale = constraint_scale_for_episode(env, cfg)
    global_step = 0
    episode = 0
    episode_steps = 0
    episode_trade_pnl = 0.0
    episode_objective_reward = 0.0
    last_update_metrics: dict[str, float] = {}
    rng = np.random.default_rng(cfg.seed)
    eval_cfg = make_env_config(cfg, seed=cfg.seed + 1, random_reset=False)
    print("[train] loop started; progress will be printed during training", flush=True)

    try:
        while episode < cfg.episodes:
            obs = normalizer.normalize(obs_raw)
            if global_step < cfg.random_steps:
                action = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)

            next_obs_raw, _base_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if not done:
                normalizer.update(next_obs_raw)
            next_obs = normalizer.normalize(next_obs_raw)

            trade_pnl = float(info.get("trade_pnl", 0.0))
            objective_reward = trade_pnl / episode_reward_scale
            final_net = float(info.get("net_rolling_volume", 0.0))
            constraint_cost = abs(final_net) / episode_constraint_scale if done else 0.0

            replay.add(obs, action, next_obs, done, objective_reward, constraint_cost)
            episode_trade_pnl += trade_pnl
            episode_objective_reward += objective_reward
            episode_steps += 1
            global_step += 1

            if replay.size >= cfg.learning_starts:
                for _ in range(cfg.gradient_steps):
                    batch = replay.sample(cfg.batch_size, device)
                    last_update_metrics = agent.update(batch, lagrange.value)
                for key, value in last_update_metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                writer.add_scalar("lagrange/lambda_step", lagrange.value, global_step)

            if cfg.step_log_every and global_step % cfg.step_log_every == 0:
                elapsed = time.time() - started_at
                print(
                    f"[step] step={global_step} episode={episode + 1}/{cfg.episodes} "
                    f"episode_steps={episode_steps} replay={replay.size} "
                    f"lambda={lagrange.value:.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if done:
                episode += 1
                lambda_before = lagrange.value
                constraint_cost_raw = abs(final_net)
                constraint_cost_normalized = constraint_cost_raw / episode_constraint_scale
                lambda_after = lagrange.update(constraint_cost_normalized)
                lagrangian_before = episode_objective_reward - lambda_before * constraint_cost_normalized
                lagrangian_after = episode_objective_reward - lambda_after * constraint_cost_normalized
                buy_volume = float(info.get("rolling_buy_volume", 0.0))
                sell_volume = float(info.get("rolling_sell_volume", 0.0))
                market_id = str(info.get("market_id", ""))
                month = market_id[:7] if market_id else str(reset_info.get("episode_month", ""))

                row = {
                    "episode": episode,
                    "steps": episode_steps,
                    "lambda_before": lambda_before,
                    "lambda_after": lambda_after,
                    "objective_reward": episode_objective_reward,
                    "lagrangian_reward_before": lagrangian_before,
                    "lagrangian_reward_after": lagrangian_after,
                    "trade_pnl": episode_trade_pnl,
                    "constraint_cost": constraint_cost_raw,
                    "constraint_cost_normalized": constraint_cost_normalized,
                    "final_net": final_net,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "reward_scale": episode_reward_scale,
                    "constraint_scale": episode_constraint_scale,
                    "month": month,
                }
                metrics_writer.writerow(row)
                metrics_file.flush()

                writer.add_scalar("episode/objective_reward", episode_objective_reward, episode)
                writer.add_scalar("episode/lagrangian_reward_before", lagrangian_before, episode)
                writer.add_scalar("episode/lagrangian_reward_after", lagrangian_after, episode)
                writer.add_scalar("episode/trade_pnl", episode_trade_pnl, episode)
                writer.add_scalar("episode/coin_profit", episode_trade_pnl, episode)
                writer.add_scalar("episode/constraint_cost", constraint_cost_raw, episode)
                writer.add_scalar("episode/constraint_cost_normalized", constraint_cost_normalized, episode)
                writer.add_scalar("episode/final_net", final_net, episode)
                writer.add_scalar("episode/month_end_unbalance_abs", constraint_cost_raw, episode)
                writer.add_scalar("episode/buy_volume", buy_volume, episode)
                writer.add_scalar("episode/sell_volume", sell_volume, episode)
                writer.add_scalar("episode/buy_sell_gap", buy_volume - sell_volume, episode)
                writer.add_scalar("episode/reward_scale", episode_reward_scale, episode)
                writer.add_scalar("episode/constraint_scale", episode_constraint_scale, episode)
                writer.add_scalar("lagrange/lambda", lambda_after, episode)

                progress = {
                    "episode": episode,
                    "target_episodes": cfg.episodes,
                    "global_step": global_step,
                    "replay_size": replay.size,
                    "last_episode": row,
                    "lambda": lambda_after,
                    "last_update_metrics": last_update_metrics,
                    "elapsed_seconds": time.time() - started_at,
                    "tensorboard_logdir": str(tensorboard_dir),
                    "episode_metrics": str(metrics_path),
                }
                write_json(cfg.model_dir / "progress.json", progress)

                if cfg.eval_every and episode % cfg.eval_every == 0:
                    validation_snapshot = run_evaluation(
                        agent,
                        normalizer,
                        eval_cfg,
                        validation_indices,
                        cfg,
                    )
                    test_snapshot = run_evaluation(
                        agent,
                        normalizer,
                        eval_cfg,
                        test_indices,
                        cfg,
                    )
                    log_metric_group(writer, "eval/validation", validation_snapshot, episode)
                    log_metric_group(writer, "eval/test", test_snapshot, episode)
                    record_eval(episode, validation_snapshot, test_snapshot)
                    write_json(
                        cfg.model_dir / "latest_eval.json",
                        {
                            "episode": episode,
                            "validation": validation_snapshot,
                            "test": test_snapshot,
                        },
                    )
                    print(
                        f"[eval] episode={episode} "
                        f"val_pnl={validation_snapshot['trade_pnl_mean']:.2f} "
                        f"val_abs_net={validation_snapshot['constraint_cost_mean']:.2f} "
                        f"test_pnl={test_snapshot['trade_pnl_mean']:.2f} "
                        f"test_abs_net={test_snapshot['constraint_cost_mean']:.2f}",
                        flush=True,
                    )

                if cfg.checkpoint_every and episode % cfg.checkpoint_every == 0:
                    checkpoint_metadata = {
                        "episode": episode,
                        "global_step": global_step,
                        "lambda": lambda_after,
                        "last_episode": row,
                    }
                    checkpoint_path = cfg.model_dir / "checkpoints" / f"checkpoint_ep{episode:06d}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        agent,
                        normalizer,
                        lagrange,
                        cfg,
                        obs_dim,
                        action_dim,
                        checkpoint_metadata,
                    )
                    save_checkpoint(
                        cfg.model_dir / "latest_checkpoint.pt",
                        agent,
                        normalizer,
                        lagrange,
                        cfg,
                        obs_dim,
                        action_dim,
                        checkpoint_metadata,
                    )
                    print(f"[checkpoint] saved {checkpoint_path}", flush=True)

                if episode == 1 or episode % cfg.log_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"[episode] {episode}/{cfg.episodes} month={month} steps={episode_steps} "
                        f"pnl={episode_trade_pnl:.2f} final_net={final_net:.2f} "
                        f"constraint={constraint_cost_raw:.2f} lambda={lambda_after:.4f} "
                        f"objective={episode_objective_reward:.2f} "
                        f"lag_after={lagrangian_after:.2f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                obs_raw, reset_info = env.reset()
                normalizer.update(obs_raw)
                episode_reward_scale = reward_scale_for_episode(env, cfg)
                episode_constraint_scale = constraint_scale_for_episode(env, cfg)
                episode_steps = 0
                episode_trade_pnl = 0.0
                episode_objective_reward = 0.0
            else:
                obs_raw = next_obs_raw
    finally:
        metrics_file.close()
        writer.flush()
        env.close()

    if cfg.skip_final_eval:
        print("[eval] skipped final validation/test evaluation", flush=True)
        validation_metrics: dict[str, float] = {}
        test_metrics: dict[str, float] = {}
    else:
        print("[eval] running validation and test evaluation", flush=True)
        validation_metrics = run_evaluation(
            agent,
            normalizer,
            eval_cfg,
            validation_indices,
            cfg,
        )
        test_metrics = run_evaluation(
            agent,
            normalizer,
            eval_cfg,
            test_indices,
            cfg,
        )
        log_metric_group(writer, "final/validation", validation_metrics, episode)
        log_metric_group(writer, "final/test", test_metrics, episode)
        record_eval(episode, validation_metrics, test_metrics)

    model_path = cfg.model_dir / "constrained_sac_model.pt"
    save_checkpoint(
        model_path,
        agent,
        normalizer,
        lagrange,
        cfg,
        obs_dim,
        action_dim,
        {
            "episode": episode,
            "global_step": global_step,
            "lambda": lagrange.value,
            "kind": "final_model",
        },
    )
    save_checkpoint(
        cfg.model_dir / "latest_checkpoint.pt",
        agent,
        normalizer,
        lagrange,
        cfg,
        obs_dim,
        action_dim,
        {
            "episode": episode,
            "global_step": global_step,
            "lambda": lagrange.value,
            "kind": "final_latest_checkpoint",
        },
    )

    summary = {
        "algorithm": "Strict Lagrangian SAC",
        "note": "Replay buffer stores objective_reward and constraint_cost; Lagrangian reward is recomputed with current lambda during critic updates.",
        "model_path": str(model_path),
        "best_model_path": str(cfg.model_dir / "best_model.pt") if (cfg.model_dir / "best_model.pt").exists() else "",
        "best_model_metadata": best_eval or {},
        "episode_metrics": str(metrics_path),
        "eval_history": str(eval_history_path),
        "tensorboard_logdir": str(tensorboard_dir),
        "target_episodes": cfg.episodes,
        "actual_trained_episodes": episode,
        "actual_trained_timesteps": global_step,
        "final_lambda": lagrange.value,
        "lagrange": asdict(lagrange),
        "train_months": cfg.train_months,
        "validation_months": cfg.validation_months,
        "test_months": cfg.test_months,
        "validation": validation_metrics,
        "test": test_metrics,
        "config": jsonable_config(cfg),
    }
    with open(cfg.model_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    writer.add_text("run/summary", f"```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```", episode)
    writer.flush()
    writer.close()

    print("[done] training complete", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
