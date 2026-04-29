from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 stable_baselines3/gymnasium。请使用 "
        "/Users/Master/白马湖/code/.venv/bin/python 运行这个脚本。"
    ) from exc

from environment_feedback import (
    DataDrivenTradingEnv,
    EnvConfig,
    get_episode_months,
)


DEFAULT_MODEL_DIR = Path("/Users/Master/白马湖/Trade/ppo_artifacts")
DEFAULT_FORECAST_FILE = Path("/Users/Master/白马湖/Trade/predict_price.csv")
TRAIN_MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]
VALIDATION_MONTHS = ["2025-12"]
TEST_MONTHS = ["2026-01"]
DEFAULT_START_DATE = "2025-07-01"
DEFAULT_END_DATE = "2026-01-31"


class EpisodeMetricsTensorboardCallback(BaseCallback):
    """
    Log extra per-episode diagnostics to TensorBoard:
    - rollout/ep_coin_profit_mean
    - rollout/ep_month_end_unbalance_abs_mean
    - episode/ep_reward
    - episode/ep_coin_profit
    - episode/ep_month_end_unbalance_abs

    And optionally stop training after a target number of episodes.
    """

    def __init__(self, target_episodes: int | None = None) -> None:
        super().__init__()
        self.target_episodes = target_episodes
        self.completed_episodes = 0
        self._rollout_coin_profit: list[float] = []
        self._rollout_unbalance_abs: list[float] = []
        self._rollout_ep_reward: list[float] = []
        self._tb_writer = None

    def _on_rollout_start(self) -> None:
        self._rollout_coin_profit = []
        self._rollout_unbalance_abs = []
        self._rollout_ep_reward = []

    def _on_training_start(self) -> None:
        for output_format in self.logger.output_formats:
            writer = getattr(output_format, "writer", None)
            if writer is not None:
                self._tb_writer = writer
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos is None or dones is None:
            return True

        should_continue = True
        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            self.completed_episodes += 1

            ep_info = info.get("episode", {})
            episode_reward = float(ep_info.get("r", 0.0))
            episode_coin_profit = float(info.get("equity", 0.0))
            month_end_unbalance_abs = abs(float(info.get("monthly_balance_violation", 0.0)))

            self._rollout_ep_reward.append(episode_reward)
            self._rollout_coin_profit.append(episode_coin_profit)
            self._rollout_unbalance_abs.append(month_end_unbalance_abs)

            if self._tb_writer is not None:
                self._tb_writer.add_scalar("episode/ep_reward", episode_reward, self.completed_episodes)
                self._tb_writer.add_scalar("episode/ep_coin_profit", episode_coin_profit, self.completed_episodes)
                self._tb_writer.add_scalar("episode/ep_month_end_unbalance_abs", month_end_unbalance_abs, self.completed_episodes)

            if self.target_episodes is not None and self.completed_episodes >= self.target_episodes:
                should_continue = False

        return should_continue

    def _on_rollout_end(self) -> None:
        if self._rollout_ep_reward:
            self.logger.record("rollout/ep_reward_mean_by_episode", float(np.mean(self._rollout_ep_reward)))
        if self._rollout_coin_profit:
            self.logger.record("rollout/ep_coin_profit_mean", float(np.mean(self._rollout_coin_profit)))
        if self._rollout_unbalance_abs:
            self.logger.record("rollout/ep_month_end_unbalance_abs_mean", float(np.mean(self._rollout_unbalance_abs)))


def select_episode_indices(episode_months: list[str], target_months: list[str]) -> list[int]:
    month_to_index = {month: index for index, month in enumerate(episode_months)}
    missing = [month for month in target_months if month not in month_to_index]
    if missing:
        raise ValueError(f"缺少指定月份 episode: {missing}")
    return [month_to_index[month] for month in target_months]


def make_env_fn(cfg: EnvConfig, episode_indices: list[int]):
    def _init():
        return Monitor(DataDrivenTradingEnv(cfg=cfg, episode_indices=episode_indices))

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO trading model on DATA folder summaries.")
    parser.add_argument("--data-root", default="/Users/Master/白马湖/DATA")
    parser.add_argument("--forecast-file", default=str(DEFAULT_FORECAST_FILE))
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    forecast_file = Path(args.forecast_file)
    if not forecast_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {forecast_file}")

    cfg = EnvConfig(
        data_root=Path(args.data_root),
        forecast_file=forecast_file,
        seed=args.seed,
        random_reset=True,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )

    probe_env = DataDrivenTradingEnv(cfg=cfg)
    episode_months = get_episode_months(probe_env.episodes)
    train_indices = select_episode_indices(episode_months, TRAIN_MONTHS)
    validation_indices = select_episode_indices(episode_months, VALIDATION_MONTHS)
    test_indices = select_episode_indices(episode_months, TEST_MONTHS)
    max_train_episode_steps = max(len(probe_env.episodes[idx]) for idx in train_indices)
    timestep_budget = int(max_train_episode_steps * args.episodes)
    del probe_env

    train_env = VecNormalize(
        DummyVecEnv([make_env_fn(cfg=cfg, episode_indices=train_indices)]),
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    evaluation_cfg = replace(
        cfg,
        random_reset=False,
        seed=args.seed + 1,
    )
    validation_env = VecNormalize(
        DummyVecEnv([make_env_fn(cfg=evaluation_cfg, episode_indices=validation_indices)]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    test_env = VecNormalize(
        DummyVecEnv([make_env_fn(cfg=evaluation_cfg, episode_indices=test_indices)]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.003,
        vf_coef=0.8,
        max_grad_norm=0.3,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(model_dir / "tensorboard"),
    )

    episode_callback = EpisodeMetricsTensorboardCallback(target_episodes=args.episodes)
    model.learn(
        total_timesteps=timestep_budget,
        callback=episode_callback,
    )

    # Reuse the train normalization statistics for evaluation.
    validation_env.obs_rms = deepcopy(train_env.obs_rms)
    test_env.obs_rms = deepcopy(train_env.obs_rms)

    model_path = model_dir / "ppo_trade_model"
    model.save(str(model_path))
    vecnorm_path = model_dir / "vecnormalize.pkl"
    train_env.save(str(vecnorm_path))

    validation_mean_reward, validation_std_reward = evaluate_policy(
        model,
        validation_env,
        n_eval_episodes=len(validation_indices),
        deterministic=True,
    )
    test_mean_reward, test_std_reward = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=len(test_indices),
        deterministic=True,
    )

    metadata: dict[str, Any] = {
        "data_root": str(cfg.data_root),
        "forecast_file": str(cfg.forecast_file),
        "target_episodes": int(args.episodes),
        "timestep_budget": int(timestep_budget),
        "actual_trained_episodes": int(episode_callback.completed_episodes),
        "actual_trained_timesteps": int(model.num_timesteps),
        "seed": int(args.seed),
        "vecnormalize_path": str(vecnorm_path),
        "train_months": TRAIN_MONTHS,
        "validation_months": VALIDATION_MONTHS,
        "test_months": TEST_MONTHS,
        "train_episode_count": len(train_indices),
        "validation_episode_count": len(validation_indices),
        "test_episode_count": len(test_indices),
        "validation_mean_reward": float(validation_mean_reward),
        "validation_std_reward": float(validation_std_reward),
        "test_mean_reward": float(test_mean_reward),
        "test_std_reward": float(test_std_reward),
        "env_config": {
            "price_level_count": cfg.price_level_count,
            "beta_concentration": cfg.beta_concentration,
            "min_price_step": cfg.min_price_step,
            "quote_padding_levels": cfg.quote_padding_levels,
            "max_visible_depth_multiplier": cfg.max_visible_depth_multiplier,
            "hold_deadband": cfg.hold_deadband,
            "inventory_limit": cfg.inventory_limit,
            "enforce_balance_corridor": cfg.enforce_balance_corridor,
            "balance_feasibility_penalty": cfg.balance_feasibility_penalty,
            "close_reward_scale": cfg.close_reward_scale,
            "inventory_penalty_scale": cfg.inventory_penalty_scale,
            "expansion_penalty_scale": cfg.expansion_penalty_scale,
            "expansion_penalty_start": cfg.expansion_penalty_start,
            "late_penalty_scale": cfg.late_penalty_scale,
            "late_penalty_start": cfg.late_penalty_start,
            "terminal_penalty_scale": cfg.terminal_penalty_scale,
            "monthly_balance_tolerance": cfg.monthly_balance_tolerance,
            "reward_scale": cfg.reward_scale,
        },
        "ppo_config": {
            "learning_rate": 1e-4,
            "n_steps": 512,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "clip_range": 0.15,
            "ent_coef": 0.003,
            "vf_coef": 0.8,
            "max_grad_norm": 0.3,
            "policy_net_arch": {"pi": [256, 256], "vf": [256, 256]},
        },
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward_train": False,
            "norm_reward_eval": False,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
    }

    with open(model_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "model_path": str(model_path) + ".zip",
                "vecnormalize_path": str(vecnorm_path),
                "target_episodes": int(args.episodes),
                "actual_trained_episodes": int(episode_callback.completed_episodes),
                "actual_trained_timesteps": int(model.num_timesteps),
                "train_months": TRAIN_MONTHS,
                "validation_months": VALIDATION_MONTHS,
                "test_months": TEST_MONTHS,
                "validation_mean_reward": float(validation_mean_reward),
                "validation_std_reward": float(validation_std_reward),
                "test_mean_reward": float(test_mean_reward),
                "test_std_reward": float(test_std_reward),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    train_env.close()
    validation_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
