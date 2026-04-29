from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from trading_env import DataDrivenTradingEnv, EnvConfig, Slot, _first_price, _quote_bounds


SAC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/Users/Master/白马湖/DATA")
DEFAULT_FORECAST_FILE = Path("/Users/Master/白马湖/Trade/predict_price.csv")
DEFAULT_OUTPUT_ROOT = SAC_DIR / "baseline_artifacts"
DEFAULT_MONTHS = ["2025-12", "2026-01"]
EPS = 1e-9


def parse_months(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def select_episode_indices(env: DataDrivenTradingEnv, months: list[str]) -> list[int]:
    month_to_index = {episode[0].date[:7]: idx for idx, episode in enumerate(env.episodes)}
    missing = [month for month in months if month not in month_to_index]
    if missing:
        raise ValueError(f"缺少月份: {missing}")
    return [month_to_index[month] for month in months]


def action_from_order(env: DataDrivenTradingEnv, side: str, quote_price: float, quantity: float) -> np.ndarray:
    if side == "hold" or quantity <= EPS:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    slot = env._slot()
    low, high = _quote_bounds(slot, env.cfg)
    max_quantity = max(env._max_side_quantity(slot, side), EPS)
    side_signal = 1.0 if side == "buy" else -1.0
    price_signal = 2.0 * (quote_price - low) / max(high - low, EPS) - 1.0
    quantity_signal = 2.0 * min(max(quantity / max_quantity, 0.0), 1.0) - 1.0
    return np.array(
        [
            np.clip(side_signal, -1.0, 1.0),
            np.clip(price_signal, -1.0, 1.0),
            np.clip(quantity_signal, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def choose_baseline_action(
    env: DataDrivenTradingEnv,
    threshold: float,
    edge_scale: float,
    max_fraction: float,
    flatten_start: float,
    flatten_fraction: float,
    aggressiveness: float,
) -> np.ndarray:
    slot: Slot = env._slot()
    total_steps = max(len(env._slots()), 1)
    progress = env.step_index / max(total_steps - 1, 1)
    net = float(env.net_volume)

    ask1 = _first_price(slot.asks_p, slot.asks_v, slot.last_price)
    bid1 = _first_price(slot.bids_p, slot.bids_v, slot.last_price)
    ask3 = float(slot.asks_p[-1])
    bid3 = float(slot.bids_p[-1])

    # Late in the month, prioritize reducing inventory before taking new views.
    if progress >= flatten_start and abs(net) > EPS:
        side = "sell" if net > 0 else "buy"
        max_quantity = env._max_side_quantity(slot, side)
        quantity = min(abs(net) * flatten_fraction, max_quantity)
        quote = bid3 if side == "sell" else ask3
        return action_from_order(env, side, quote, quantity)

    buy_edge = float(slot.forecast - ask1)
    sell_edge = float(bid1 - slot.forecast)
    if buy_edge > threshold:
        side = "buy"
        quote = ask1 + aggressiveness * max(ask3 - ask1, 0.0)
        fraction = min(max_fraction, max(buy_edge / max(edge_scale, EPS), 0.05))
    elif sell_edge > threshold:
        side = "sell"
        quote = bid1 - aggressiveness * max(bid1 - bid3, 0.0)
        fraction = min(max_fraction, max(sell_edge / max(edge_scale, EPS), 0.05))
    else:
        return action_from_order(env, "hold", slot.last_price, 0.0)

    max_quantity = env._max_side_quantity(slot, side)
    return action_from_order(env, side, quote, fraction * max_quantity)


def run_month(env: DataDrivenTradingEnv, episode_index: int, args: argparse.Namespace) -> dict[str, Any]:
    obs, reset_info = env.reset(options={"episode_index": episode_index})
    _ = obs
    done = False
    trade_pnl = 0.0
    steps = 0
    last_info: dict[str, Any] = {}
    while not done:
        action = choose_baseline_action(
            env,
            threshold=args.threshold,
            edge_scale=args.edge_scale,
            max_fraction=args.max_fraction,
            flatten_start=args.flatten_start,
            flatten_fraction=args.flatten_fraction,
            aggressiveness=args.aggressiveness,
        )
        _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        trade_pnl += float(info.get("trade_pnl", 0.0))
        last_info = info
        steps += 1

    final_net = float(last_info.get("net_rolling_volume", 0.0))
    return {
        "month": str(reset_info.get("episode_month", "")),
        "steps": steps,
        "trade_pnl": trade_pnl,
        "constraint_cost": abs(final_net),
        "final_net": final_net,
        "buy_volume": float(last_info.get("rolling_buy_volume", 0.0)),
        "sell_volume": float(last_info.get("rolling_sell_volume", 0.0)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {"months": float(len(rows))}
    if not rows:
        return out
    for key in ["trade_pnl", "constraint_cost", "final_net", "buy_volume", "sell_volume", "steps"]:
        out[f"{key}_mean"] = float(np.mean([float(row[key]) for row in rows]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple forecast-vs-book baseline strategy.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--forecast-file", default=str(DEFAULT_FORECAST_FILE))
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--start-date", default="2025-07-01")
    parser.add_argument("--end-date", default="2026-01-31")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--edge-scale", type=float, default=50.0)
    parser.add_argument("--max-fraction", type=float, default=0.35)
    parser.add_argument("--flatten-start", type=float, default=0.7)
    parser.add_argument("--flatten-fraction", type=float, default=1.0)
    parser.add_argument("--aggressiveness", type=float, default=0.5)
    parser.add_argument("--enforce-balance-corridor", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    months = parse_months(args.months)
    output_dir = Path(args.output_root) / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        data_root=Path(args.data_root),
        forecast_file=Path(args.forecast_file),
        start_date=args.start_date,
        end_date=args.end_date,
        random_reset=False,
        seed=args.seed,
        enforce_balance_corridor=args.enforce_balance_corridor,
        balance_feasibility_penalty=0.0,
        close_reward_scale=0.0,
        inventory_penalty_scale=0.0,
        expansion_penalty_scale=0.0,
        late_penalty_scale=0.0,
        terminal_penalty_scale=0.0,
    )
    env = DataDrivenTradingEnv(cfg=cfg)
    indices = select_episode_indices(env, months)
    rows = [run_month(env, idx, args) for idx in indices]
    env.close()

    csv_path = output_dir / "baseline_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "months": months,
        "params": vars(args),
        "results": rows,
        "summary": summarize(rows),
        "csv_path": str(csv_path),
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
