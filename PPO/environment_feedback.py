from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


EPS = 1e-9
BOOK_LEVELS = 3

DATE_COL = "标的日开始日期"
SLOT_COL = "分时段类型"
VOLUME_COL = "总交易量"
HIGH_COL = "最高价"
LOW_COL = "最低价"
WEIGHTED_COL = "加权价格"
MEDIAN_COL = "中位数价格"
DATA_COLUMNS = {DATE_COL, SLOT_COL, VOLUME_COL, HIGH_COL, LOW_COL, WEIGHTED_COL, MEDIAN_COL}

warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

_CACHE: dict[tuple[Any, ...], list[list["Slot"]]] = {}


@dataclass
class EnvConfig:
    data_root: Path = Path("/Users/Master/白马湖/DATA")
    forecast_file: Path = Path("/Users/Master/白马湖/Trade/predict_price.csv")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    price_level_count: int = 21
    beta_concentration: float = 10.0
    min_price_step: float = 1.0
    quote_padding_levels: float = 3.0
    max_visible_depth_multiplier: float = 1.0
    hold_deadband: float = 0.2
    inventory_limit: float = 1500.0
    enforce_balance_corridor: bool = True
    balance_feasibility_penalty: float = 0.0
    close_reward_scale: float = 50.0
    inventory_penalty_scale: float = 1.0
    expansion_penalty_scale: float = 60.0
    expansion_penalty_start: float = 0.7
    late_penalty_scale: float = 50.0
    late_penalty_start: float = 0.8
    terminal_penalty_scale: float = 300.0
    monthly_balance_tolerance: float = 1e-6
    reward_scale: float = 100.0
    random_reset: bool = True
    seed: int = 42


@dataclass(frozen=True)
class Slot:
    date: str
    period: str
    hour: int
    total_volume: float
    low: float
    high: float
    weighted: float
    median: float
    forecast: float
    settlement: float
    prices: np.ndarray
    volumes: np.ndarray
    asks_p: np.ndarray
    asks_v: np.ndarray
    bids_p: np.ndarray
    bids_v: np.ndarray
    last_price: float

    @property
    def market_id(self) -> str:
        return f"{self.date}|{self.period}"

    @property
    def price_range(self) -> float:
        return float(max(self.high - self.low, EPS))


@dataclass(frozen=True)
class Execution:
    market_id: str
    side: str
    quote_price: float
    order_quantity: float
    accessible_volume: float
    filled_quantity: float
    fill_ratio: float
    execution_price: Optional[float]
    remaining_quantity: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Action:
    side: str
    quote_price: float
    order_quantity: float
    side_signal: float
    price_signal: float
    quantity_signal: float
    quantity_fraction: float
    quote_lower_bound: float
    quote_upper_bound: float
    max_order_quantity: float
    visible_depth: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slot_key(text: str) -> tuple[int, int]:
    h, m = str(text).split("-")[0].split(":")
    return int(h), int(m)


def _hour_slot(ts: pd.Timestamp) -> str:
    h = int(ts.hour)
    return f"{h:02d}:00-{'24:00' if h == 23 else f'{h + 1:02d}:00'}"


def _in_range(date_text: str, start: Optional[str], end: Optional[str]) -> bool:
    value = pd.to_datetime(date_text).date()
    return not ((start and value < pd.to_datetime(start).date()) or (end and value > pd.to_datetime(end).date()))


def _normalize(value: float, low: float, span: float) -> float:
    return float(np.clip((value - low) / max(span, EPS), EPS, 1.0 - EPS))


def _beta_curve(row: pd.Series, cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    low, high = float(row[LOW_COL]), float(row[HIGH_COL])
    if high < low:
        low, high = high, low
    span = max(high - low, EPS)
    mean = _normalize(float(row[WEIGHTED_COL]), low, span)
    median = _normalize(float(row[MEDIAN_COL]), low, span)
    center = float(np.clip(0.7 * mean + 0.3 * median, EPS, 1.0 - EPS))
    concentration = cfg.beta_concentration * (1.0 + 4.0 * abs(median - mean))
    dist = beta_dist(max(center * concentration, 0.2), max((1.0 - center) * concentration, 0.2))
    edges = np.linspace(low, high, cfg.price_level_count + 1)
    x = np.clip((edges - low) / span, 0.0, 1.0)
    prices = 0.5 * (edges[:-1] + edges[1:])
    volumes = float(row[VOLUME_COL]) * np.diff(dist.cdf(x))
    return prices.astype(float), np.maximum(volumes.astype(float), 0.0)


def _pad_levels(prices: np.ndarray, volumes: np.ndarray, reference: float, step: float, sign: int) -> tuple[np.ndarray, np.ndarray]:
    p = [float(x) for x in prices[:BOOK_LEVELS]]
    v = [float(x) for x in volumes[:BOOK_LEVELS]]
    while len(p) < BOOK_LEVELS:
        p.append(float(max(reference + sign * step * (len(p) + 1), EPS)))
        v.append(0.0)
    return np.array(p, dtype=float), np.array(v, dtype=float)


def _visible_book(prices: np.ndarray, volumes: np.ndarray, reference: float, cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diffs = np.diff(np.sort(np.unique(prices)))
    positive_diffs = diffs[diffs > EPS]
    step = float(np.median(positive_diffs)) if len(positive_diffs) else cfg.min_price_step
    step = max(step, cfg.min_price_step)
    ask = prices >= reference
    bid = prices <= reference
    asks_p, asks_v = _pad_levels(prices[ask], volumes[ask], reference, step, 1)
    bids_p, bids_v = _pad_levels(prices[bid][::-1], volumes[bid][::-1], reference, step, -1)
    return asks_p, asks_v, bids_p, bids_v


def _forecast_map(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    need = {"time", "predicted_value", "true_value"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 必须包含列: {sorted(need)}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    df["true_value"] = pd.to_numeric(df["true_value"], errors="coerce")
    df = df.dropna(subset=list(need)).sort_values("time").drop_duplicates("time", keep="last")
    return {
        f"{ts.date()}|{_hour_slot(ts)}": (float(pred), float(true))
        for ts, pred, true in zip(df["time"], df["predicted_value"], df["true_value"])
    }


def _read_day(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    missing = DATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")
    df = df.dropna(subset=list(DATA_COLUMNS)).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date.astype(str)
    return df.sort_values(SLOT_COL, key=lambda s: s.map(_slot_key)).reset_index(drop=True)


def _file_date_key(path: Path) -> tuple[pd.Timestamp, str]:
    value = pd.to_datetime(path.stem, errors="coerce")
    if pd.isna(value):
        value = pd.Timestamp.max
    return value, str(path)


def load_monthly_market_episodes(data_root: str | Path, cfg: EnvConfig) -> list[list[Slot]]:
    forecast_path = Path(cfg.forecast_file)
    key = (
        str(Path(data_root).resolve()),
        cfg.start_date,
        cfg.end_date,
        cfg.price_level_count,
        cfg.beta_concentration,
        cfg.min_price_step,
        str(forecast_path),
        forecast_path.stat().st_mtime_ns,
    )
    if key in _CACHE:
        return _CACHE[key]

    forecasts = _forecast_map(forecast_path)
    episodes: list[list[Slot]] = []
    month_slots: list[Slot] = []
    current_month: Optional[str] = None

    files = sorted(
        (p for p in Path(data_root).rglob("*.xlsx") if p.is_file() and not p.name.startswith("~$")),
        key=_file_date_key,
    )
    for file in files:
        for _, row in _read_day(file).iterrows():
            date, period = str(row[DATE_COL]), str(row[SLOT_COL])
            if not _in_range(date, cfg.start_date, cfg.end_date):
                continue

            volume = float(row[VOLUME_COL])
            low, high = sorted((float(row[LOW_COL]), float(row[HIGH_COL])))
            weighted = float(np.clip(float(row[WEIGHTED_COL]), low, high))
            median = float(np.clip(float(row[MEDIAN_COL]), low, high))
            forecast, settlement = forecasts.get(f"{date}|{period}", (weighted, weighted))
            curve_row = pd.Series({LOW_COL: low, HIGH_COL: high, WEIGHTED_COL: weighted, MEDIAN_COL: median, VOLUME_COL: volume})
            prices, volumes = _beta_curve(curve_row, cfg)
            asks_p, asks_v, bids_p, bids_v = _visible_book(prices, volumes, weighted, cfg)
            slot = Slot(
                date=date,
                period=period,
                hour=_slot_key(period)[0],
                total_volume=volume,
                low=low,
                high=high,
                weighted=weighted,
                median=median,
                forecast=forecast,
                settlement=settlement,
                prices=prices,
                volumes=volumes,
                asks_p=asks_p,
                asks_v=asks_v,
                bids_p=bids_p,
                bids_v=bids_v,
                last_price=weighted,
            )

            month = date[:7]
            if current_month is None:
                current_month = month
            if month != current_month:
                episodes.append(month_slots)
                month_slots = []
                current_month = month
            month_slots.append(slot)

    if month_slots:
        episodes.append(month_slots)
    if not episodes:
        raise ValueError("没有加载到有效的月度 episode。")
    _CACHE[key] = episodes
    return episodes


def get_episode_months(episodes: list[list[Slot]]) -> list[str]:
    return [ep[0].date[:7] for ep in episodes]


def _first_price(prices: np.ndarray, volumes: np.ndarray, fallback: float) -> float:
    for p, v in zip(prices, volumes):
        if v > EPS:
            return float(p)
    return float(fallback)


def _spread(slot: Slot, cfg: EnvConfig) -> float:
    ask = _first_price(slot.asks_p, slot.asks_v, slot.last_price)
    bid = _first_price(slot.bids_p, slot.bids_v, slot.last_price)
    return float(max(ask - bid, cfg.min_price_step))


def _price_scale(slot: Slot, cfg: EnvConfig) -> float:
    levels = np.concatenate([slot.asks_p[slot.asks_v > EPS], slot.bids_p[slot.bids_v > EPS]])
    diffs = np.diff(np.sort(np.unique(levels))) if len(levels) > 1 else np.array([])
    positive_diffs = diffs[diffs > EPS]
    step = float(np.median(positive_diffs)) if len(positive_diffs) else cfg.min_price_step
    return float(max(cfg.min_price_step, _spread(slot, cfg), step))


def _anchor(slot: Slot, side: str) -> float:
    if side == "buy":
        return _first_price(slot.asks_p, slot.asks_v, slot.last_price)
    if side == "sell":
        return _first_price(slot.bids_p, slot.bids_v, slot.last_price)
    return float(slot.last_price)


def _side_visible_depth(slot: Slot, side: str) -> float:
    if side == "buy":
        return float(slot.asks_v.sum())
    if side == "sell":
        return float(slot.bids_v.sum())
    return 0.0


def _visible_depth(slot: Slot) -> float:
    return float(slot.asks_v.sum() + slot.bids_v.sum())


def _quote_bounds(slot: Slot, cfg: EnvConfig) -> tuple[float, float]:
    visible_prices = [float(slot.last_price)]
    visible_prices.extend(float(p) for p, v in zip(slot.asks_p, slot.asks_v) if v > EPS)
    visible_prices.extend(float(p) for p, v in zip(slot.bids_p, slot.bids_v) if v > EPS)
    scale = _price_scale(slot, cfg)
    low = max(EPS, min(visible_prices) - cfg.quote_padding_levels * scale)
    high = max(visible_prices) + cfg.quote_padding_levels * scale
    if high <= low + EPS:
        high = low + cfg.min_price_step
    return float(low), float(high)


def visible_order_book_as_dict(slot: Slot) -> dict[str, float]:
    out = {"last_trade_price": float(slot.last_price)}
    for i in range(BOOK_LEVELS):
        n = i + 1
        out[f"ask_price_{n}"] = float(slot.asks_p[i])
        out[f"ask_volume_{n}"] = float(slot.asks_v[i])
        out[f"bid_price_{n}"] = float(slot.bids_p[i])
        out[f"bid_volume_{n}"] = float(slot.bids_v[i])
    return out


def execute_order(slot: Slot, side: str, quote_price: float, order_quantity: float) -> Execution:
    side = side.lower()
    if side not in {"buy", "sell", "hold"}:
        raise ValueError("side must be buy, sell, or hold.")
    if side == "hold" or order_quantity <= EPS:
        return Execution(slot.market_id, side, quote_price, order_quantity, 0.0, 0.0, 0.0, None, order_quantity)

    if side == "buy":
        mask = slot.prices <= quote_price + EPS
        prices, volumes = slot.prices[mask], slot.volumes[mask]
    else:
        mask = slot.prices >= quote_price - EPS
        prices, volumes = slot.prices[mask][::-1], slot.volumes[mask][::-1]

    accessible = float(volumes.sum())
    remaining = float(min(order_quantity, accessible))
    filled = 0.0
    notional = 0.0
    for p, v in zip(prices, volumes):
        if remaining <= EPS:
            break
        if float(v) <= EPS:
            continue
        use = min(remaining, float(v))
        filled += use
        notional += float(p) * use
        remaining -= use

    exec_price = None if filled <= EPS else notional / filled
    return Execution(
        market_id=slot.market_id,
        side=side,
        quote_price=float(quote_price),
        order_quantity=float(order_quantity),
        accessible_volume=accessible,
        filled_quantity=float(filled),
        fill_ratio=float(filled / order_quantity) if order_quantity > EPS else 0.0,
        execution_price=exec_price,
        remaining_quantity=float(max(order_quantity - filled, 0.0)),
    )


if gym is not None:

    class DataDrivenTradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, cfg: EnvConfig, episode_indices: Optional[list[int]] = None) -> None:
            super().__init__()
            self.cfg = cfg
            self.rng = np.random.default_rng(cfg.seed)
            self.episodes = load_monthly_market_episodes(cfg.data_root, cfg)
            self.episode_indices = episode_indices or list(range(len(self.episodes)))
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = spaces.Box(-1_000_000.0, 1_000_000.0, shape=(33,), dtype=np.float32)
            self._episode_cursor = 0
            self._reset_state()

        def _reset_state(self) -> None:
            self.step_index = 0
            self.net_volume = 0.0
            self.buy_volume = 0.0
            self.sell_volume = 0.0
            self.cash = 0.0
            self.prev_fill = 0.0
            self.prev_exec = 0.0
            self.prev_price = 0.0
            self.prev_volume = 0.0
            self.prev_range = 0.0
            self.remaining_buy_capacity = np.zeros(1, dtype=float)
            self.remaining_sell_capacity = np.zeros(1, dtype=float)

        def _choose_episode(self) -> int:
            if self.cfg.random_reset:
                return int(self.rng.choice(self.episode_indices))
            idx = self.episode_indices[self._episode_cursor % len(self.episode_indices)]
            self._episode_cursor += 1
            return idx

        def _slots(self) -> list[Slot]:
            return self.episodes[self.current_episode]

        def _slot(self) -> Slot:
            return self._slots()[self.step_index]

        def _max_side_quantity(self, slot: Slot, side: str) -> float:
            return _side_visible_depth(slot, side) * self.cfg.max_visible_depth_multiplier

        def _prepare_remaining_capacities(self) -> None:
            slots = self._slots()
            n = len(slots)
            self.remaining_buy_capacity = np.zeros(n + 1, dtype=float)
            self.remaining_sell_capacity = np.zeros(n + 1, dtype=float)
            for i in range(n - 1, -1, -1):
                self.remaining_buy_capacity[i] = self.remaining_buy_capacity[i + 1] + self._max_side_quantity(slots[i], "buy")
                self.remaining_sell_capacity[i] = self.remaining_sell_capacity[i + 1] + self._max_side_quantity(slots[i], "sell")

        def _feasibility_violation(self, net_volume: float, future_buy_capacity: float, future_sell_capacity: float) -> float:
            positive_violation = max(net_volume - future_sell_capacity, 0.0)
            negative_violation = max(-net_volume - future_buy_capacity, 0.0)
            return float(positive_violation + negative_violation)

        def _balance_corridor_cap(self, side: str, net_before: float, future_buy_capacity: float, future_sell_capacity: float) -> float:
            if not self.cfg.enforce_balance_corridor:
                return float("inf")
            if side == "buy":
                return float(max(future_sell_capacity - net_before, 0.0))
            if side == "sell":
                return float(max(future_buy_capacity + net_before, 0.0))
            return 0.0

        def _book_features(self, slot: Slot) -> list[float]:
            depth = max(_visible_depth(slot), 1.0)
            ask_p = [float(p) for p in slot.asks_p]
            bid_p = [float(p) for p in slot.bids_p]
            ask_v = [np.log1p(v) / np.log1p(depth) for v in slot.asks_v]
            bid_v = [np.log1p(v) / np.log1p(depth) for v in slot.bids_v]
            return [*ask_p, *ask_v, *bid_p, *bid_v]

        def _obs(self) -> np.ndarray:
            slot = self._slot()
            depth = max(_visible_depth(slot), 1.0)
            phase = 2.0 * np.pi * slot.hour / 24.0
            total_steps = max(len(self._slots()), 1)
            remaining_steps = max(total_steps - self.step_index, 0)
            last_step = max(len(self._slots()) - 1, 1)
            month_progress = self.step_index / last_step
            rem_buy = float(self.remaining_buy_capacity[self.step_index])
            rem_sell = float(self.remaining_sell_capacity[self.step_index])
            must_buy_to_flatten = max(-self.net_volume, 0.0)
            must_sell_to_flatten = max(self.net_volume, 0.0)
            required_buy_per_step = must_buy_to_flatten / max(remaining_steps, 1)
            required_sell_per_step = must_sell_to_flatten / max(remaining_steps, 1)
            base = [
                float(slot.forecast),
                float(slot.last_price),
                float(slot.forecast - slot.last_price),
                float(np.sin(phase)),
                float(np.cos(phase)),
                float(month_progress),
                float(1.0 - month_progress),
                float(remaining_steps),
                float(np.clip(self.net_volume / self.cfg.inventory_limit, -5.0, 5.0)),
                float(np.clip(rem_buy / self.cfg.inventory_limit, -20.0, 20.0)),
                float(np.clip(rem_sell / self.cfg.inventory_limit, -20.0, 20.0)),
                float(np.clip(must_buy_to_flatten / self.cfg.inventory_limit, -20.0, 20.0)),
                float(np.clip(must_sell_to_flatten / self.cfg.inventory_limit, -20.0, 20.0)),
                float(required_buy_per_step),
                float(required_sell_per_step),
                float(self.prev_price) if self.prev_price else 0.0,
                float(self.prev_price - slot.last_price) if self.prev_price else 0.0,
                float(self.prev_exec) if self.prev_exec else 0.0,
                float(self.prev_fill),
                np.log1p(self.prev_volume) / np.log1p(depth) if self.prev_volume else 0.0,
                float(self.prev_range) if self.prev_range else 0.0,
            ]
            return np.asarray([*base, *self._book_features(slot)], dtype=np.float32)

        def decode_action(self, action: np.ndarray | list[float] | tuple[float, float, float]) -> Action:
            side_signal, price_signal, quantity_signal = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).tolist()
            if side_signal > self.cfg.hold_deadband:
                side = "buy"
            elif side_signal < -self.cfg.hold_deadband:
                side = "sell"
            else:
                side = "hold"

            slot = self._slot()
            low, high = _quote_bounds(slot, self.cfg)
            price_fraction = float((price_signal + 1.0) / 2.0)
            quote = float(low + price_fraction * (high - low))
            quantity_fraction = float((quantity_signal + 1.0) / 2.0)
            visible_depth = _side_visible_depth(slot, side)
            max_quantity = self._max_side_quantity(slot, side)
            quantity = 0.0 if side == "hold" else quantity_fraction * max_quantity

            return Action(
                side=side,
                quote_price=quote,
                order_quantity=float(quantity),
                side_signal=float(side_signal),
                price_signal=float(price_signal),
                quantity_signal=float(quantity_signal),
                quantity_fraction=quantity_fraction,
                quote_lower_bound=float(low),
                quote_upper_bound=float(high),
                max_order_quantity=float(max_quantity),
                visible_depth=float(visible_depth),
            )

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict[str, Any]]:
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.current_episode = int(options["episode_index"]) if options and "episode_index" in options else self._choose_episode()
            self._reset_state()
            self._prepare_remaining_capacities()
            slot = self._slot()
            return self._obs(), {"episode_month": slot.date[:7], "delivery_date": slot.date, "episode_index": self.current_episode}

        def step(self, action: np.ndarray | list[float] | tuple[float, float, float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            slot = self._slot()
            mapped = self.decode_action(action)
            net_before = self.net_volume
            next_index = min(self.step_index + 1, len(self._slots()))
            future_buy_capacity = float(self.remaining_buy_capacity[next_index])
            future_sell_capacity = float(self.remaining_sell_capacity[next_index])
            corridor_cap = self._balance_corridor_cap(mapped.side, net_before, future_buy_capacity, future_sell_capacity)
            requested_quantity = float(mapped.order_quantity)
            executable_quantity = float(min(requested_quantity, corridor_cap))
            executable_quantity = max(executable_quantity, 0.0)
            executable = replace(mapped, order_quantity=executable_quantity)
            fill = execute_order(slot, executable.side, executable.quote_price, executable.order_quantity)

            pnl = 0.0
            if fill.execution_price is not None and fill.filled_quantity > EPS:
                if executable.side == "buy":
                    self.buy_volume += fill.filled_quantity
                    self.net_volume += fill.filled_quantity
                    pnl = (slot.settlement - fill.execution_price) * fill.filled_quantity
                elif executable.side == "sell":
                    self.sell_volume += fill.filled_quantity
                    self.net_volume -= fill.filled_quantity
                    pnl = (fill.execution_price - slot.settlement) * fill.filled_quantity

            self.cash += pnl
            progress = (self.step_index + 1) / max(len(self._slots()), 1)
            imbalance_before = abs(net_before)
            imbalance_after = abs(self.net_volume)
            trade_reward = pnl
            close_position_reward = self.cfg.close_reward_scale * max(imbalance_before - imbalance_after, 0.0)
            inventory_penalty = self.cfg.inventory_penalty_scale * imbalance_after * (progress ** 2)
            expansion = max(imbalance_after - imbalance_before, 0.0)
            expansion_penalty = 0.0
            if progress > self.cfg.expansion_penalty_start:
                expansion_progress = (progress - self.cfg.expansion_penalty_start) / max(1.0 - self.cfg.expansion_penalty_start, EPS)
                expansion_penalty = self.cfg.expansion_penalty_scale * expansion * expansion_progress
            late_penalty = 0.0
            if progress > self.cfg.late_penalty_start:
                late_progress = (progress - self.cfg.late_penalty_start) / max(1.0 - self.cfg.late_penalty_start, EPS)
                late_penalty = self.cfg.late_penalty_scale * imbalance_after * (late_progress ** 2)
            feasibility_violation = self._feasibility_violation(self.net_volume, future_buy_capacity, future_sell_capacity)
            feasibility_penalty = self.cfg.balance_feasibility_penalty * feasibility_violation
            self.prev_fill = fill.fill_ratio
            self.prev_exec = fill.execution_price or 0.0
            self.prev_price = slot.settlement
            self.prev_volume = slot.total_volume
            self.prev_range = slot.price_range
            self.step_index += 1

            done = self.step_index >= len(self._slots())
            imbalance = abs(self.net_volume)
            terminal_penalty = self.cfg.terminal_penalty_scale * imbalance_after if done else 0.0
            reward = (
                trade_reward
                + close_position_reward
                - inventory_penalty
                - expansion_penalty
                - late_penalty
                - feasibility_penalty
                - terminal_penalty
            ) / self.cfg.reward_scale
            obs = np.zeros(self.observation_space.shape, dtype=np.float32) if done else self._obs()
            info = {
                "market_id": slot.market_id,
                "decoded_action": executable.as_dict(),
                "requested_action": mapped.as_dict(),
                "action_constraints": {
                    "enforce_balance_corridor": bool(self.cfg.enforce_balance_corridor),
                    "requested_order_quantity": requested_quantity,
                    "corridor_quantity_cap": float(corridor_cap) if np.isfinite(corridor_cap) else None,
                    "quantity_was_capped": bool(executable_quantity + EPS < requested_quantity),
                },
                "feedback": fill.as_dict(),
                "visible_order_book": visible_order_book_as_dict(slot),
                "progress": float(progress),
                "forecast_price": float(slot.forecast),
                "settlement_price": float(slot.settlement),
                "trade_pnl": float(pnl),
                "trade_reward": float(trade_reward),
                "close_position_reward": float(close_position_reward),
                "inventory_penalty": float(inventory_penalty),
                "expansion_penalty": float(expansion_penalty),
                "late_penalty": float(late_penalty),
                "balance_feasibility_penalty": float(feasibility_penalty),
                "imbalance_increase": float(expansion),
                "feasibility_violation": float(feasibility_violation),
                "future_buy_capacity": float(future_buy_capacity),
                "future_sell_capacity": float(future_sell_capacity),
                "monthly_balance_penalty": float(terminal_penalty),
                "monthly_balance_violation": float(imbalance if done else 0.0),
                "monthly_balance_ok": bool((not done) or imbalance <= self.cfg.monthly_balance_tolerance),
                "rolling_buy_volume": float(self.buy_volume),
                "rolling_sell_volume": float(self.sell_volume),
                "net_rolling_volume": float(self.net_volume),
                "equity": float(self.cash),
            }
            return obs, float(reward), done, False, info

else:

    class DataDrivenTradingEnv:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            raise ImportError("gymnasium is required. Use /Users/Master/白马湖/code/.venv/bin/python.")


def main() -> None:
    cfg = EnvConfig(start_date="2025-07-01", end_date="2026-01-31")
    episodes = load_monthly_market_episodes(cfg.data_root, cfg)
    first = episodes[0][0]
    _, quote = _quote_bounds(first, cfg)
    quantity = 0.5 * _side_visible_depth(first, "buy")
    print({"months": get_episode_months(episodes), "first_market": first.market_id, "book": visible_order_book_as_dict(first)})
    print(execute_order(first, "buy", quote, quantity).as_dict())


if __name__ == "__main__":
    main()
