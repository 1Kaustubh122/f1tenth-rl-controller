"""Shared Gymnasium environments for the baseline and experimental controllers."""

from __future__ import annotations

import logging
import time
from typing import Sequence
from uuid import uuid4

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32

from autodrive_race.constants import (
    ADVANCED_ACTION_DIMS,
    ADVANCED_LIDAR_BINS,
    ADVANCED_MAX_LAPS_PER_EPISODE,
    ADVANCED_OBSERVATION_SIZE,
    ADVANCED_STEERING_LEVELS,
    ADVANCED_THROTTLE_LEVELS,
    BASELINE_ACTIONS,
    BASELINE_ACTION_SIZE,
    BASELINE_LIDAR_BINS,
    BASELINE_MAX_LAPS_PER_EPISODE,
    BASELINE_OBSERVATION_SIZE,
    COLLISION_COUNT_TOPIC,
    EXPERIMENTAL_ACTION_DIMS,
    EXPERIMENTAL_LIDAR_BINS,
    EXPERIMENTAL_OBSERVATION_SIZE,
    EXPERIMENTAL_STEERING_LEVELS,
    EXPERIMENTAL_THROTTLE_LEVELS,
    LAP_COUNT_TOPIC,
    LAST_LAP_TIME_TOPIC,
    LIDAR_TOPIC,
    MAX_LIDAR_RANGE_METERS,
    MAX_SPEED_METERS_PER_SECOND,
    ROS_SPIN_TIMEOUT_SEC,
    SENSOR_WAIT_TIMEOUT_SEC,
    STEERING_COMMAND_TOPIC,
    THROTTLE_COMMAND_TOPIC,
)

LOGGER = logging.getLogger(__name__)


def preprocess_lidar_ranges(
    ranges: Sequence[float],
    target_bins: int,
    max_range_meters: float = MAX_LIDAR_RANGE_METERS,
    reduction: str = "mean",
) -> np.ndarray:
    """Normalize a LiDAR scan to a fixed-size observation vector."""

    scan = np.asarray(ranges, dtype=np.float32)
    if scan.size == 0:
        return np.zeros(target_bins, dtype=np.float32)

    scan = np.where(np.isfinite(scan), scan, max_range_meters)
    scan = np.clip(scan, 0.0, max_range_meters)
    chunks = np.array_split(scan, target_bins)
    if reduction == "mean":
        reducer = lambda chunk: float(chunk.mean())
    elif reduction == "min":
        reducer = lambda chunk: float(chunk.min())
    elif reduction == "p25":
        reducer = lambda chunk: float(np.percentile(chunk, 25))
    else:
        raise ValueError(f"Unsupported LiDAR reduction mode: {reduction}")

    reduced = np.array([reducer(chunk) if chunk.size else max_range_meters for chunk in chunks], dtype=np.float32)
    return reduced / max_range_meters


class RosRaceEnv(gym.Env):
    """Small ROS-backed base class for racing environments."""

    metadata = {"render_modes": []}

    def __init__(self, *, lidar_bins: int, train: bool, node_prefix: str, lidar_reduction: str = "mean") -> None:
        super().__init__()
        self.train = train
        self.lidar_bins = lidar_bins
        self.lidar_reduction = lidar_reduction
        self.max_range = MAX_LIDAR_RANGE_METERS
        self.max_speed = MAX_SPEED_METERS_PER_SECOND

        node_name = f"{node_prefix}_{'train' if train else 'eval'}_{uuid4().hex[:8]}"
        self.node = Node(node_name)

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            LIDAR_TOPIC,
            self.lidar_callback,
            100,
        )
        self.collision_sub = self.node.create_subscription(
            Int32,
            COLLISION_COUNT_TOPIC,
            self.collision_callback,
            10,
        )
        self.throttle_pub = self.node.create_publisher(Float32, THROTTLE_COMMAND_TOPIC, 10)
        self.steering_pub = self.node.create_publisher(Float32, STEERING_COMMAND_TOPIC, 10)

        self.lidar_data: np.ndarray | None = None
        self.collision_count = 0
        self.last_collision_count = 0

    def lidar_callback(self, msg: LaserScan) -> None:
        self.lidar_data = preprocess_lidar_ranges(
            msg.ranges,
            self.lidar_bins,
            self.max_range,
            reduction=self.lidar_reduction,
        )

    def collision_callback(self, msg: Int32) -> None:
        self.collision_count = int(msg.data)

    def publish_control(self, throttle: float, steering: float) -> None:
        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        steering_msg = Float32()
        steering_msg.data = float(steering)

        self.throttle_pub.publish(throttle_msg)
        self.steering_pub.publish(steering_msg)

    def spin_once(self, timeout_sec: float = ROS_SPIN_TIMEOUT_SEC) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def wait_for_lidar(self, timeout_sec: float = SENSOR_WAIT_TIMEOUT_SEC) -> None:
        deadline = time.monotonic() + timeout_sec
        while self.lidar_data is None and time.monotonic() < deadline:
            self.spin_once()

        if self.lidar_data is None:
            LOGGER.warning(
                "No LiDAR scan received within %.1f s; using a zero observation until data arrives.",
                timeout_sec,
            )

    def current_lidar(self) -> np.ndarray:
        if self.lidar_data is None:
            return np.zeros(self.lidar_bins, dtype=np.float32)
        return self.lidar_data.astype(np.float32, copy=False)

    def consume_collision_event(self) -> bool:
        if self.collision_count < self.last_collision_count:
            self.last_collision_count = self.collision_count

        collision = self.collision_count > self.last_collision_count
        self.last_collision_count = self.collision_count
        return collision

    def close(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None


class BaselineRaceEnv(RosRaceEnv):
    """Mainline environment aligned to the saved PPO artifacts."""

    def __init__(self, train: bool = True) -> None:
        super().__init__(lidar_bins=BASELINE_LIDAR_BINS, train=train, node_prefix="f110_baseline_env")

        self.last_lap_sub = self.node.create_subscription(
            Float32,
            LAST_LAP_TIME_TOPIC,
            self.lap_time_callback,
            10,
        )
        self.lap_count_sub = self.node.create_subscription(
            Int32,
            LAP_COUNT_TOPIC,
            self.lap_count_callback,
            10,
        )

        self.action_space = spaces.Discrete(BASELINE_ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BASELINE_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.actions = BASELINE_ACTIONS
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.total_lap_count = 0
        self.last_lap_count = 0
        self.episode_start_lap_count = 0

    def lap_time_callback(self, msg: Float32) -> None:
        self.last_lap_time = float(msg.data)

    def lap_count_callback(self, msg: Int32) -> None:
        self.total_lap_count = int(msg.data)

    def step(self, action: int | np.ndarray):
        action_index = int(np.asarray(action).item())
        throttle, steering = self.actions[action_index]

        self.publish_control(throttle, steering)
        self.spin_once()

        collision = self.consume_collision_event()
        reward = -50.0 if collision else 1.0 + 0.01 * throttle * self.max_speed
        terminated = collision

        if self.total_lap_count < self.last_lap_count:
            self.last_lap_count = self.total_lap_count
            self.episode_start_lap_count = self.total_lap_count

        lap_completed = self.total_lap_count > self.last_lap_count
        if not collision and lap_completed:
            self.last_lap_count = self.total_lap_count
            if self.last_lap_time is not None and self.last_lap_time > 0.0:
                if self.best_lap_time is None or self.last_lap_time < self.best_lap_time:
                    self.best_lap_time = self.last_lap_time
                    reward += 50.0

        laps_completed = max(0, self.total_lap_count - self.episode_start_lap_count)
        truncated = laps_completed >= BASELINE_MAX_LAPS_PER_EPISODE

        info = {
            "collision": collision,
            "collision_count": self.collision_count,
            "lap_count": self.total_lap_count,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "throttle": throttle,
            "steering": steering,
        }
        return self.current_lidar(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.lidar_data = None
        self.wait_for_lidar()
        self.last_collision_count = self.collision_count
        self.last_lap_count = self.total_lap_count
        self.episode_start_lap_count = self.total_lap_count
        return self.current_lidar(), {}


class AdvancedRaceEnv(RosRaceEnv):
    """Higher-speed training environment with denser LiDAR and previous-action context."""

    def __init__(self, train: bool = True) -> None:
        super().__init__(
            lidar_bins=ADVANCED_LIDAR_BINS,
            train=train,
            node_prefix="f110_advanced_env",
            lidar_reduction="p25",
        )

        self.last_lap_sub = self.node.create_subscription(
            Float32,
            LAST_LAP_TIME_TOPIC,
            self.lap_time_callback,
            10,
        )
        self.lap_count_sub = self.node.create_subscription(
            Int32,
            LAP_COUNT_TOPIC,
            self.lap_count_callback,
            10,
        )

        self.action_space = spaces.MultiDiscrete(np.array(ADVANCED_ACTION_DIMS, dtype=np.int64))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ADVANCED_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.throttle_levels = ADVANCED_THROTTLE_LEVELS
        self.steering_levels = ADVANCED_STEERING_LEVELS
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.total_lap_count = 0
        self.last_lap_count = 0
        self.episode_start_lap_count = 0
        self.previous_action = np.array([0.25, 0.5], dtype=np.float32)

    def lap_time_callback(self, msg: Float32) -> None:
        self.last_lap_time = float(msg.data)

    def lap_count_callback(self, msg: Int32) -> None:
        self.total_lap_count = int(msg.data)

    def _build_observation(self) -> np.ndarray:
        observation = np.concatenate((self.current_lidar(), self.previous_action))
        return observation.astype(np.float32, copy=False)

    def step(self, action: np.ndarray | Sequence[int]):
        throttle_idx, steering_idx = np.asarray(action, dtype=np.int64).reshape(-1).tolist()
        throttle = self.throttle_levels[throttle_idx]
        steering = self.steering_levels[steering_idx]
        self.previous_action = np.array([throttle, (steering + 1.0) * 0.5], dtype=np.float32)

        self.publish_control(throttle, steering)
        self.spin_once()

        collision = self.consume_collision_event()
        if collision:
            reward = -300.0 - 100.0 * throttle
        else:
            reward = 1.0 + 2.5 * throttle
        terminated = collision

        if self.total_lap_count < self.last_lap_count:
            self.last_lap_count = self.total_lap_count
            self.episode_start_lap_count = self.total_lap_count

        lap_completed = self.total_lap_count > self.last_lap_count
        if not collision and lap_completed:
            self.last_lap_count = self.total_lap_count
            if self.last_lap_time is not None and self.last_lap_time > 0.0:
                if self.best_lap_time is None or self.last_lap_time < self.best_lap_time:
                    self.best_lap_time = self.last_lap_time

        laps_completed = max(0, self.total_lap_count - self.episode_start_lap_count)
        truncated = laps_completed >= ADVANCED_MAX_LAPS_PER_EPISODE

        info = {
            "collision": collision,
            "collision_count": self.collision_count,
            "lap_count": self.total_lap_count,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "throttle": throttle,
            "steering": steering,
        }
        return self._build_observation(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.lidar_data = None
        self.wait_for_lidar()
        self.last_collision_count = self.collision_count
        self.last_lap_count = self.total_lap_count
        self.episode_start_lap_count = self.total_lap_count
        self.previous_action = np.array([0.25, 0.5], dtype=np.float32)
        return self._build_observation(), {}


class ExperimentalRaceEnv(RosRaceEnv):
    """Legacy MultiDiscrete experiment kept isolated from the mainline."""

    def __init__(self, train: bool = True) -> None:
        super().__init__(
            lidar_bins=EXPERIMENTAL_LIDAR_BINS,
            train=train,
            node_prefix="f110_experimental_env",
        )

        self.action_space = spaces.MultiDiscrete(np.array(EXPERIMENTAL_ACTION_DIMS, dtype=np.int64))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(EXPERIMENTAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.throttle_levels = EXPERIMENTAL_THROTTLE_LEVELS
        self.steering_levels = EXPERIMENTAL_STEERING_LEVELS

    def _build_observation(self) -> np.ndarray:
        observation = np.concatenate((self.current_lidar(), np.array([0.0], dtype=np.float32)))
        return observation.astype(np.float32, copy=False)

    def step(self, action: np.ndarray | Sequence[int]):
        throttle_idx, steering_idx = np.asarray(action, dtype=np.int64).tolist()
        throttle = self.throttle_levels[throttle_idx]
        steering = self.steering_levels[steering_idx]

        self.publish_control(throttle, steering)
        self.spin_once()

        collision = self.consume_collision_event()
        reward = -100.0 if collision else 1.0 + 0.01 * throttle * self.max_speed
        terminated = collision
        truncated = False

        info = {
            "collision": collision,
            "collision_count": self.collision_count,
            "throttle": throttle,
            "steering": steering,
        }
        return self._build_observation(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.lidar_data = None
        self.wait_for_lidar()
        self.last_collision_count = self.collision_count
        return self._build_observation(), {}


def make_baseline_env(train: bool = True) -> BaselineRaceEnv:
    return BaselineRaceEnv(train=train)


def make_experimental_env(train: bool = True) -> ExperimentalRaceEnv:
    return ExperimentalRaceEnv(train=train)


def make_advanced_env(train: bool = True) -> AdvancedRaceEnv:
    return AdvancedRaceEnv(train=train)
