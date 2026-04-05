"""Evaluation entry point for saved PPO models."""

from __future__ import annotations

import argparse
import logging

from autodrive_race.constants import DEFAULT_EVAL_STEPS
from autodrive_race.utils import inspect_saved_model_contract, resolve_model_path

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO racing model.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_EVAL_STEPS)
    return parser.parse_known_args(argv)


def select_eval_env(model_contract):
    from autodrive_race.env import make_advanced_env, make_baseline_env, make_experimental_env

    if model_contract.is_baseline:
        LOGGER.info(
            "Using baseline evaluation environment for %s (obs=%s, action=%s).",
            model_contract.model_path,
            model_contract.observation_shape,
            model_contract.action_size,
        )
        return make_baseline_env(train=False)

    if model_contract.is_advanced:
        LOGGER.info(
            "Using advanced evaluation environment for %s (obs=%s, action=%s).",
            model_contract.model_path,
            model_contract.observation_shape,
            model_contract.action_size,
        )
        return make_advanced_env(train=False)

    if model_contract.is_experimental:
        LOGGER.warning(
            "Model %s uses the isolated experimental contract; evaluating with the experimental env.",
            model_contract.model_path,
        )
        return make_experimental_env(train=False)

    raise ValueError(
        "Unsupported saved model contract for "
        f"{model_contract.model_path}: action={model_contract.action_space_type}, "
        f"obs={model_contract.observation_shape}"
    )


def main(args=None) -> None:
    cli_args, ros_args = parse_args(args)
    logging.basicConfig(level=logging.INFO)

    import rclpy
    from stable_baselines3 import PPO

    rclpy.init(args=ros_args)
    env = None
    try:
        model_path = resolve_model_path(cli_args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find a saved PPO model at {model_path}")

        model_contract = inspect_saved_model_contract(model_path)
        env = select_eval_env(model_contract)
        model = PPO.load(str(model_path), env=env, device="cpu")

        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(cli_args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                LOGGER.info("Episode reward: %.3f", episode_reward)
                obs, _ = env.reset()
                episode_reward = 0.0
    finally:
        if env is not None:
            env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
