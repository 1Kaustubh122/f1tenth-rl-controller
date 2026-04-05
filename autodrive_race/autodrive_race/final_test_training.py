"""Advanced high-speed PPO training entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from autodrive_race.constants import DEFAULT_CHECKPOINT_FREQ, DEFAULT_TOTAL_TIMESTEPS
from autodrive_race.utils import ensure_directories, find_latest_compatible_checkpoint, get_primary_package_root

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train the higher-speed advanced PPO controller.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_known_args(argv)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(env, checkpoint_path: Path | None, cli_args: argparse.Namespace):
    from stable_baselines3 import PPO

    device = resolve_device(cli_args.device)
    if checkpoint_path is not None and checkpoint_path.exists():
        LOGGER.info("Resuming advanced run from checkpoint: %s on %s", checkpoint_path, device)
        return PPO.load(str(checkpoint_path), env=env, device=device)

    LOGGER.info("No advanced checkpoint found; starting a new advanced run on %s.", device)
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cli_args.learning_rate,
        batch_size=cli_args.batch_size,
        n_steps=cli_args.n_steps,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.15,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]}},
        device=device,
    )


def main(args=None) -> None:
    cli_args, ros_args = parse_args(args)
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Running advanced high-speed PPO training.")

    import rclpy
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    from autodrive_race.env import make_advanced_env

    rclpy.init(args=ros_args)
    train_env = None
    eval_env = None
    model = None
    try:
        package_root = get_primary_package_root()
        checkpoint_dir = package_root / "advanced_checkpoints"
        best_model_dir = package_root / "advanced_best_model"
        logs_dir = package_root / "advanced_logs"
        ensure_directories(checkpoint_dir, best_model_dir, logs_dir)

        latest_checkpoint, latest_steps = find_latest_compatible_checkpoint(
            checkpoint_dir,
            lambda contract: contract.is_advanced,
        )
        if latest_checkpoint is not None:
            LOGGER.info("Latest advanced checkpoint: %s (%d steps)", latest_checkpoint.name, latest_steps)

        train_env = make_vec_env(lambda: make_advanced_env(train=True), n_envs=1)
        eval_env = make_vec_env(lambda: make_advanced_env(train=False), n_envs=1)
        model = build_model(train_env, latest_checkpoint, cli_args)

        checkpoint_callback = CheckpointCallback(
            save_freq=cli_args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_model",
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(logs_dir),
            eval_freq=cli_args.checkpoint_freq,
            deterministic=True,
        )

        try:
            model.learn(
                total_timesteps=cli_args.timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True,
            )
        except KeyboardInterrupt:
            interrupt_checkpoint = checkpoint_dir / f"ppo_model_{int(model.num_timesteps)}_steps.zip"
            model.save(str(interrupt_checkpoint))
            LOGGER.warning("Training interrupted; saved resumable checkpoint to %s", interrupt_checkpoint)
            raise

        final_model_path = checkpoint_dir / "ppo_race_model_final.zip"
        model.save(str(final_model_path))
        LOGGER.info("Advanced final model saved to %s", final_model_path)
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
