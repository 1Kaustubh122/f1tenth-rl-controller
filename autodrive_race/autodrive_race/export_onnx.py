"""Export a saved PPO policy to ONNX."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from autodrive_race.constants import DEFAULT_ONNX_OPSET
from autodrive_race.inference_utils import (
    assert_supported_contract,
    build_dummy_observation,
    default_onnx_output_path,
    default_triton_model_path,
    get_action_output_width,
    get_logits_output_width,
    get_onnx_io_names,
)
from autodrive_race.utils import ensure_directories, get_repository_root, inspect_saved_model_contract, resolve_model_path

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PPO policy to ONNX.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--copy-to-triton-repo", action="store_true")
    return parser.parse_args(argv)


class DeterministicPolicyExporter:  # pragma: no cover - runtime exercised where torch is installed
    def __init__(self, model, contract) -> None:
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, sb3_model, saved_contract) -> None:
                super().__init__()
                self.policy = sb3_model.policy
                self.contract = saved_contract

            def forward(self, observation):
                distribution = self.policy.get_distribution(observation)
                action = distribution.get_actions(deterministic=True).to(dtype=torch.int32)
                if action.ndim == 1:
                    action = action.unsqueeze(-1)

                raw_distribution = distribution.distribution
                if isinstance(raw_distribution, list):
                    logits = torch.cat([categorical.logits for categorical in raw_distribution], dim=1)
                else:
                    logits = raw_distribution.logits

                return action, logits.to(dtype=torch.float32)

        self.module = _Wrapper(model, contract)


def export_model(model_path: Path, output_path: Path, opset: int, dynamic_batch: bool) -> Path:
    import torch
    from stable_baselines3 import PPO

    contract = inspect_saved_model_contract(model_path)
    assert_supported_contract(contract)

    model = PPO.load(str(model_path), device="cpu")
    model.policy.eval()

    exporter = DeterministicPolicyExporter(model, contract).module
    dummy_observation = build_dummy_observation(contract, batch_size=2 if dynamic_batch else 1)
    dummy_tensor = torch.from_numpy(dummy_observation)
    input_name, action_output_name, logits_output_name = get_onnx_io_names()

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            action_output_name: {0: "batch"},
            logits_output_name: {0: "batch"},
        }

    ensure_directories(output_path.parent)
    torch.onnx.export(
        exporter,
        dummy_tensor,
        str(output_path),
        input_names=[input_name],
        output_names=[action_output_name, logits_output_name],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    LOGGER.info("Exported ONNX policy to %s", output_path)
    _sanity_check_export(model, contract, output_path)
    return output_path


def _sanity_check_export(model, contract, output_path: Path) -> None:  # pragma: no cover - runtime exercised where deps exist
    observation = build_dummy_observation(contract, batch_size=1, seed=123)
    sb3_action, _ = model.predict(observation, deterministic=True)
    sb3_action = np.asarray(sb3_action, dtype=np.int32).reshape(1, -1)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export succeeded but onnxruntime is not installed, so the export sanity check "
            "could not run."
        ) from exc

    input_name, action_output_name, _ = get_onnx_io_names()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        [action_output_name],
        {input_name: observation.astype(np.float32)},
    )
    onnx_action = np.asarray(outputs[0], dtype=np.int32).reshape(1, -1)

    if onnx_action.shape != sb3_action.shape:
        raise RuntimeError(
            f"Export sanity check failed: SB3 action shape {sb3_action.shape} != ONNX shape {onnx_action.shape}"
        )
    if not np.array_equal(onnx_action, sb3_action):
        raise RuntimeError(
            f"Export sanity check failed: SB3 deterministic action {sb3_action.tolist()} "
            f"!= ONNX action {onnx_action.tolist()}"
        )

    LOGGER.info("ONNX sanity check passed for %s", output_path)


def maybe_copy_to_triton_repo(onnx_path: Path, contract) -> None:
    triton_model_path = default_triton_model_path()
    ensure_directories(triton_model_path.parent)
    shutil.copy2(onnx_path, triton_model_path)
    write_triton_config(contract)
    LOGGER.info("Copied ONNX model to Triton repository path %s", triton_model_path)


def write_triton_config(contract) -> None:
    config_path = get_repository_root() / "triton_model_repo" / "f110_policy" / "config.pbtxt"
    action_width = get_action_output_width(contract)
    logits_width = get_logits_output_width(contract)
    config_text = f"""name: "f110_policy"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "observation"
    data_type: TYPE_FP32
    dims: [ 1, {contract.observation_shape[0]} ]
  }}
]

output [
  {{
    name: "action"
    data_type: TYPE_INT32
    dims: [ 1, {action_width} ]
  }},
  {{
    name: "action_logits"
    data_type: TYPE_FP32
    dims: [ 1, {logits_width} ]
  }}
]
"""
    ensure_directories(config_path.parent)
    config_path.write_text(config_text, encoding="utf-8")
    LOGGER.info("Wrote Triton config to %s", config_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        import onnx  # noqa: F401
        import torch  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "export_onnx.py requires torch, onnx, and stable-baselines3 in the active environment."
        ) from exc

    model_path = resolve_model_path(args.model_path)
    contract = inspect_saved_model_contract(model_path)
    assert_supported_contract(contract)

    output_path = Path(args.output).expanduser() if args.output else default_onnx_output_path(model_path)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    export_model(model_path, output_path, args.opset, args.dynamic_batch)
    if args.copy_to_triton_repo:
        maybe_copy_to_triton_repo(output_path, contract)


if __name__ == "__main__":
    main()
