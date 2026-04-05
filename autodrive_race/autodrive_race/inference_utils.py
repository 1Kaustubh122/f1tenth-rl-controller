"""Helpers shared by ONNX export, benchmarking, and Triton client code."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from autodrive_race.constants import (
    ADVANCED_ACTION_DIMS,
    ADVANCED_OBSERVATION_SIZE,
    ADVANCED_STEERING_LEVELS,
    ADVANCED_THROTTLE_LEVELS,
    BASELINE_ACTIONS,
    BASELINE_ACTION_SIZE,
    BASELINE_OBSERVATION_SIZE,
    EXPERIMENTAL_ACTION_DIMS,
    EXPERIMENTAL_STEERING_LEVELS,
    EXPERIMENTAL_THROTTLE_LEVELS,
    EXPERIMENTAL_OBSERVATION_SIZE,
    ONNX_ACTION_OUTPUT_NAME,
    ONNX_INPUT_NAME,
    ONNX_LOGITS_OUTPUT_NAME,
    TRITON_MODEL_NAME,
    TRITON_MODEL_REPOSITORY,
    TRITON_MODEL_VERSION,
)
from autodrive_race.utils import SavedModelContract, get_repository_root


def assert_supported_contract(contract: SavedModelContract) -> None:
    if contract.is_baseline or contract.is_advanced or contract.is_experimental:
        return

    raise ValueError(
        "Unsupported saved-model contract: "
        f"obs={contract.observation_shape}, action_type={contract.action_space_type}, "
        f"action_size={contract.action_size}, action_dims={contract.action_dims}"
    )


def build_dummy_observation(
    contract: SavedModelContract,
    batch_size: int = 1,
    seed: int = 0,
) -> np.ndarray:
    assert_supported_contract(contract)
    observation_size = contract.observation_shape[0]
    rng = np.random.default_rng(seed)
    return rng.random((batch_size, observation_size), dtype=np.float32)


def get_action_output_width(contract: SavedModelContract) -> int:
    assert_supported_contract(contract)
    return 1 if contract.is_baseline else len(contract.action_dims or ())


def get_logits_output_width(contract: SavedModelContract) -> int:
    assert_supported_contract(contract)
    if contract.is_baseline:
        return int(contract.action_size or BASELINE_ACTION_SIZE)
    if contract.is_advanced:
        return sum(contract.action_dims or ADVANCED_ACTION_DIMS)
    return sum(contract.action_dims or EXPERIMENTAL_ACTION_DIMS)


def decode_action_to_control(
    contract: SavedModelContract,
    action: int | np.ndarray | Sequence[int],
) -> tuple[float, float]:
    assert_supported_contract(contract)

    if contract.is_baseline:
        action_index = int(np.asarray(action).reshape(-1)[0])
        return BASELINE_ACTIONS[action_index]

    if contract.is_advanced:
        action_values = np.asarray(action, dtype=np.int64).reshape(-1)
        throttle_idx, steering_idx = action_values.tolist()
        return (
            ADVANCED_THROTTLE_LEVELS[throttle_idx],
            ADVANCED_STEERING_LEVELS[steering_idx],
        )

    action_values = np.asarray(action, dtype=np.int64).reshape(-1)
    throttle_idx, steering_idx = action_values.tolist()
    return (
        EXPERIMENTAL_THROTTLE_LEVELS[throttle_idx],
        EXPERIMENTAL_STEERING_LEVELS[steering_idx],
    )


def decode_logits_to_action(contract: SavedModelContract, logits: np.ndarray) -> np.ndarray:
    assert_supported_contract(contract)

    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[None, :]

    if contract.is_baseline:
        return np.argmax(logits, axis=1, keepdims=True).astype(np.int64)
    if contract.is_advanced:
        head_sizes = contract.action_dims or ADVANCED_ACTION_DIMS
        head_logits = np.split(logits, np.cumsum(head_sizes[:-1]), axis=1)
        decoded = [np.argmax(head_logit, axis=1) for head_logit in head_logits]
        return np.stack(decoded, axis=1).astype(np.int64)

    head_sizes = contract.action_dims or EXPERIMENTAL_ACTION_DIMS
    head_logits = np.split(logits, np.cumsum(head_sizes[:-1]), axis=1)
    decoded = [np.argmax(head_logit, axis=1) for head_logit in head_logits]
    return np.stack(decoded, axis=1).astype(np.int64)


def ensure_action_matrix(action: np.ndarray | Sequence[int] | int) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.int64)
    if action_array.ndim == 0:
        return action_array.reshape(1, 1)
    if action_array.ndim == 1:
        return action_array.reshape(1, -1)
    return action_array


def default_onnx_output_path(model_path: Path) -> Path:
    return model_path.with_suffix(".onnx")


def default_triton_model_path() -> Path:
    return get_repository_root() / TRITON_MODEL_REPOSITORY / TRITON_MODEL_NAME / TRITON_MODEL_VERSION / "model.onnx"


def get_onnx_io_names() -> tuple[str, str, str]:
    return ONNX_INPUT_NAME, ONNX_ACTION_OUTPUT_NAME, ONNX_LOGITS_OUTPUT_NAME


def observation_size_for_contract(contract: SavedModelContract) -> int:
    assert_supported_contract(contract)
    if contract.is_baseline:
        return BASELINE_OBSERVATION_SIZE
    if contract.is_advanced:
        return ADVANCED_OBSERVATION_SIZE
    return EXPERIMENTAL_OBSERVATION_SIZE
