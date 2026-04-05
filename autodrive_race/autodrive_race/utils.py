"""Shared helpers for paths, checkpoints, and saved-model metadata."""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autodrive_race.constants import (
    ADVANCED_ACTION_DIMS,
    ADVANCED_OBSERVATION_SIZE,
    BASELINE_ACTION_SIZE,
    BASELINE_OBSERVATION_SIZE,
    EXPERIMENTAL_ACTION_DIMS,
    EXPERIMENTAL_OBSERVATION_SIZE,
    PACKAGE_NAME,
    TRITON_MODEL_NAME,
    TRITON_MODEL_REPOSITORY,
    TRITON_MODEL_VERSION,
)

LOGGER = logging.getLogger(__name__)
_CHECKPOINT_PATTERN = re.compile(r"ppo_model_(\d+)_steps\.zip$")

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except ImportError:  # pragma: no cover - depends on ROS 2 environment
    PackageNotFoundError = LookupError
    get_package_share_directory = None


@dataclass(frozen=True)
class SavedModelContract:
    """Contract inferred from a Stable-Baselines3 zip without importing gym."""

    model_path: Path
    observation_shape: tuple[int, ...]
    action_space_type: str
    action_size: int | None = None
    action_dims: tuple[int, ...] | None = None

    @property
    def is_baseline(self) -> bool:
        return (
            "MultiDiscrete" not in self.action_space_type
            and "Discrete" in self.action_space_type
            and self.action_size == BASELINE_ACTION_SIZE
            and self.observation_shape == (BASELINE_OBSERVATION_SIZE,)
        )

    @property
    def is_advanced(self) -> bool:
        return (
            "MultiDiscrete" in self.action_space_type
            and self.action_dims == ADVANCED_ACTION_DIMS
            and self.observation_shape == (ADVANCED_OBSERVATION_SIZE,)
        )

    @property
    def is_experimental(self) -> bool:
        return (
            "MultiDiscrete" in self.action_space_type
            and self.action_dims == EXPERIMENTAL_ACTION_DIMS
            and self.observation_shape == (EXPERIMENTAL_OBSERVATION_SIZE,)
        )


def _source_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_primary_package_root() -> Path:
    """Return the preferred package root for artifacts."""

    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory(PACKAGE_NAME))
        except PackageNotFoundError:
            LOGGER.debug("Package %s not found in ament index; using source tree.", PACKAGE_NAME)

    return _source_package_root()


def iter_package_roots() -> Iterable[Path]:
    primary_root = get_primary_package_root()
    yield primary_root

    source_root = _source_package_root()
    if source_root != primary_root:
        yield source_root


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def find_latest_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    latest_path: Path | None = None
    latest_steps = -1

    for checkpoint_path in checkpoint_dir.glob("ppo_model_*_steps.zip"):
        match = _CHECKPOINT_PATTERN.match(checkpoint_path.name)
        if not match:
            continue

        steps = int(match.group(1))
        if steps > latest_steps:
            latest_path = checkpoint_path
            latest_steps = steps

    return latest_path, latest_steps


def find_latest_compatible_checkpoint(
    checkpoint_dir: Path,
    compatibility_check: Callable[[SavedModelContract], bool],
) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []

    for checkpoint_path in checkpoint_dir.glob("ppo_model_*_steps.zip"):
        match = _CHECKPOINT_PATTERN.match(checkpoint_path.name)
        if match:
            candidates.append((int(match.group(1)), checkpoint_path))

    for steps, checkpoint_path in sorted(candidates, reverse=True):
        try:
            contract = inspect_saved_model_contract(checkpoint_path)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            LOGGER.warning("Skipping unreadable checkpoint %s: %s", checkpoint_path, exc)
            continue

        if compatibility_check(contract):
            return checkpoint_path, steps

    return None, -1


def resolve_model_path(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []

    if explicit_path:
        raw_path = Path(explicit_path).expanduser()
        candidates.append(raw_path if raw_path.is_absolute() else Path.cwd() / raw_path)

    env_override = os.environ.get("AUTODRIVE_RACE_MODEL")
    if env_override:
        raw_path = Path(env_override).expanduser()
        candidates.append(raw_path if raw_path.is_absolute() else Path.cwd() / raw_path)

    for package_root in iter_package_roots():
        candidates.append(package_root / "best_model" / "best_model.zip")
        candidates.append(package_root / "best_model" / "best_model")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else get_primary_package_root() / "best_model" / "best_model.zip"


def get_triton_model_dir(
    model_name: str = TRITON_MODEL_NAME,
    version: str = TRITON_MODEL_VERSION,
) -> Path:
    return get_repository_root() / TRITON_MODEL_REPOSITORY / model_name / version


def inspect_saved_model_contract(model_path: Path) -> SavedModelContract:
    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))

    observation_space = data.get("observation_space", {})
    action_space = data.get("action_space", {})

    return SavedModelContract(
        model_path=model_path,
        observation_shape=tuple(int(dim) for dim in observation_space.get("_shape", [])),
        action_space_type=str(action_space.get(":type:", "")),
        action_size=_parse_single_int(action_space.get("n")),
        action_dims=_parse_int_sequence(action_space.get("nvec")),
    )


def _parse_single_int(raw_value: object) -> int | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, int):
        return raw_value

    matches = re.findall(r"-?\d+", str(raw_value))
    return int(matches[0]) if matches else None


def _parse_int_sequence(raw_value: object) -> tuple[int, ...] | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, (list, tuple)):
        return tuple(int(value) for value in raw_value)

    matches = re.findall(r"-?\d+", str(raw_value))
    return tuple(int(match) for match in matches) if matches else None
