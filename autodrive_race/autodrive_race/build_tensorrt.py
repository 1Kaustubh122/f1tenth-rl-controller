"""Build a TensorRT engine from an exported ONNX model."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from autodrive_race.constants import BASELINE_OBSERVATION_SIZE, ONNX_INPUT_NAME
from autodrive_race.inference_utils import assert_supported_contract
from autodrive_race.utils import ensure_directories, inspect_saved_model_contract, resolve_model_path

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX policy export.")
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--engine-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--input-name", type=str, default=ONNX_INPUT_NAME)
    parser.add_argument("--shape", type=str, default=None)
    parser.add_argument("--min-shape", type=str, default=None)
    parser.add_argument("--opt-shape", type=str, default=None)
    parser.add_argument("--max-shape", type=str, default=None)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    return parser.parse_args(argv)


def infer_shape(args: argparse.Namespace) -> str:
    if args.shape:
        return args.shape
    if args.model_path:
        contract = inspect_saved_model_contract(resolve_model_path(args.model_path))
        assert_supported_contract(contract)
        return f"1x{contract.observation_shape[0]}"
    return f"1x{BASELINE_OBSERVATION_SIZE}"


def build_with_trtexec(args: argparse.Namespace) -> None:
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        raise RuntimeError("TensorRT build requires `trtexec` to be installed and available on PATH.")

    onnx_path = Path(args.onnx_path).expanduser()
    engine_path = Path(args.engine_path).expanduser()
    if not onnx_path.is_absolute():
        onnx_path = Path.cwd() / onnx_path
    if not engine_path.is_absolute():
        engine_path = Path.cwd() / engine_path

    ensure_directories(engine_path.parent)

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={args.workspace_mib}",
    ]
    if args.fp16:
        command.append("--fp16")
    if args.int8:
        command.append("--int8")

    if args.min_shape and args.opt_shape and args.max_shape:
        command.extend(
            [
                f"--minShapes={args.input_name}:{args.min_shape}",
                f"--optShapes={args.input_name}:{args.opt_shape}",
                f"--maxShapes={args.input_name}:{args.max_shape}",
            ]
        )
    else:
        command.append(f"--shapes={args.input_name}:{infer_shape(args)}")

    LOGGER.info("Running TensorRT build command: %s", " ".join(command))
    subprocess.run(command, check=True)
    LOGGER.info("TensorRT engine written to %s", engine_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    build_with_trtexec(args)


if __name__ == "__main__":
    main()
