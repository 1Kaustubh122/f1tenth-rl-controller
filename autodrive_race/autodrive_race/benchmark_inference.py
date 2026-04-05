"""Benchmark SB3, ONNX Runtime, and TensorRT inference latency."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np

from autodrive_race.constants import DEFAULT_BENCHMARK_RUNS, DEFAULT_BENCHMARK_WARMUP
from autodrive_race.inference_utils import (
    assert_supported_contract,
    build_dummy_observation,
    default_onnx_output_path,
    get_onnx_io_names,
)
from autodrive_race.utils import inspect_saved_model_contract, resolve_model_path

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PPO, ONNX Runtime, and TensorRT inference.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--onnx-path", type=str, default=None)
    parser.add_argument("--engine-path", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_BENCHMARK_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "median_ms": float(statistics.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "max_ms": float(max(latencies_ms)),
    }


def benchmark_callable(func, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        func()

    latencies_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        func()
        latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    return summarize_latencies(latencies_ms)


def benchmark_sb3(model_path: Path, observation: np.ndarray, device: str, warmup: int, runs: int) -> dict[str, float]:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for SB3 benchmarking.") from exc

    model = PPO.load(str(model_path), device=device)

    def _predict() -> None:
        model.predict(observation, deterministic=True)

    return benchmark_callable(_predict, warmup, runs)


def benchmark_onnxruntime(
    onnx_path: Path,
    observation: np.ndarray,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX benchmarking.") from exc

    input_name, action_output_name, _ = get_onnx_io_names()
    requested_provider = "CUDAExecutionProvider" if device.lower() == "cuda" else "CPUExecutionProvider"
    providers = [requested_provider, "CPUExecutionProvider"] if requested_provider != "CPUExecutionProvider" else [requested_provider]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    def _predict() -> None:
        session.run([action_output_name], {input_name: observation})

    return benchmark_callable(_predict, warmup, runs)


def benchmark_tensorrt(
    engine_path: Path,
    observation: np.ndarray,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT benchmarking requires tensorrt and pycuda in the active environment."
        ) from exc

    logger = trt.Logger(trt.Logger.ERROR)
    with engine_path.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Failed to create TensorRT execution context for {engine_path}")

    stream = cuda.Stream()
    host_input = np.ascontiguousarray(observation.astype(np.float32))

    if hasattr(engine, "num_io_tensors"):
        input_name = next(
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        )
        context.set_input_shape(input_name, tuple(observation.shape))

        device_input = cuda.mem_alloc(host_input.nbytes)
        context.set_tensor_address(input_name, int(device_input))

        device_outputs = []
        for index in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                continue

            output_shape = tuple(context.get_tensor_shape(tensor_name))
            output_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            host_output = np.empty(output_shape, dtype=output_dtype)
            device_output = cuda.mem_alloc(host_output.nbytes)
            context.set_tensor_address(tensor_name, int(device_output))
            device_outputs.append((device_output, host_output))

        def _predict() -> None:
            cuda.memcpy_htod_async(device_input, host_input, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            for device_output, host_output in device_outputs:
                cuda.memcpy_dtoh_async(host_output, device_output, stream)
            stream.synchronize()

    else:
        input_index = next(index for index in range(engine.num_bindings) if engine.binding_is_input(index))
        context.set_binding_shape(input_index, tuple(observation.shape))

        bindings: list[int] = [0] * engine.num_bindings
        device_input = cuda.mem_alloc(host_input.nbytes)
        bindings[input_index] = int(device_input)

        device_outputs = []
        for binding_index in range(engine.num_bindings):
            if engine.binding_is_input(binding_index):
                continue

            output_shape = tuple(context.get_binding_shape(binding_index))
            output_dtype = trt.nptype(engine.get_binding_dtype(binding_index))
            host_output = np.empty(output_shape, dtype=output_dtype)
            device_output = cuda.mem_alloc(host_output.nbytes)
            bindings[binding_index] = int(device_output)
            device_outputs.append((device_output, host_output))

        def _predict() -> None:
            cuda.memcpy_htod_async(device_input, host_input, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            for device_output, host_output in device_outputs:
                cuda.memcpy_dtoh_async(host_output, device_output, stream)
            stream.synchronize()

    return benchmark_callable(_predict, warmup, runs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    model_path = resolve_model_path(args.model_path)
    contract = inspect_saved_model_contract(model_path)
    assert_supported_contract(contract)

    observation = build_dummy_observation(contract, batch_size=1)
    results: dict[str, dict[str, float] | str] = {}

    try:
        results["sb3"] = benchmark_sb3(model_path, observation, args.device, args.warmup, args.runs)
    except Exception as exc:
        results["sb3"] = {"error": str(exc)}

    onnx_path = None
    if args.onnx_path:
        onnx_path = Path(args.onnx_path).expanduser()
        if not onnx_path.is_absolute():
            onnx_path = Path.cwd() / onnx_path
    else:
        onnx_path = default_onnx_output_path(model_path)

    if onnx_path.exists():
        try:
            results["onnxruntime"] = benchmark_onnxruntime(onnx_path, observation, args.device, args.warmup, args.runs)
        except Exception as exc:
            results["onnxruntime"] = {"error": str(exc)}
    else:
        results["onnxruntime"] = {"error": f"ONNX model not found at {onnx_path}"}

    if args.engine_path:
        engine_path = Path(args.engine_path).expanduser()
        if not engine_path.is_absolute():
            engine_path = Path.cwd() / engine_path
        if engine_path.exists():
            try:
                results["tensorrt"] = benchmark_tensorrt(engine_path, observation, args.warmup, args.runs)
            except Exception as exc:
                results["tensorrt"] = {"error": str(exc)}
        else:
            results["tensorrt"] = {"error": f"TensorRT engine not found at {engine_path}"}
    else:
        results["tensorrt"] = {"error": "No TensorRT engine path provided."}

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.write_text(rendered + "\n", encoding="utf-8")
        LOGGER.info("Benchmark results written to %s", output_path)


if __name__ == "__main__":
    main()
