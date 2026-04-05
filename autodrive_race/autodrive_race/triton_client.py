"""Minimal Triton client for the exported racing policy."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from autodrive_race.constants import TRITON_MODEL_NAME, TRITON_MODEL_VERSION
from autodrive_race.inference_utils import (
    assert_supported_contract,
    build_dummy_observation,
    decode_action_to_control,
    decode_logits_to_action,
    ensure_action_matrix,
    get_onnx_io_names,
)
from autodrive_race.utils import inspect_saved_model_contract, resolve_model_path

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a single observation to Triton and decode the action.")
    parser.add_argument("--url", type=str, default="localhost:8001")
    parser.add_argument("--protocol", choices=("grpc", "http"), default="grpc")
    parser.add_argument("--model-name", type=str, default=TRITON_MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=TRITON_MODEL_VERSION)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--sample-npy", type=str, default=None)
    parser.add_argument("--observation", type=str, default=None)
    return parser.parse_args(argv)


def load_observation(args: argparse.Namespace, contract) -> np.ndarray:
    if args.sample_npy:
        observation = np.load(Path(args.sample_npy).expanduser()).astype(np.float32)
    elif args.observation:
        values = [float(value.strip()) for value in args.observation.split(",") if value.strip()]
        observation = np.asarray(values, dtype=np.float32)
    else:
        observation = build_dummy_observation(contract, batch_size=1, seed=7)[0]

    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    expected_width = contract.observation_shape[0]
    if observation.shape[1] != expected_width:
        raise ValueError(f"Observation width {observation.shape[1]} does not match contract width {expected_width}.")
    return observation


def create_triton_client(protocol: str, url: str):
    if protocol == "grpc":
        try:
            import tritonclient.grpc as triton_client
        except ImportError as exc:
            raise RuntimeError("tritonclient[grpc] is required for the gRPC Triton client path.") from exc
        return triton_client, triton_client.InferenceServerClient(url=url)

    try:
        import tritonclient.http as triton_client
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required for the HTTP Triton client path.") from exc
    return triton_client, triton_client.InferenceServerClient(url=url)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    contract = inspect_saved_model_contract(resolve_model_path(args.model_path))
    assert_supported_contract(contract)
    observation = load_observation(args, contract)

    triton_module, client = create_triton_client(args.protocol, args.url)
    input_name, action_output_name, logits_output_name = get_onnx_io_names()

    infer_input = triton_module.InferInput(input_name, observation.shape, "FP32")
    infer_input.set_data_from_numpy(observation)
    requested_outputs = [
        triton_module.InferRequestedOutput(action_output_name),
        triton_module.InferRequestedOutput(logits_output_name),
    ]

    result = client.infer(
        model_name=args.model_name,
        model_version=args.model_version,
        inputs=[infer_input],
        outputs=requested_outputs,
    )

    action_output = result.as_numpy(action_output_name)
    logits_output = result.as_numpy(logits_output_name)
    if action_output is None and logits_output is None:
        raise RuntimeError("Triton response did not include either action or action_logits outputs.")

    action_matrix = (
        ensure_action_matrix(action_output)
        if action_output is not None
        else decode_logits_to_action(contract, logits_output)
    )
    throttle, steering = decode_action_to_control(contract, action_matrix[0])

    payload = {
        "action": action_matrix.tolist(),
        "decoded_control": {"throttle": throttle, "steering": steering},
    }
    if logits_output is not None:
        payload["logits_shape"] = list(np.asarray(logits_output).shape)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
