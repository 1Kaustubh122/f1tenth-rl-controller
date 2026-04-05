# f1tenth-rl-controller

ROS 2 Python package for training, evaluating, exporting, and serving PPO-based autonomous racing policies for the AutoDRIVE RoboRacer / F1TENTH-style simulator workflow.

This repository contains:

- a baseline PPO racing controller compatible with the saved model artifacts already included in the repo
- a higher-speed training path for new experiments
- ONNX export utilities
- inference benchmarking utilities for SB3, ONNX Runtime, and TensorRT
- a minimal Triton model repository and client

## Project Scope

The repository is centered on reinforcement learning policy inference for autonomous racing. The current policy family uses Stable-Baselines3 PPO and ROS 2 topic I/O to interact with the simulator.

At a high level, the code supports:

1. training a PPO controller
2. evaluating a saved PPO model in the simulator
3. exporting a trained policy to ONNX
4. benchmarking inference backends
5. serving exported models through Triton Inference Server

## Repository Layout

```text
.
├── autodrive_race/
│   ├── autodrive_race/
│   │   ├── benchmark_inference.py
│   │   ├── build_tensorrt.py
│   │   ├── constants.py
│   │   ├── env.py
│   │   ├── eval.py
│   │   ├── export_onnx.py
│   │   ├── final_test_training.py
│   │   ├── inference_utils.py
│   │   ├── ppo_training.py
│   │   ├── triton_client.py
│   │   └── utils.py
│   ├── best_model/
│   ├── checkpoints/
│   ├── package.xml
│   └── setup.py
└── triton_model_repo/
    └── f110_policy/
```

## Policy Paths

The repo currently contains two main policy paths.

### 1. Baseline Path

The baseline path is the one used by the saved PPO model already stored in the repository.

- entry point: `autodrive_race.ppo_training`
- evaluation entry point: `autodrive_race.eval`
- environment factory: `make_baseline_env`
- saved-model contract:
  - observation shape: `180`
  - action space: `Discrete(25)`

This path is the compatibility-preserving path. If you want to resume from the included checkpoints or evaluate the included best model, use this one.

### 2. Advanced Path

The advanced path is a train-from-scratch higher-speed variant.

- entry point: `autodrive_race.final_test_training`
- environment factory: `make_advanced_env`
- contract:
  - observation shape: `242`
  - action space: `MultiDiscrete([4, 11])`
  - observation contents:
    - 240 LiDAR features
    - 2 previous-action features

This path is intended for new experiments and does not load the legacy baseline weights.

## Requirements

Minimum expected software stack:

- Ubuntu with ROS 2 Humble
- Python 3.10+
- `colcon`
- simulator / devkit environment that publishes and consumes the expected AutoDRIVE topics

For training and evaluation:

- `stable-baselines3`
- `gymnasium`
- `numpy`
- `rclpy`
- ROS 2 message packages used by the simulator

For ONNX export:

- `torch`
- `onnx`
- `onnxruntime`

For TensorRT benchmarking / engine build:

- `tensorrt`
- `pycuda`
- `trtexec`

For Triton client usage:

- `tritonclient[grpc]` or `tritonclient[http]`

## ROS 2 Workspace Setup

Clone the repository into a ROS 2 workspace and build the package.

```bash
mkdir -p ~/roboracer_ws/src
cd ~/roboracer_ws/src
git clone https://github.com/1Kaustubh122/f1tenth-rl-controller.git

cd ~/roboracer_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select autodrive_race
source install/setup.bash
```

## Expected ROS Topics

The environments in this repository expect the simulator bridge to provide the following topics:

- `/autodrive/roboracer_1/lidar`
- `/autodrive/roboracer_1/collision_count`
- `/autodrive/roboracer_1/last_lap_time`
- `/autodrive/roboracer_1/lap_count`

The controller publishes:

- `/autodrive/roboracer_1/throttle_command`
- `/autodrive/roboracer_1/steering_command`

## Training

Before launching training, make sure:

1. the simulator is running
2. the AutoDRIVE bridge is running
3. the ROS 2 graph is healthy and the LiDAR topic is active

### Baseline Training

Resume from the latest compatible checkpoint, or start a fresh baseline run if none exists:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race ppo_training -- --timesteps 1000000 --checkpoint-freq 10000
```

Outputs:

- checkpoints: `autodrive_race/checkpoints/`
- best model: `autodrive_race/best_model/`
- eval logs: `autodrive_race/logs/`

### Advanced Training

Train the higher-speed model:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race final_test_training -- \
  --device cuda \
  --timesteps 3000000 \
  --learning-rate 1e-4 \
  --n-steps 8192 \
  --batch-size 1024 \
  --checkpoint-freq 10000
```

If GPU memory is tighter than expected, use:

```bash
ros2 run autodrive_race final_test_training -- \
  --device cuda \
  --timesteps 3000000 \
  --learning-rate 1e-4 \
  --n-steps 4096 \
  --batch-size 512 \
  --checkpoint-freq 10000
```

Outputs:

- checkpoints: `autodrive_race/advanced_checkpoints/`
- best model: `autodrive_race/advanced_best_model/best_model.zip`
- eval logs: `autodrive_race/advanced_logs/`

## Evaluation

Evaluate the default saved model:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race eval -- --steps 5000
```

Evaluate a specific model:

```bash
ros2 run autodrive_race eval -- --model-path /absolute/path/to/model.zip --steps 5000
```

The evaluator inspects the saved-model contract and selects the appropriate environment automatically.

## Export to ONNX

Export a PPO model to ONNX:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race export_onnx -- \
  --model-path /absolute/path/to/model.zip \
  --output /absolute/path/to/model.onnx \
  --opset 17
```

Copy the exported model into the Triton repository structure at the same time:

```bash
ros2 run autodrive_race export_onnx -- \
  --model-path /absolute/path/to/model.zip \
  --output /absolute/path/to/model.onnx \
  --copy-to-triton-repo
```

The export path performs a deterministic-action sanity check with ONNX Runtime when the required dependencies are installed.

## Benchmark Inference

Compare SB3, ONNX Runtime, and TensorRT:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race benchmark_inference -- \
  --model-path /absolute/path/to/model.zip \
  --onnx-path /absolute/path/to/model.onnx \
  --engine-path /absolute/path/to/model.plan \
  --warmup 20 \
  --runs 200 \
  --device cuda
```

Write results to a JSON file:

```bash
ros2 run autodrive_race benchmark_inference -- \
  --model-path /absolute/path/to/model.zip \
  --onnx-path /absolute/path/to/model.onnx \
  --output /absolute/path/to/results.json
```

If ONNX Runtime or TensorRT is unavailable, the script reports that backend as unavailable instead of crashing the whole benchmark run.

## Build a TensorRT Engine

If `trtexec` is installed:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race build_tensorrt -- \
  --onnx-path /absolute/path/to/model.onnx \
  --engine-path /absolute/path/to/model.plan \
  --model-path /absolute/path/to/model.zip \
  --fp16
```

## Triton Inference Server

This repository includes a minimal Triton model repository:

```text
triton_model_repo/f110_policy/
├── config.pbtxt
└── 1/
```

After exporting a model into the Triton repository location, launch Triton with that repository:

```bash
tritonserver --model-repository /absolute/path/to/f1tenth-rl-controller/triton_model_repo
```

## Triton Client

Send a single observation to Triton and decode the returned action into throttle and steering:

```bash
source /opt/ros/humble/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 run autodrive_race triton_client -- \
  --model-path /absolute/path/to/model.zip \
  --url localhost:8001 \
  --protocol grpc
```

Optionally provide an explicit observation:

```bash
ros2 run autodrive_race triton_client -- \
  --model-path /absolute/path/to/model.zip \
  --observation "0.2,0.3,0.4,..."
```

## Simulator Notes

- For LiDAR-only training, no-graphics / headless simulator mode is usually preferable because it avoids rendering overhead.
- GUI mode is useful for visual inspection and playback of a trained model.
- If your simulator stack distinguishes between true headless camera rendering and no-graphics mode, check the simulator’s own documentation for camera-specific limitations. This repository’s policies are LiDAR-based.

## Public Usage Notes

- This repository does not assume any private shell aliases.
- All runnable entry points are exposed through ROS 2 console scripts.
- Use absolute paths when passing model or artifact locations unless you are sure about your working directory.

## License

See [LICENSE](/media/kaustubh/SSD1/GitHub/f1tenth-rl-controller/LICENSE).
