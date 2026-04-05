# Overnight Training Runbook

## Save Locations

- Best model by evaluation reward:
  - `autodrive_race/advanced_best_model/best_model.zip`
- Advanced checkpoints:
  - `autodrive_race/advanced_checkpoints/ppo_model_*_steps.zip`
- Final model after training ends:
  - `autodrive_race/advanced_checkpoints/ppo_race_model_final.zip`
- Evaluation logs:
  - `autodrive_race/advanced_logs/`

The best model is saved automatically by the training callback.

## Full Command Flow

### 1. Pull Images

```bash
rr_pull
```

### 2. Start Containers

```bash
rr_api_start
```

In another terminal:

```bash
rr_sim_start
```

### 3. Prepare Workspace

```bash
rr_ws_init
rr_rebuild
```

### 4. Start Simulator in No-Graphics Mode

```bash
rr_sim_headless
```

### 5. Start Headless Bridge

```bash
rr_bridge_headless
```

### 6. Start Overnight Training

Recommended:

```bash
sudo docker exec -it autodrive_roboracer_api bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd '${RR_WS_IN}' &&
  source install/setup.bash &&
  ros2 run autodrive_race final_test_training -- \
    --device cuda \
    --timesteps 3000000 \
    --learning-rate 1e-4 \
    --n-steps 8192 \
    --batch-size 1024 \
    --checkpoint-freq 10000
"
```

Lower-memory fallback:

```bash
sudo docker exec -it autodrive_roboracer_api bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd '${RR_WS_IN}' &&
  source install/setup.bash &&
  ros2 run autodrive_race final_test_training -- \
    --device cuda \
    --timesteps 3000000 \
    --learning-rate 1e-4 \
    --n-steps 4096 \
    --batch-size 512 \
    --checkpoint-freq 10000
"
```

### 7. Resume Later

Run the same command again. It resumes from the latest compatible file in `autodrive_race/advanced_checkpoints/`.

## View Best Model in GUI Later

### Restart Containers If Needed

```bash
rr_api_restart
rr_sim_restart
```

### Start GUI Simulator

```bash
rr_sim_gui
```

### Start Graphics Bridge

```bash
rr_bridge_graphics
```

### Evaluate Best Trained Model in GUI

```bash
sudo docker exec -it autodrive_roboracer_api bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd '${RR_WS_IN}' &&
  source install/setup.bash &&
  ros2 run autodrive_race eval -- \
    --model-path ${RR_PKG_IN}/advanced_best_model/best_model.zip \
    --steps 5000
"
```

## Quick Debug Commands

```bash
rr_topics
rr_lidar
rr_ros_fix
```
