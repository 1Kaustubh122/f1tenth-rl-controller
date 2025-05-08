
import re
import os
import rclpy
import numpy as np
import gymnasium as gym
from pathlib import Path
from rclpy.node import Node
from gymnasium import spaces
from stable_baselines3 import PPO
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from ament_index_python.packages import get_package_share_directory


class F110RaceEnv(gym.Env):
    def __init__(self, train=True):
        super(F110RaceEnv, self).__init__()
        self.train = train
        self.node = Node('f110_race_env')
        
        # Subs
        self.lidar_sub = self.node.create_subscription(
            LaserScan, 
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback, 
            100
        )
        
        self.last_lap_sub = self.node.create_subscription(
            Float32, 
            '/autodrive/roboracer_1/last_lap_time', 
            self.lap_time_callback, 
            10
        )
        
        self.collision_sub = self.node.create_subscription(
            Int32, 
            '/autodrive/roboracer_1/collision_count', 
            self.collision_callback, 
            10
        )
        
        self.lap_count_subs = self.node.create_subscription(
            Int32,
            '/autodrive/roboracer_1/lap_count',
            self.lap_count_callback,
            10
        )
        
        
        # Pubs
        self.throttle_pub = self.node.create_publisher(
            Float32, 
            '/autodrive/roboracer_1/throttle_command',
            10)
        
        self.steering_pub = self.node.create_publisher(
            Float32, 
            '/autodrive/roboracer_1/steering_command',
            10
        )
        
        self.action_space = spaces.Discrete(25)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(180, ), dtype=np.float32)
        
        ## Dec Params
        self.lidar_data = None
        self.speed = None
        self.last_lap_time = None
        self.best_lap_time = None
        self.collision_count = 0
        self.last_collision_count = 0
        self.throttle = 0.0
        self.total_lap_count = 0.0
        self.last_lap_count = 0.0
        
        ## Params
        self.max_range = 10.0  
        self.max_speed = 10.0   
        
        ## Throttle, steer
        self.actions = [
            (0.15, -0.15),
            (0.15, 0.15),
            (0.15, -0.35),
            (0.15, 0.35),
            (0.15, -0.55),
            (0.15, 0.55),
            (0.15, -0.75),
            (0.15, 0.75),
            (0.15, -1.0),
            (0.15, 1.0),
            (0.25, -0.15),
            (0.25, 0.15),
            (0.25, -0.35),
            (0.25, 0.35),
            (0.25, -0.55),
            (0.25, 0.55),
            (0.25, -0.75),
            (0.25, 0.75),
            (0.25, -1.0),
            (0.25, 1.0),
            (0.25, 0.0),
            (0.45, 0.0),
            (0.55, 0.0),
            (0.65, 0.0),
            (0.75, 0.0)
        ]  
    
    def lap_count_callback(self, msg):
        self.total_lap_count = msg.data
        
    def lap_time_callback(self, msg):
        self.last_lap_time = msg.data
        if self.best_lap_time == None and self.last_lap_time > 0:
            self.best_lap_time = msg.data
        
    
    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        step = len(ranges) // 180
        self.lidar_data = np.clip(ranges[::step][:180], 0, self.max_range) / self.max_range

    def collision_callback(self, msg):
        self.collision_count = msg.data

    def step(self, action):
        reward = 0
        self.throttle, self.steer = self.actions[action]
        
        self.throttle_msg = Float32()
        self.throttle_msg.data = float(self.throttle)
        
        self.steer_msg = Float32()
        self.steer_msg.data = float(self.steer)
        
        self.throttle_pub.publish(self.throttle_msg)
        self.steering_pub.publish(self.steer_msg)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        is_better_lap = self.last_lap_time < self.best_lap_time
            
        collision = self.collision_count > self.last_collision_count
        self.last_collision_count = self.collision_count
        
        truncated = False
        terminated = False
        if collision:
            reward = -50
            terminated = True
        else:
            if self.total_lap_count > self.last_lap_count:
                self.last_lap_count = self.total_lap_count
                if is_better_lap:
                    self.best_lap_time = self.last_lap_time
                    reward += 50
                # else:
                #     reward -= 25        ## Maybe later in training
                
            reward += 1 + 0.01 * self.throttle * self.max_speed
        
        if self.total_lap_count % 12 == 0:
            truncated = True
        
        if self.lidar_data is None :
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = np.concatenate([self.lidar_data])
        
        info = {}  
        
        ## DEBUG
        # print("STEP OBS SHAPE:", np.shape(obs))
        # print(f'''
        # curr lap count: {self.total_lap_count}, last lap: {self.last_lap_count}
        # lap times:      {self.last_lap_time}     best {self.best_lap_time}  better lap: {is_better_lap}
        # collision:      {collision} 
        # ''')
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        self.lidar_data = None
        self.speed = None

        while self.lidar_data is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = np.concatenate([self.lidar_data])
        return obs, {}

def main(args=None):
    rclpy.init(args=args)
    
    total_timesteps=1000000
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    workspace_root = os.path.dirname(install_dir)
    src_dir = os.path.join(workspace_root, 'autodrive_devkit/src', 'autodrive_race')
    
    checkpoint_dir = Path(os.path.join(src_dir, 'checkpoints'))
    best_model_dir = Path(os.path.join(src_dir, 'best_model'))
    logs_dir = Path(os.path.join(src_dir, 'logs'))
    
    
    print(checkpoint_dir)

    for d in (checkpoint_dir, best_model_dir, logs_dir):
        os.makedirs(d, exist_ok=True)


    ckpt_files = list(checkpoint_dir.glob('ppo_model_*_steps.zip'))
    latest_ckpt = None
    max_steps = -1
    pattern = re.compile(r'ppo_model_(\d+)_steps\.zip')

    for f in ckpt_files:
        m = pattern.match(f.name)
        if m:
            steps = int(m.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_ckpt = f
                

    env = make_vec_env(lambda: F110RaceEnv(train=True), n_envs=1)
    

    if latest_ckpt and latest_ckpt.exists():
        print(f"Resuming from checkpoint: {latest_ckpt.name} ({max_steps} steps)")
        model = PPO.load(str(latest_ckpt), env=env, device="cpu")
    else:
        print("No checkpoint found; starting new training.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            device="cpu",
        )


    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoint_dir),
        name_prefix='ppo_model'
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(best_model_dir),
        log_path=str(logs_dir),
        eval_freq=10_000
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    final_path = checkpoint_dir / 'ppo_race_model_final.zip'
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    rclpy.shutdown()

if __name__ == '__main__':
    main()