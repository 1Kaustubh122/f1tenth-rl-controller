import os
import time
import rclpy
import numpy as np
import gymnasium as gym
from rclpy.node import Node
from gymnasium import spaces
from stable_baselines3 import PPO
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, Int32
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from ament_index_python.packages import get_package_share_directory


class F110RaceEnv(gym.Env):
    '''
    F110RaceEnv:
    start training: ros2 run autodrive_race qual_round
    
    Subscribe to lidar, speed, and collision count,
    Publish steer and throttle value.
    
    Aim of the controller:
        1. Car will keep driving at preferred speeds [0.15, 0.25, 0.55] (for now).
        2. Algo have to determine, the actions mainly the steer value, so that the car doest crash.
    
    Downsizing total beam from 1081 to 108
    
    Observation space as Lidar and Speed, Speed for first 5_00_000 steps then remove that part.
    
    Action space, 3 throttle value and 11 Steer value
    
    '''
    
    def __init__(self, train=True):
        super(F110RaceEnv, self).__init__()
        self.train = train
        self.node = Node('f110_race_env')
        
        # Subs
        self.lidar_sub = self.node.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, 100)
        
        ## ONLY FOR FIRST 5_00_000 steps
        # self.speed_sub = self.node.create_subscription(
        #     Float32, '/autodrive/roboracer_1/speed', self.speed_callback, 10)
        
        self.collision_sub = self.node.create_subscription(
            Int32, '/autodrive/roboracer_1/collision_count', self.collision_callback, 10)
        
        # Pubs
        self.throttle_pub = self.node.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.node.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # Action Space: [throttle (3 options), steering (5 options)]
        self.action_space = spaces.MultiDiscrete([3, 11])
        
        # Observation Space: Downsampled LiDAR (108 values) + speed (1 value)
        lidar_size = 108 
        self.observation_space = spaces.Box(low=0, high=1, shape=(lidar_size + 2,), dtype=np.float32)
        
        self.lidar_data = None
        # self.speed = None         ## ONLY FOR FIRST 5_00_000 steps
        self.speed = [0, ]          ## FOR REST 5_00_000 steps
        self.collision_count = 0
        self.last_collision_count = 0
        
        self.max_range = 10.0  
        self.max_speed = 10.0   
        self.throttle_levels = [0.15, 0.25, 0.55]  
        self.steering_levels = [-1.0, -0.75, -0.55, -0.35, -0.15, 0, 0.15, 0.35, 0.35, 0.55, 1.0]  
        

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        downsampled = [np.mean(ranges[i:i+10]) for i in range(0, len(ranges), 10)]
        self.lidar_data = np.clip(downsampled, 0, self.max_range) / self.max_range
        
    ## ONLY FOR FIRST 5_00_000 steps
    # def speed_callback(self, msg):
    #     self.speed = msg.data / self.max_speed

    def collision_callback(self, msg):
        self.collision_count = msg.data

    def step(self, action):
        throttle_idx, steering_idx = action
        throttle = self.throttle_levels[throttle_idx]
        steering = self.steering_levels[steering_idx]
        
        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.throttle_pub.publish(throttle_msg)
        self.steering_pub.publish(steering_msg)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        collision = self.collision_count > self.last_collision_count
        self.last_collision_count = self.collision_count
        
        if collision:
            reward = -100
            terminated = True
            truncated = False  
        else:
            reward = 1 + 0.01 * self.speed * self.max_speed
            terminated = False
            truncated = False 
        
        if self.lidar_data is None or self.speed is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = np.concatenate([self.lidar_data, [self.speed]])
        
        info = {} 
        
        ## DEBUG
        # print("STEP OBS SHAPE:", np.shape(obs))
        
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        self.lidar_data = None
        self.speed = None

        while self.lidar_data is None or self.speed is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = np.concatenate([self.lidar_data, [self.speed]])
        return obs, {}

def main(args=None):
    rclpy.init(args=args)
    
    total_timesteps=1000000
    
    checkpoint_path = os.path.join(
        get_package_share_directory('autodrive_race'),
        'checkpoints'
    )
    
    best_model_path = os.path.join(
        get_package_share_directory('autodrive_race'),
        'best_model'
    )
    
    logs_path = os.path.join(
        get_package_share_directory('autodrive_race'),
        'logs'
    )
    
    print(checkpoint_path)

    env = make_vec_env(lambda: F110RaceEnv(train=True), n_envs=1)
    
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_path, name_prefix='ppo_model')
    
    
    checkpoint_path = checkpoint_path + '/ppo_last_model.zip'
    if os.path.exists(checkpoint_path):
        model = PPO.load(checkpoint_path, env=env)
        print("Loaded model from checkpoint.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device="cuda",
            policy_kwargs={"net_arch": [256, 256]}
        )

    
    eval_callback = EvalCallback(env, best_model_save_path=best_model_path, log_path=logs_path, eval_freq=10000)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback], progress_bar=True)
    
    model.save("ppo_race_model")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()