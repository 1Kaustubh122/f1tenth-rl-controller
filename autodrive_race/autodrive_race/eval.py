import os
import rclpy
import numpy as np
import gymnasium as gym
from pathlib import Path
from rclpy.node import Node
from gymnasium import spaces
from stable_baselines3 import PPO
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, Int32
from ament_index_python.packages import get_package_share_directory



class F110RaceEnv(gym.Env):
    def __init__(self, train=True):
        super(F110RaceEnv, self).__init__()
        self.train = train
        self.node = Node('f110_race_env')
        
        # Subs
        self.lidar_sub = self.node.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, 100)
       
    
        # Pubs
        self.reset_pub = self.node.create_publisher(Bool, '/autodrive/reset_command', 10)
        self.throttle_pub = self.node.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.node.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        #                                   [throttle, steer]
        self.action_space = spaces.MultiDiscrete([3, 11])
        
        #                                                       [lidar beams avg]
        self.observation_space = spaces.Box(low=0, high=1, shape=(110, ), dtype=np.float32)
        
   
        self.lidar_data = None
        self.speed = None

        self.throttle = 0.0
        
        self.max_range = 10.0  
        self.max_speed = 10.0   
        self.throttle_levels = [0.15, 0.25, 0.55]  
        self.steering_levels = [-1.0, -0.75, -0.55, -0.35, -0.15, 0, 0.15, 0.35, 0.35, 0.55, 1.0]  
        

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        downsampled = [np.mean(ranges[i:i+10]) for i in range(0, len(ranges), 10)]
        self.lidar_data = np.clip(downsampled, 0, self.max_range) / self.max_range

    def step(self, action):
        throttle_idx, steering_idx = action
        self.throttle = self.throttle_levels[throttle_idx]
        steering = self.steering_levels[steering_idx]
        
        self.throttle_msg = Float32()
        self.throttle_msg.data = float(self.throttle)
        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.throttle_pub.publish(self.throttle_msg)
        self.steering_pub.publish(steering_msg)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
      
        reward = 1 + 0.01 * self.throttle * self.max_speed
        terminated = False
        truncated = False  
        
        if self.lidar_data is None :
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = np.concatenate([self.lidar_data, np.array([0.0])])
        
        info = {}  
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        self.lidar_data = None
        self.speed = None

        while self.lidar_data is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = np.concatenate([self.lidar_data, np.array([0.0])])
        return obs, {}
    
def main(args=None):
    rclpy.init(args=args)

    env = F110RaceEnv(train=False)

    # pkg_share = Path(get_package_share_directory('roboracer_submission'))
    # model_path = pkg_share / 'best_model' / 'best_model' 
    model_path = os.path.join(os.path.dirname(__file__), "best_model/best_model")

    # model_path = 'best_model/best_model.zip'
    model = PPO.load(str(model_path), device="cpu")

    obs, _ = env.reset()
    episode_reward = 0

    ## Need to be adjusted, but lap toic is restricted
    for _ in range(5000):  
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            print(f"Total reward: {episode_reward}")
            obs, _ = env.reset()
            episode_reward = 0

    rclpy.shutdown()

if __name__ == '__main__':
    main()
