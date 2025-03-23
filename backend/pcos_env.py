import gym
from gym import spaces
import numpy as np

class PCOSDetectionEnv(gym.Env):
    def __init__(self, X, y):
        super(PCOSDetectionEnv, self).__init__()
        self.X = X
        self.y = y
        self.current_index = 0
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        correct = int(action == self.y[self.current_index])
        reward = 1 if correct else -1
        
        self.current_index += 1
        done = self.current_index >= len(self.X)
        
        next_state = self.X[self.current_index] if not done else np.zeros(self.X.shape[1])
        return next_state, reward, done, {}
