import gym
from gym import spaces
from car_model import CarModel
import numpy as np
import time


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.normal_env = CarModel()

        self.action_space = spaces.Discrete(self.normal_env.act_dim)
        # print(self.action_space)
        self.observation_space = spaces.Box(low=-100, high=100,
                                            shape=(self.normal_env.obs_dim,), dtype=np.float32)
        # print(self.observation_space)

    def step(self, action):
        observation, reward, done = self.normal_env.step(u=action)

        return observation, reward, done, {}

    def reset(self):
        observation = self.normal_env.reset()
        return observation

    def render(self, mode='human'):
        self.normal_env.render(mode=mode)

    def close(self):
        self.normal_env.close()
