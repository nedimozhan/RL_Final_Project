from custom_gym_env import CustomEnv
import time
from stable_baselines3 import A2C, DQN, PPO


env = CustomEnv()

model = DQN('MlpPolicy', env, verbose=0, learning_rate=0.0001,
            exploration_fraction=0.995, tensorboard_log="./dqn_tensorboard/")

model.learn(total_timesteps=5e6)
model.save("model/yavsaknedim")
