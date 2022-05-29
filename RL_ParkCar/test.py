from stable_baselines3 import A2C, DQN, PPO
from custom_gym_env import CustomEnv
import time


env = CustomEnv()
policy = DQN.load('model/yavsaknedim3')

state = env.reset()

while True:
    action = policy.predict(state)
    new_state, reward, done, info = env.step(action=action[0])
    env.render()
    time.sleep(.2)
    if done:
        break
