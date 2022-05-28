from custom_gym_env import CustomEnv
import numpy as np
import time
from stable_baselines3.common.env_checker import check_env

env = CustomEnv()

check_env(env)


s = env.reset()

arr = []
while True:
    action = np.random.randint(0, 9)
    ns, reward, done, info = env.step(action=action)
    print("Reward:", reward)
    arr.append(reward)
    env.render()
    time.sleep(.2)
    if done:
        break

r_mean = np.mean(np.array(arr))
print("r_mean:", r_mean)
