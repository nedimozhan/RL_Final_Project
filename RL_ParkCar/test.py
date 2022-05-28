from custom_gym_env import CustomEnv
import time
env = CustomEnv()

env.reset()

while True:
    action = 0
    ns, r, done, info = env.step(action=action)
    env.render()
    time.sleep(.2)
    if done:
        break
