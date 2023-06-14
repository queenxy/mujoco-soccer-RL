import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time


env = gym.make("Soccer_1v1-v0",render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env,video_folder="./video/")
env.reset()
t = time.time()
for i in range(100):
    action = np.ones(4)
    obs, reward, done,_,info = env.step(action)
    # print(obs,reward,done)
print(time.time()-t)
env.close()