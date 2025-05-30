from numpy import convolve, ones, mean
import gymnasium as gym

from supervisor_controller_SB3 import CartPoleSupervisor
from utilities import plot_data

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env


def run(train_mode=True):
    env = CartPoleSupervisor()
    model_file = "ppo_cartpole"
    if train_mode:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=40000)
        model.save(model_file)
        print("Model trained and saved!")
    else:
        model = PPO.load(model_file, env=env)
        print("Model loaded!")
    
    obs, info = env.reset()
    env.episode_score = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.episode_score += reward
        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            obs, info = env.reset()

