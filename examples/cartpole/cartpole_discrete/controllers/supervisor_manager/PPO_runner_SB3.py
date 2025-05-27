from numpy import convolve, ones, mean
import gymnasium as gym

from supervisor_controller_SB3 import CartPoleSupervisor
from utilities import plot_data

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env


def run():
    env = CartPoleSupervisor()

    # Verify that the environment is working as a gym-style env
    #check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=40000)
    model.save("ppo_cartpole")


    obs = env.reset()
    env.episode_score = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.episode_score += reward
        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            obs = env.reset()
