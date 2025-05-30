from numpy import convolve, ones
import gymnasium as gym
import torch

from supervisor_controller_SB3 import CartPoleSupervisor
from utilities import plot_data

from stable_baselines3 import DDPG, PPO, TD3, SAC
from stable_baselines3.common.env_checker import check_env


def run(train_mode=True):   # Add a flag

    env = CartPoleSupervisor()
    model_file = "SAC_cartpole_model_256"

    check_env(env)  # Check if the environment follows the Gymnasium API
    policy_kwargs = dict(net_arch=[256, 256])
    learning_rate = 1e-4  # Set a learning rate for DDPG

    if train_mode:
        model = SAC("MlpPolicy", 
                    env, 
                    verbose=1, 
                    tensorboard_log="./ddpg_cartpole_tensorboard/",
                    policy_kwargs=policy_kwargs,
                    learning_rate=learning_rate,
                    device="cuda" if torch.cuda.is_available() else "cpu")
        model.learn(total_timesteps=100000)
        model.save(model_file)
        print("Model trained and saved!")
    else:
        model = SAC.load(model_file, env=env)
        print("Model loaded!")

    obs, info = env.reset()
    env.episode_score = 0

    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.episode_score += reward
        if done:
            print("Episode score:", env.episode_score)
            env.episode_score = 0

