"""
More runners for continuous RL algorithms can be added here.
"""
import DDPG_runner
import DDPG_runner_SB3
# Modify these constants if needed.
EPISODE_LIMIT = 10000
STEPS_PER_EPISODE = 200  # How many steps to run each episode (changing this messes up the solved condition)

if __name__ == '__main__':
    DDPG_runner_SB3.run(True) # True to train, False to test
