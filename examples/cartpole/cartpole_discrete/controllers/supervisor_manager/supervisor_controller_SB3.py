import sys
sys.path.append('D:/EDU/Webots/lib/controller/python')


from deepbots.supervisor import CSVSupervisorEnv
from utilities import normalize_to_range

from gymnasium.spaces import Box, Discrete
import numpy as np


class CartPoleSupervisor(CSVSupervisorEnv):
    def __init__(self):
       # Set up gym spaces
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        self.action_space = Discrete(2)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.robot = self.getFromDef("ROBOT")

        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.message_received = None  # Variable to save the messages received from the robot



        # Set up misc
        self.steps_per_episode = 2000  # How many steps to run each episode (changing this messes up the solved condition)
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

        def get_observations(self):
            """
            This get_observation implementation builds the required observation for the CartPole problem.
            All values apart are gathered here from the robot and pole_endpoint objects.
            All values are normalized appropriately to [-1, 1], according to their original ranges.

            :return: Observation: [cart_position, cart_velocity, pole_angle, poleTipVelocity]
            :rtype: list
            """
            # Position on x axis
            cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
            # Linear velocity on x axis
            cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)

            self.message_received = self.handle_receiver()  # update message received from robot, which contains pole angle
            if self.message_received is not None:
                pole_angle = normalize_to_range(float(self.message_received[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
            else:
                # method is called before message_received is initialized
                pole_angle = 0.0

            # Angular velocity y of endpoint
            endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

            return [cart_position, cart_velocity, pole_angle, endpoint_velocity]
        
    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space)]
    
    def get_reward(self,action):
        return 1
    
    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is on the
        arena's edges.

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if self.episode_score > 195.0:
            return True

        if self.message_received is not None:
            pole_angle = round(float(self.message_received[0]), 2)
        else:
            # method is called before message_received is initialized
            pole_angle = 0.0
        if abs(pole_angle) > 0.261799388:  # 15 degrees off vertical
            return True

        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x axis
        if abs(cart_position) > 0.39:
            return True

        return False
    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 1950.0:  # Last 100 episode scores average value
                return True
        return False