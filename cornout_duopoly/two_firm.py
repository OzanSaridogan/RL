import random
import gymnasium as gym
import numpy as np
from typing import Optional
import time
import matplotlib.pyplot as plt

class CornoutDuopolyEnv(gym.Env):
    """
    Gymnasium environment for Cornout Duopoly model from game theory.
    This environment is for two firms competing in quantity.
    """

    def __init__(self, a=300, b=1, cost=30):
        """
        Initialize the Cornout Duopoly Environment.
        
        Parameters:
        - a: the intercept of the demand function (default 300)
        - b: the slope of the demand function (default 1)
        - cost: the cost of producing one unit of the good (default 30)
        """
        super().__init__()

        # Parameters for the game
        self.a = a
        self.b = b
        self.cost = cost

        # Initial quantities and profits
        self._quantity1 = 0
        self._quantity2 = 0
        self._profit1 = 0
        self._profit2 = 0

        # Maximum production limit (based on the parameters)
        self._max_production = int((self.a - self.cost) / 2)

        # The optimal quantity where profit is maximized
        self._optimal_quantity = (self.a - self.cost) / 3

        # Action and observation space definitions
        self.observation_space = gym.spaces.MultiDiscrete([self._max_production + 1, self._max_production + 1])
        self.action_space = gym.spaces.Discrete(self._max_production * 2 + 1, start=-self._max_production)

        # Initialize agents (player 1 and player 2)
        self.agent1 = Agent(self.action_space, self._max_production, 0.1, 0.99, 0.1)
        self.agent2 = Agent(self.action_space, self._max_production, 0.1, 0.99, 0.1)

    def step(self, action):
        """
        Take one step in the environment. This function calculates the new state, 
        rewards, and checks for termination conditions.

        Parameters:
        - action: the action (quantity adjustment) taken by the agent

        Returns:
        - observation: the new state after taking the action
        - reward: the reward received after performing the action
        - terminated: whether the episode is over
        - truncated: if the episode was cut short
        - info: additional information (e.g., the difference in quantities)
        """
        action = action - self._max_production  # Adjust action to range [-max_production, max_production]

        # Assign action to corresponding agent (Firm 1 or Firm 2)
        if self.turn == 0:
            self._quantity1 = np.clip(self._quantity1 + action, 0, self._max_production)
        if self.turn == 1:
            self._quantity2 = np.clip(self._quantity2 + action, 0, self._max_production)

        # Calculate the profits of both firms based on the quantities and demand function
        self._profit2 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity2
        self._profit1 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity1

        # Define termination condition: when the quantities are close enough to the optimal quantity
        terminated = abs(self._quantity1 - self._quantity2) < 1 and abs(self._quantity1 - self._optimal_quantity) < 100
        truncated = False

        # Define reward: based on how close the quantity is to the optimal quantity
        reward = (200 if self._optimal_quantity - self._quantity1 == 0 else -abs(self._optimal_quantity - self._quantity1)) if self.turn == 0 else (200 if self._optimal_quantity - self._quantity2 == 0 else -abs(self._optimal_quantity - self._quantity2))

        # Gather additional info about the environment
        info = self._get_info()

        # Return the new observation (state) after the action
        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state before starting a new episode.

        Returns:
        - observation: the initial state of the environment
        - info: additional information (e.g., quantity difference)
        """
        self._quantity1 = 0
        self._quantity2 = 0
        self._profit1 = 0
        self._profit2 = 0
        self.turn = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        """
        Get the current state of the environment (quantities of both firms).

        Returns:
        - numpy array: current quantities of firm 1 and firm 2
        """
        return np.array([self._quantity1, self._quantity2], dtype=np.int64)

    def _get_info(self):
        """
        Get additional information about the environment (quantity difference).

        Returns:
        - dict: dictionary containing the absolute difference between the two quantities
        """
        return {"quantity_difference": abs(self._quantity1 - self._quantity2)}

    def render(self):
        """
        Render the current state of the environment, i.e., print the current quantities and profits of both firms.
        """
        print(f"Firm 1 Quantity - profit: {self._quantity1} - {self._profit1}, Firm 2 Quantity - profit: {self._quantity2} - {self._profit2}")


class Agent:
    """
    Class representing a Q-learning agent for the duopoly game.
    The agent learns the optimal quantity to produce over time using Q-learning.
    """

    def __init__(self, action_space, _max_production, _learning_rate, _discount_factor, _epsilon):
        """
        Initialize the agent with parameters for Q-learning.

        Parameters:
        - action_space: the action space of the agent
        - _max_production: maximum number of units the agent can produce
        - _learning_rate: learning rate for Q-learning
        - _discount_factor: discount factor for future rewards
        - _epsilon: exploration rate for epsilon-greedy action selection
        """
        self.action_space = action_space
        self._max_production = _max_production
        self._learning_rate = _learning_rate
        self._discount_factor = _discount_factor
        self._epsilon = _epsilon

        # Initialize the Q-table with random values. Q-table represents the state of the opponent firm.
        self.q_table = np.random.uniform(low=-1, high=1, size=(self._max_production + 1, (self._max_production) * 2 + 1))

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        
        Parameters:
        - state: current state of the environment
        
        Returns:
        - action: chosen action
        """
        if random.uniform(0, 1) < self._epsilon:
            return self.action_space.sample()  # Random action (exploration)
        else:
            return np.argmax(self.q_table[state])  # Best action (exploitation)

    def update_q_table(self, state, new_state, action, reward):
        """
        Update the Q-table using the Q-learning update rule.
        
        Parameters:
        - state: current state of the agent
        - new_state: new state after the action
        - action: action taken by the agent
        - reward: reward received from the environment
        """
        max_future_q = np.max(self.q_table[new_state])  # Maximum Q-value for the next state
        current_q = self.q_table[state, action]  # Current Q-value for the state-action pair
        new_q = current_q + self._learning_rate * (reward + self._discount_factor * max_future_q - current_q)  # Updated Q-value
        self.q_table[state, action] = new_q  # Update Q-table with the new Q-value


# Register the environment in Gymnasium
gym.register(
    id="gymnassium_env/Duopoly-v1",
    entry_point=CornoutDuopolyEnv,
)

# Create the environment instance
env = gym.make("gymnassium_env/Duopoly-v1")

# Number of episodes for training
episodes = 15000
times = []  # List to store the time taken for each episode

# Loop through episodes
for episode in range(episodes):
    observation, info = env.reset()
    state = tuple(observation)  # Initial state as a tuple
    episode_over = False
    start_time = time.time()  # Start timing the episode

    # Run the episode until termination or truncation
    while not episode_over:
        if env.unwrapped.turn == 0:
            action = env.unwrapped.agent1.choose_action(state[1])
            observation, reward, done, truncated, info = env.step(action)
            next_state = tuple(observation)
            env.unwrapped.agent1.update_q_table(state[1], next_state[1], action, reward)
            state = next_state
        else:
            action = env.unwrapped.agent2.choose_action(state[0])
            observation, reward, done, truncated, info = env.step(action)
            next_state = tuple(observation)
            env.unwrapped.agent2.update_q_table(state[0], next_state[0], action, reward)
            state = next_state

        # Switch turn between the two agents (firms)
        env.unwrapped.turn = (env.unwrapped.turn + 1) % 2
        episode_over = done or truncated  # Episode ends if done or truncated
        if episode_over:
            break

    end_time = time.time()  # End timing the episode
    elapsed_time = round(end_time - start_time, 2)  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds {env.render()}")  # Print time and render
    times.append(elapsed_time)  # Store elapsed time for plotting

# Output the best action based on Q-table for a particular state
print(np.argmax(env.unwrapped.agent1.q_table[130]))

# Plot the elapsed time per episode
plt.plot(range(len(times)), times)
plt.xlabel('Episode Index')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Elapsed Time per Episode')
plt.show()
