import math
import random
from typing import Optional
import numpy as np
import gymnasium as gym

class DuopolyEnv(gym.Env):

    def __init__(self, a=400, b=1, cost0=20, cost1=20):
        super(DuopolyEnv, self).__init__()
        # variables
        self.a = a
        self.b = b
        self.cost0 = cost0
        self.cost1 = cost1
        self._max_profit = (self.a - self.cost0)*(self.a + self.cost0)/(9 * self.b)

        # location of f1 and f2
        self._quantity0 = 0.0
        self._quantity1 = 0.0

        # profit of f1 and f2
        self._profit0 = 0.0
        self._profit1 = 0.0

        self._max_production = (self.a-self.cost0)/2 if (self.cost0 < self.cost1) else (self.a-self.cost1)/2

        self.turn = 0

        # observation space quantity0, quantity1, profit0, profit1
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([self._max_production, self._max_production], dtype=np.float64),
            dtype=np.float64
        )

        # action space 51 discrete actions from -25 to 25
        self.action_space = gym.spaces.Discrete(51) 

    def step(self, action):
        if self.turn == 0:
            expected_quantity = (self.a - self.b * self._quantity1 - self.cost0) / (2 * self.b)
            expected_profit = (self.a - self.b * (expected_quantity + self._quantity1) - self.cost0) * expected_quantity

            # action is between -25 25
            action_value = (action - 25)
            self._quantity0 = np.clip(self._quantity0 + action_value, 0, self._max_production)
            self._profit0 = (self.a - self.b * (self._quantity0 + self._quantity1) - self.cost0) * self._quantity0

            reward = self._profit0 - expected_profit

        if self.turn == 1:
            expected_quantity = (self.a - self.b * self._quantity0 - self.cost1) / (2 * self.b)
            expected_profit = (self.a - self.b * (expected_quantity + self._quantity0) - self.cost1) * expected_quantity

            action_value = (action - 25)
            self._quantity1 = np.clip(self._quantity1 + action_value, 0, self._max_production)
            self._profit1 = (self.a - self.b * (self._quantity0 + self._quantity1) - self.cost1) * self._quantity1

            reward = self._profit1 - expected_profit

        terminated = ((abs(self._quantity0 - self._quantity1) < 1) and self._quantity0 > 100)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        self.turn = (self.turn + 1) % 2

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._quantity0 = 0.0
        self._quantity1 = 0.0
        
        self._profit0 = 0.0
        self._profit1 = 0.0

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        return np.array([self._quantity0, self._quantity1], dtype=np.float64)
    
    def _get_info(self):
        return {"quantity_difference" : self._quantity0 - self._quantity1}

    def render(self):
        print(f"Firm 1 Quantity - profit: {self._quantity0} - {self._profit0}, Firm 2 Quantity - profit: {self._quantity1} - {self._profit1}, ")

gym.register(
    id="gymnasium_env/Duopoly-v0",
    entry_point=DuopolyEnv,
)

env = gym.make("gymnasium_env/Duopoly-v0")

discrete_os_size = [190] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

#q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [51]))
q_table = np.zeros((discrete_os_size + [51]))

learning_rate = 0.1
discount = 0.95
episodes = 50
epsilon = 0.5

def get_discrete_state(state):
    discrete_state = ((state[0] - env.observation_space.low) / discrete_os_win_size).astype(np.int64)
    discrete_state = np.clip(discrete_state, 0, np.array(discrete_os_size) - 1)
    return tuple(discrete_state)

for episode in range(episodes):
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon: # exploit
            action = env.action_space.sample()
        else:  # exploitation
            action = np.argmax(q_table[discrete_state])

        observation, reward, terminated, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(observation)
        

        if not terminated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state

        episode_over = terminated or truncated
        if episode_over:
            break
    env.render()
    print("done")
