import random
from typing import Optional
import numpy as np
import gymnasium as gym

class DuopolyEnv(gym.Env):

    """
    The Cornout Duopoly

    ### Description:
    There are 2 firms and these firms are 
    """

    def __init__(self, a=400, b=1, cost0=20, cost1=20):
        super(DuopolyEnv, self).__init__()
        # variables
        self.a = a
        self.b = b
        self.cost0 = cost0
        self.cost1 = cost1
        self._max_profit = ((self.a - self.cost0)*(self.a + self.cost0)*2)/(9 * self.b)

        # location of f1 and f2
        self._quantity0 = 0.0
        self._quantity1 = 0.0

        # profit of f1 and f2
        self._profit0 = 0.0
        self._profit1 = 0.0

        self._max_production = a/2 

        self.turn = 0

        # observation space quantity0, quantity1, profit0, profit1
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -self._max_profit, -self._max_profit]),
            high=np.array([self._max_production, self._max_production, self._max_profit, self._max_profit]),
            dtype=np.float64
        )

        # action space, integer
        # it will allow a firm to make an action either decrease by max 25 or increase by max 25
        self.action_space = gym.spaces.Box(
            low = -25,
            high = 25,
            shape=(1,),
            dtype = np.int64
        )

    def step(self, action):
        action = action[0]
        if self.turn == 0:
            expected_quantity = (self.a - self.b * self._quantity1 - self.cost0) / (2 * self.b)
            expected_profit = (self.a - self.b * (expected_quantity + self._quantity1) - self.cost0) * expected_quantity

            # quantity doesnt go negative
            self._quantity0 = max(self._quantity0 + action, 0)
            self._profit0 = (self.a - self.b * (self._quantity0 + self._quantity1) - self.cost0) * self._quantity0

            reward = self._profit0 - expected_profit

        if self.turn == 1:
            expected_quantity = (self.a - self.b * self._quantity0 - self.cost1) / (2 * self.b)
            expected_profit = (self.a - self.b * (expected_quantity + self._quantity0) - self.cost1) * expected_quantity

            # quantity doesnt go negative
            self._quantity1 = max(self._quantity1 + action, 0)
            self._profit1 = (self.a - self.b * (self._quantity0 + self._quantity1) - self.cost1) * self._quantity1

            reward = self._profit1 - expected_profit

        terminated = (self._quantity0 != self._quantity1)
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
        return np.array([self._quantity0, self._quantity1, self._profit0, self._profit1], dtype=np.float64)
    
    def _get_info(self):
        return {"quantity_difference" : self._quantity0 - self._quantity1}

    def render(self):
        print(f"Firm 1 Quantity - profit: {self._quantity0} - {self._profit0}, Firm 2 Quantity - profit: {self._quantity1} - {self._profit1}, ")

gym.register(
    id="gymnasium_env/Duopoly-v0",
    entry_point=DuopolyEnv,
)

env = gym.make("gymnasium_env/Duopoly-v0")

