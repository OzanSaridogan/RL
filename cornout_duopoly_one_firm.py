import random
from typing import Optional
import numpy as np
import gymnasium as gym

class OneDuopoly(gym.Env):

    def __init__(self, a = 400, b = 1, cost = 10):
        
        # variables
        self.a = a
        self.b = b
        self.cost = cost

        self._quantity1 = 0
        self._quantity2 = 0

        self._profit1 = 0
        self._profit2 = 0

        self._max_production = int(2*(self.a - self.cost)/3)
        self._max_profit = int(self.a - self.cost - b*self._max_production) * self._max_production

        self.observation_space = gym.spaces.Discrete(self._max_production + 1)
        self.action_space = gym.spaces.Discrete(n = 101, start=-50)

    def step(self, action):
        expected_quantity1 = (self.a - self.cost - self.b * self._quantity2) / (2 * self.b)
        expected_profit1 = (self.a - self.cost - self.b * (expected_quantity1 + self._quantity2)) * expected_quantity1

        self._quantity1 = np.clip(self._quantity1 + action, 0, self._max_production)
        self._profit1 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity1

        self._quantity2 = (self.a - self.cost - self.b * self._quantity1) / (2 * self.b)
        self._profit2 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity2
        reward = self._profit1 + self._profit2

        terminated = (abs(self._quantity1 - self._quantity2) < 2) and self._quantity1 > 20
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._quantity1 = 0
        self._profit1 = 0

        self._quantity2 = 0
        self._profit2 = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _get_obs(self):
        return self._quantity1
    
    def _get_info(self):
        return {}
    
    def render(self):
        print(f"Firm 1 Quantity - profit: {self._quantity1} - {self._profit1}, Firm 2 Quantity - profit: {self._quantity2} - {self._profit2}")


gym.register(
    id="gymnassium_evnv/Duopoly-v1",
    entry_point=OneDuopoly,
)


env = gym.make("gymnassium_evnv/Duopoly-v1")

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# hyperparameters
learning_rate = 0.1
discount = 0.99
episodes = 100
epsilon = 0.1

episode_rewards = []

for episode in range(episodes):
    observation, info = env.reset()
    state = observation
    episode_over = False
    total_episode_reward = 0
    
    while not episode_over:
        if random.uniform(0, 1) < epsilon:  # explore
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])  # exploit

        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        new_state = observation
        
        if not terminated:
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state, action]

            new_q = current_q + learning_rate * (reward + discount * max_future_q - current_q)
            q_table[state, action] = new_q

        state = new_state

        episode_over = terminated or truncated
        total_episode_reward += reward

        

        if episode_over:
            break
    env.render()
    episode_rewards.append(total_episode_reward)
    print(f"Episode {episode}: Episode Reward: {total_episode_reward}")

average_reward = sum(episode_rewards) / len(episode_rewards)
print(f"Average Reward over {episodes} episodes: {average_reward}")

        
