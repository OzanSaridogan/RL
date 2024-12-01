import random
from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class CornoutDuopolyEnv(gym.Env):

    def __init__(self, a = 400, b = 1, cost = 10):
        super(CornoutDuopolyEnv, self).__init__()

        # variables
        self.a = a
        self.b = b
        self.cost = cost

        # quantity of firms
        self._quantity1 = 0
        self._quantity2 = 0

        # profits
        self._profit1 = 0
        self._profit2 = 0

        # maximum amount that a firm can produce
        self._max_production = int(((self.a - self.cost)/2))

        # play of turn
        self.turn = 0

        # observation space for quantity1 and quantity2
        self.observation_space = gym.spaces.MultiDiscrete(nvec=([self._max_production, self._max_production]))

        # action space is discrete from -75 to 75 so that means a firm can increase or decrease the quantity by 75
        self.action_space = gym.spaces.Discrete(start=-75, n=150)

    def step(self, action):

        if self.turn == 0:
            expected_quantity1 = (self.a - self.cost - self.b * (self._quantity2)) / (2 * self.b)
            expected_profit1 = (self.a - self.cost - self.b * (expected_quantity1 + self._quantity2)) * expected_quantity1
            
            self._quantity1 = np.clip(self._quantity1 + action, 0, self._max_production) # make new quantity1
            self._profit1 = (self.a - self.b * (self._quantity1 + self._quantity2) - self.cost) * self._quantity1  # profit of making quantity1

            reward = self._profit1 - expected_profit1

        if self.turn == 1:
            expected_quantity2 = (self.a - self.cost - self.b * (self._quantity1)) / (2 * self.b)
            expected_profit2 = (self.a - self.cost - self.b * (expected_quantity2 + self._quantity1)) * expected_quantity2

            self._quantity2 = np.clip(self._quantity2 + action, 0, self._max_production) # make new quantity2 
            self._profit2 = (self.a - self.b * (self._quantity1 + self._quantity2) - self.cost) * self._quantity2 # profit of making quantity2

            reward = self._profit2 - expected_profit2

        terminated = (abs(self._quantity1 - self._quantity2) < 2) and self._quantity2 > 100
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        self.turn = (self.turn + 1) % 2

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._quantity1 = 0
        self._quantity2 = 0
            
        self._profit1 = 0
        self._profit2 = 0

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _get_obs(self):
        return np.array([self._quantity1, self._quantity2], dtype=np.int64)
    
    def _get_info(self):
        return {"quantity_difference" : self._quantity1 - self._quantity2}
    
    def render(self):
        print(f"Firm 1 Quantity - profit: {self._quantity1} - {self._profit1}, Firm 2 Quantity - profit: {self._quantity2} - {self._profit2}, last turn = {self.turn}")


gym.register(
    id="gymnassium_evnv/Duopoly-v1",
    entry_point=CornoutDuopolyEnv,
)


env = gym.make("gymnassium_evnv/Duopoly-v1")

print(env.reset())

q_table = np.zeros((196, 196, env.action_space.n))

# hyperparameters
learning_rate = 0.1
discount = 0.95
episodes = 150000
epsilon = 0.1

episode_rewards = []

# Inside your main loop:
for episode in range(episodes):
    epsilon = max(0.01, epsilon * 0.995)
    observation, info = env.reset()
    state = tuple(observation)
    episode_over = False
    total_episode_reward = 0
    
    while not episode_over:
        if random.uniform(0, 1) < epsilon:  # explore
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0], state[1]])  # exploit

        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        new_state = tuple(observation)
        
        if not terminated:
            max_future_q = np.max(q_table[new_state[0], new_state[1]])
            current_q = q_table[state[0], state[1], action]

            new_q = current_q + learning_rate * (reward + discount * max_future_q - current_q)
            q_table[state[0], state[1], action] = new_q

        state = new_state

        episode_over = terminated or truncated
        total_episode_reward += reward

        if episode_over:
            break
    env.render()
    episode_rewards.append(total_episode_reward)
    print(f"Episode {episode}: Episode Reward: {total_episode_reward}")

plt.plot(range(episodes), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.show()

average_reward = sum(episode_rewards) / len(episode_rewards)
print(f"Average Reward over {episodes} episodes: {average_reward}")
