import random
import gymnasium as gym
import numpy as np
from typing import Optional
import time
import matplotlib.pyplot as plt

class CornoutDuopolyEnv(gym.Env):
    def __init__(self, a=400, b=1, cost=10):
        super().__init__()
        # variables
        self.a = a
        self.b = b
        self.cost = cost

        self._quantity1 = 0
        self._quantity2 = 0
        self._profit1 = 0
        self._profit2 = 0

        self._max_production = int((self.a - self.cost) / 2)

        self.turn = 0

        self._optimal_quantity = (self.a - self.cost) / 3

        self.observation_space = gym.spaces.MultiDiscrete([self._max_production + 1, self._max_production + 1])
        self.action_space = gym.spaces.Discrete(self._max_production * 2 + 1, start=-self._max_production)

        self.agent1 = Agent(self.action_space, self._max_production, 0.1, 0.99, 0.1)
        self.agent2 = Agent(self.action_space, self._max_production, 0.1, 0.99, 0.1)

    def step(self, action):
        action = action - self._max_production
        if self.turn == 0:
            self._quantity1 = np.clip(self._quantity1 + action, 0, self._max_production)

        if self.turn == 1:
            self._quantity2 = np.clip(self._quantity2 + action, 0, self._max_production)

        self._profit2 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity2
        self._profit1 = (self.a - self.cost - self.b * (self._quantity1 + self._quantity2)) * self._quantity1

        terminated = abs(self._quantity1 - self._quantity2) < 1 and abs(self._quantity1 - self._optimal_quantity) < 100
        truncated = False

        reward = (200 if self._optimal_quantity - self._quantity1 == 0 else -abs(self._optimal_quantity - self._quantity1)) if self.turn == 0 else (200 if self._optimal_quantity - self._quantity2 == 0 else -abs(self._optimal_quantity - self._quantity2))

        info = self._get_info()

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._quantity1 = 0
        self._quantity2 = 0
        self._profit1 = 0
        self._profit2 = 0
        self.turn = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        return np.array([self._quantity1, self._quantity2], dtype=np.int64)

    def _get_info(self):
        return {"quantity_difference": abs(self._quantity1 - self._quantity2)}

    def render(self):
        print(f"Firm 1 Quantity - profit: {self._quantity1} - {self._profit1}, Firm 2 Quantity - profit: {self._quantity2} - {self._profit2}")


class Agent:
    def __init__(self, action_space, _max_production, _learning_rate, _discount_factor, _epsilon):
        self.action_space = action_space
        self._max_production = _max_production
        self._learning_rate = _learning_rate
        self._discount_factor = _discount_factor
        self._epsilon = _epsilon

        # First part is for opp quantity, second part is for action that this agent can choose
        self.q_table = np.random.uniform(low=-1, high=1, size=(self._max_production + 1, (self._max_production) * 2 + 1))

    def choose_action(self, state):
        if random.uniform(0, 1) < self._epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, new_state, action, reward):
        max_future_q = np.max(self.q_table[new_state])
        current_q = self.q_table[state, action]
        new_q = current_q + self._learning_rate * (reward + self._discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

gym.register(
    id="gymnassium_env/Duopoly-v1",
    entry_point=CornoutDuopolyEnv,
)

env = gym.make("gymnassium_env/Duopoly-v1")

episodes = 15000
times = []
for episode in range(episodes):
    observation, info = env.reset()
    state = tuple(observation)
    episode_over = False
    start_time = time.time()
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

        env.unwrapped.turn = (env.unwrapped.turn + 1) % 2
        episode_over = done or truncated
        if episode_over:
            break
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds {env.render()}")
    times.append(elapsed_time)

print(np.argmax(env.unwrapped.agent1.q_table[130]))

plt.plot(range(len(times)), times)

plt.xlabel('Episode Index')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Elapsed Time per Episode')

plt.show()