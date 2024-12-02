import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

class CournotEnv(gym.Env):
    def __init__(self, a=400, b=1, cost=20):
        super().__init__()

        self.q1 = 0
        self.q2 = 0  # Initial q2 (opponent's quantity)
        self.cost = cost
        self.a = a
        self.b = b

        self._max_production = int(2 * (self.a - self.cost) / 3)

        # Action space determines how much to increase or decrease production
        self.action_space = gym.spaces.Discrete(n=101, start=-50)

        self.observation_space = gym.spaces.Discrete(self._max_production + 1)  # Observation space based on q1

    def step(self, action):
        self.q1 = np.clip(self.q1 + action, 0, self._max_production)
        self.q2 = self.best_response(self.q1)
        Q = self.q1 + self.q2
        P = max(0, self.a - self.b * Q)
        profit1 = P * self.q1 - self.cost * self.q1
        profit2 = P * self.q2 - self.cost * self.q2

        obs = self.q1
        truncated = False

        # Reward is based on profit difference
        reward = (profit1 - profit2)/1000
        terminated = (abs(self.q1 - self.q2) < 2) and self.q1 > 20
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q1 = 0
        self.q2 = 0
        return self.q1, {}

    def best_response(self, q1):
        # Firm 2's best response against Firm 1
        return max(0, (self.a - self.cost - self.b * q1) / (2 * self.b))


# Initialize the environment
env = CournotEnv()

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning parameters
alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01


firm1_quantities = []
firm2_quantities = []
episode_rewards = []


for episode in range(100):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        
        next_state, reward, terminated, truncated, _ = env.step(action)

        
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] += alpha * (reward + gamma * next_max - q_table[state, action])

        
        state = next_state
        
        done = terminated or truncated

    
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(episode_reward)

    # Log quantities
    firm1_quantities.append(env.q1)
    firm2_quantities.append(env.q2)

    print(f"Episode {episode + 1}:, q1 = {env.q1}, q2 = {env.q2}")

