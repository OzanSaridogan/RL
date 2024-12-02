import numpy as np
import random
import gymnasium as gym

class CournotEnv(gym.Env):
    def __init__(self, a=400, b=1, cost=20):
        super().__init__()

        self.q1 = 0
        self.q2 = 0  # Initial q2 (opponent's quantity)
        self.cost = cost
        self.a = a
        self.b = b

        # Define action space for Firm 1
        self.action_space = gym.spaces.Discrete(101)  # 0 to 100 quantities for Firm 1
        self.observation_space = gym.spaces.Discrete(101)  # Observation space based on q1
        
    def step(self, action):
        q1 = action  # Firm 1 chooses quantity q1
        q2 = self.best_response(q1)  # Firm 2 chooses its best response
        Q = q1 + q2
        P = max(0, self.a - self.b * Q)
        profit1 = P * q1 - self.cost * q1  # Profit for Firm 1
        
        state = np.array([q1, q2], dtype=np.float32)
        reward = profit1
        done = False  # Typically ends after a fixed number of steps
        return state, reward, done, {}

    def reset(self):
        self.q1 = 0  # Start Firm 1 at zero production
        self.q2 = self.best_response(self.q1)  # Firm 2 responds optimally
        return np.array([self.q1, self.q2], dtype=np.float32)

    def best_response(self, q1):
        # Firm 2's best response against Firm 1
        return (self.a - self.cost - self.b * q1) / (2 * self.b)

# Initialize the environment
env = CournotEnv(a=400, b=1, cost=20)

# Q-learning parameters
alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1  # Exploration rate
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

# Initialize Q-table (Discrete state space)
q_table = np.zeros((101, 101))  # 101 actions for Firm 1, and 101 possible states (0 to 100 for q1)

def choose_action(state):
    # Discretize the state[0] to be an integer index
    state_idx = int(state[0])  # Ensure we are using an integer index for q1
    
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration: Choose random action
    else:
        return np.argmax(q_table[state_idx, :])  # Exploitation: Choose action based on Q-table

# Training loop for Q-learning
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment at the start of each episode
    done = False

    for step in range(max_steps):
        action = choose_action(state)  # Choose action using epsilon-greedy policy
        next_state, reward, done, _ = env.step(action)

        # Discretize the state for Q-table indexing
        state_idx = int(state[0])
        next_state_idx = int(next_state[0])

        # Update Q-table using the Q-learning update rule
        old_value = q_table[state_idx, action]
        next_max = np.max(q_table[next_state_idx, :])
        q_table[state_idx, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state  # Update state

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon to reduce exploration over time

# Testing phase: After training, use the learned Q-table to make decisions
for episode in range(5):  # Run a few episodes to test the learned behavior
    state = env.reset()
    done = False

    print(f"Episode {episode + 1}")
    for step in range(max_steps):
        action = np.argmax(q_table[int(state[0]), :])  # Choose the best action based on the learned Q-table
        next_state, reward, done, _ = env.step(action)
        print(f"Step {step + 1}: q1 = {state[0]}, q2 = {state[1]}, action = {action}, reward = {reward}")

        state = next_state

        if done:
            print("Episode finished")
            break
