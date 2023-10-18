import numpy as np
from tqdm import tqdm
from typing import Callable
from WolfGoatCabbageEnv import WolfGoatCabbageEnv
from utils import *

env = WolfGoatCabbageEnv()

# Create action-value dictionary (Q table)
action_values = {}

def target_policy(state):
    actions = [(state, action) for action in range(16)]
    return max(actions, key=lambda x: action_values.get(x, 0))[1]

def exploratory_policy(state ):
    return np.random.choice(16)



def q_learning(exploratory_policy, target_policy, action_values, episodes, alpha=0.1, gamma=0.99):
    stats = {'Returns': []}

    for episode in tqdm(range(episodes)):
        min_epsilon=0.1
        max_epsilon=1.0
        state = state_to_tuple(env.reset())
        done = False
        ep_return = 0
        while not done:
            action = exploratory_policy(state)

            next_state, reward, done, _ = env.step(action)
            ep_return += reward
            next_state = state_to_tuple(next_state)
            next_action = target_policy(next_state) if not done else None

            # Update Q-values using Q-learning equation
            current_q = action_values.get((state, action), 0)
            next_max_q = action_values.get((next_state, next_action), 0)

            updated_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
            action_values[(state, action)] = updated_q

            state = next_state
        stats['Returns'].append(ep_return)
    return stats




