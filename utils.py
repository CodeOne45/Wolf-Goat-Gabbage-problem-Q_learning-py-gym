from typing import Callable
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

# Fonction to convert state to tuple because dictionary keys must be immutable
def state_to_tuple(state):
    return tuple(tuple(sorted(subset)) for subset in state)
    
def show_render(img, render,sleep=0.1):
    img.set_data(render) 
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(sleep)

# Fonction for rendering the environment with a given policy to
# see if learned policy is correct and will reach the goal
def rendering(env, policy: Callable, episodes = 10):
    for episode in range(1, episodes+1):
        state = state_to_tuple(env.reset())
        done = False
        print("Episode: ", episode)
        while not done:
            p = policy(state)
            action = int(p)
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_tuple(next_state)
            state = next_state
            if done:
                print("Action: ", action)
                print("Done: ", done)
                print("Reward: ", reward)
                print("Next State: ", next_state)
                print("")
                break

def testing(env, action_values):
    state, _ = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        step += 1
        action = np.argmax(action_values[state])
        next_state, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        if not done:
            state = next_state
        else:
            print(f"Episode finished after {step} timesteps, earn {total_reward}")
        

from matplotlib.tri import Triangulation

def extract_data(probs):
    valuesW = np.array([p[0] for p in probs])
    valuesS = np.array([p[1] for p in probs])
    valuesE = np.array([p[2] for p in probs])
    valuesN = np.array([p[3] for p in probs])

    return [valuesN, valuesE, valuesS, valuesW]

def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [ trianglesN, trianglesE, trianglesS, trianglesW]]


def heatmap(probs, M=4, N=4):
    values = extract_data(probs)

    triangul = triangulation_for_triheatmap(M, N)

    norms = [plt.Normalize(-0.5, 1) for _ in range(4)]
    fig, ax = plt.subplots()

    imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
            for t, val in zip(triangul, values)]
    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(M):
            for j in range(N):
                v = val[j*M + i]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w', ha='center', va='center')
    cbar = fig.colorbar(imgs[0], ax=ax)
    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    plt.tight_layout()
    plt.show()        
        

def plot_stats(stats,smooth=10):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-smooth:i+smooth]) for i in range(smooth, len(vals)-smooth)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()         
        
        
import gymnasium
from gymnasium.envs.toy_text import frozen_lake
from gymnasium.envs.toy_text.utils import categorical_sample

class SimFrozenLake(frozen_lake.FrozenLakeEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def simulate_step(self, s, a ):
        transitions = self.P[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        
        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})