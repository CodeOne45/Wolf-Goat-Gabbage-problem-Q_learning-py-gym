# Wolf-Goat-Gabbage-problem-Q_learning-py-gym (🥦 - 🐐 - 🐺 - 🚣🏽)

## Description

This is a simple implementation of Q-learning algorithm to solve the wolf-goat-cabbage problem. The problem is a well-known puzzle in artificial intelligence, and it is used to illustrate the power of state-space search algorithms. The problem can be stated as follows: A farmer with a wolf, a goat, and a cabbage come to the edge of a river they wish to cross. There is a boat at the river's edge that only the farmer can row. The farmer can take at most one other object besides himself on a crossing, but if the wolf is ever left with the goat, the wolf will eat the goat; similarly, if the goat is left with the cabbage, the goat will eat the cabbage. Devise a sequence of crossings of the river so that all four characters arrive safely on the other side of the river.

## Requirements

- Python 3.6
- gym
- numpy

## Installation

```python
pip3 install gym
pip3 install numpy
```
## Usage

```python
python3 main.py
```

## Analysis

The Q-learning algorithm is a model-free, online, off-policy reinforcement learning method. The main idea of Q-learning is to use the Bellman equation as an iterative update rule for action-value pairs. The algorithm in Wolf-Goat-Cabbage problem is as follows:
- Initialize Q(s,a) arbitrarily
- Repeat (for each episode):
    - Initialize s
    - Repeat (for each step of episode):
        - Choose a from s using policy derived from Q (e.g., epsilon-greedy)
        - Take action a, observe r, s'
        - Q(s,a) <- Q(s,a) + alpha[r + gamma * max(Q(s',a')) - Q(s,a)]
        - s <- s'
    - until s is terminal

