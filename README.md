# Reinforcement Learning Navigation Simulator

A grid-based simulation where an autonomous agent learns to navigate randomly generated environments using Q-learning. Built to explore how reinforcement learning handles uncertainty compared to classical path planning.

---

## What It Does

- Generates a random grid world with randomized size (5×5 to 10×10), obstacles, start, and goal
- Trains a Q-learning agent to navigate from start to goal using rewards and penalties
- Introduces **20% stochastic action noise** — every step, there is a 20% chance the agent takes a random action instead, simulating real-world uncertainty
- Compares the learned policy against a **BFS shortest-path baseline** to evaluate how close the agent gets to optimal navigation
- Visualizes the learned path vs the optimal path, and plots reward convergence over training

---

## How It Works

### Q-Learning
The agent starts with no knowledge of the grid. It explores randomly at first and gradually learns which actions lead to rewards. After each move, it updates a Q-table using the Bellman equation:

```
Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s')) - Q(s, a))
```

- **Learning rate (α):** 0.1
- **Discount factor (γ):** 0.99
- **Episodes:** 2000
- **Epsilon decay:** 1.0 → 0.01 (explores less over time)

### Reward Structure
| Event | Reward |
|---|---|
| Reach goal | +100 |
| Normal step | -1 |
| Hit wall or boundary | -10 |

### BFS Baseline
Breadth-First Search finds the guaranteed shortest path through the known grid. The agent's learned path is compared against this to measure how well Q-learning converges toward optimal behavior — especially under stochastic conditions.

---

## Project Structure

```
├── grid_world.py      # GridWorld environment, Q-learning training, BFS
├── visualize.py       # Visualizes learned path vs BFS path and reward convergence
├── experiment.py      # Runs evaluation across multiple environments
└── README.md
```

---

## How To Run

**Requirements**
```
pip install numpy matplotlib
```

**Visualize one environment**
```
python visualize.py
```

**Run evaluation across multiple environments**
```
python experiment.py
```

---

## Results

The agent consistently learns near-optimal navigation policies despite the stochastic noise. Early episodes show highly negative rewards from random exploration. By around episode 200–400, reward curves stabilize and the learned path converges toward the BFS optimal path.

Example output:
```
Grid Size: 8x8 | ✅ | RL: 8 steps | BFS: 6 steps | Final reward: 95.0
```

---

## Why BFS and Not Just RL?

BFS finds the shortest path when the full environment is known and movement is deterministic. Q-learning is useful when the agent must **discover** the environment through interaction and adapt to uncertainty. Using both shows the tradeoff between classical planning and learned policies — and gives a concrete way to measure how well the agent performs.

---

## Built With

- Python
- NumPy
- Matplotlib
