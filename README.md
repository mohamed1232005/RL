# Reinforcement Learning Projects Portfolio 🤖📚

This repository contains a complete and rigorous collection of Reinforcement Learning (RL) projects developed in an academic setting. The projects are organized to reflect the **conceptual, mathematical, and algorithmic progression of Reinforcement Learning**, starting from stochastic processes and ending with policy gradient optimization.

All implementations emphasize:
- mathematical correctness
- algorithmic transparency
- reproducibility
- interpretability
- visualization of learning dynamics

---

## Table of Contents 📑

1. Overview  
2. Project Structure  
3. Technologies Used  
4. Markov Chain State Simulation  
5. Multi-Armed Bandit Simulation  
6. Policy Evaluation and Improvement in MDPs  
7. Policy Iteration Solution for the Tower of Hanoi  
8. Monte Carlo and Temporal-Difference Value Prediction  
9. SARSA and Q-Learning for Temporal-Difference Control  
10. REINFORCE: Monte Carlo Policy Gradient Learning  
11. Learning Progression Summary  
12. References  

---

## Overview 🌍

Reinforcement Learning studies how an agent can learn optimal behavior through interaction with an environment. At each time step, the agent:
- observes the current state
- selects an action
- receives a reward
- transitions to a new state

The goal is to learn a policy that maximizes expected cumulative reward over time.

---

## Project Structure 🗂️

```text
.
├── Dynamic Programming for Markov Decision Processes
├── Markov Chain Simulation & Visualization
├── Monte Carlo Policy Gradient Learning
├── Multi-Armed Bandit Problem (Exploration vs Exploitation)
├── On-policy vs Off-policy Temporal-Difference Control
├── Policy Iteration Solution for the Tower of Hanoi
├── Value Function Prediction using Monte Carlo and TD Learning
└── README.md
```

---

## Technologies Used 🛠️

| Technology | Role |
|----------|------|
| Python | Core language |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| NetworkX | Graph modeling |
| Gymnasium | Environment framework |

---

# 1. Markov Chain State Simulation 🔗

## Problem Description

This project models a **discrete-time Markov chain**, a stochastic process in which the next state depends only on the current state.

### Markov Property

The defining property of a Markov chain is:

$$
P(X_{t+1} = s' \mid X_t = s, X_{t-1}, \dots, X_0)
=
P(X_{t+1} = s' \mid X_t = s)
$$

### Transition Matrix

The system is fully described by a transition matrix \( P \), where:

$$
P_{ij} = P(X_{t+1} = j \mid X_t = i)
$$

Each row satisfies:

$$
\sum_{j} P_{ij} = 1
$$

---

## Implementation Details

- Finite discrete state space (bus stops)
- Explicit transition probability matrix
- Random sampling of next states
- Simulation of state trajectories
- Directed weighted graph visualization

---

## Results

- Empirical trajectories follow theoretical probabilities
- Visualization reveals dominant transitions
- Clear demonstration of stochastic dynamics

---

# 2. Multi-Armed Bandit Simulation 🎰

## Problem Description

The multi-armed bandit problem models **decision-making without state transitions**. At each step, the agent chooses one action (arm) and receives a stochastic reward.

The objective is to maximize expected cumulative reward:

$$
\mathbb{E}\left[\sum_{t=1}^{T} r_t \right]
$$

---

## ε-Greedy Strategy

Action selection follows:

$$
a_t =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon
\end{cases}
$$

### Incremental Value Update

$$
Q_{t+1}(a) = Q_t(a) + \alpha \left( r_t - Q_t(a) \right)
$$

---

## Results

- Exploration prevents premature convergence
- Optimal arm identified over time
- Clear exploration–exploitation trade-off

---

# 3. Policy Evaluation and Improvement in MDPs 🧩

## Problem Description

This project introduces **Markov Decision Processes (MDPs)**, defined by the tuple:

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

---

## Policy Evaluation

The state-value function for a policy \( \pi \) is defined as:

$$
V^{\pi}(s)
=
\sum_{a \in \mathcal{A}} \pi(a \mid s)
\sum_{s' \in \mathcal{S}}
P(s' \mid s, a)
\left[
R(s,a,s') + \gamma V^{\pi}(s')
\right]
$$

---

## Policy Improvement

A greedy improvement step yields:

$$
\pi'(s)
=
\arg\max_{a}
\sum_{s'}
P(s' \mid s, a)
\left[
R(s,a,s') + \gamma V(s')
\right]
$$

---

## Results

- Guaranteed convergence to optimal policy
- Stable value function estimates
- Exact alignment with DP theory

---

# 4. Policy Iteration Solution for the Tower of Hanoi 🗼

## Problem Description

This project formulates the Tower of Hanoi as a **deterministic MDP** with a combinatorial state space.

- State space size: \( 3^n \)
- Deterministic transitions
- Explicit reward shaping

---

## Policy Iteration Algorithm

1. Initialize arbitrary policy
2. Evaluate policy until convergence
3. Improve policy greedily
4. Repeat until policy is stable

---

## Reward Design

| Event | Reward |
|-----|--------|
| Valid move | −1 |
| Invalid move | −10 |
| Goal state | +100 |

---

## Results

- Optimal solution discovered
- Minimal number of steps
- Verified via visualization

---

# 5. Monte Carlo and Temporal-Difference Value Prediction 📐

## Problem Description

This project compares **Monte Carlo** and **Temporal-Difference (TD)** learning for value prediction under a fixed policy.

---

## Monte Carlo Update

$$
V(s) \leftarrow V(s) + \alpha \left( G_t - V(s) \right)
$$

where the return is:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}
$$

---

## TD(0) Update

$$
V(s) \leftarrow V(s) + \alpha \left( r_{t+1} + \gamma V(s_{t+1}) - V(s) \right)
$$

---

## Results

- TD converges faster
- MC exhibits higher variance
- Both converge to true value function

---

# 6. SARSA and Q-Learning for TD Control 🔀

## SARSA (On-Policy)

$$
Q(s_t, a_t)
\leftarrow
Q(s_t, a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma Q(s_{t+1}, a_{t+1})
-
Q(s_t, a_t)
\right]
$$

---

## Q-Learning (Off-Policy)

$$
Q(s_t, a_t)
\leftarrow
Q(s_t, a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma \max_a Q(s_{t+1}, a)
-
Q(s_t, a_t)
\right]
$$

---

## Results

- Q-learning converges to optimal policy
- SARSA learns safer trajectories
- Clear on-policy vs off-policy distinction

---

# 7. REINFORCE: Monte Carlo Policy Gradient Learning 🚀

## Objective Function

$$
J(\theta) =
\mathbb{E}_{\pi_\theta}
\left[
\sum_{t=0}^{T-1} \gamma^t r_{t+1}
\right]
$$

---

## Policy Gradient Update

$$
\theta \leftarrow \theta
+
\alpha
\,
G_t
\,
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

---

## Results

- Direct policy optimization
- Reward improvement across episodes
- Correct gradient-based learning dynamics

---

## Learning Progression Summary 🧭

| Order | Project | Core Concept |
|----|--------|-------------|
| 1 | Markov Chains | Stochastic Processes |
| 2 | Bandits | Exploration |
| 3 | DP for MDPs | Planning |
| 4 | Policy Iteration | Optimal Control |
| 5 | MC & TD | Prediction |
| 6 | SARSA & Q-Learning | Control |
| 7 | REINFORCE | Policy Gradients |

---

## References 📚

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- Stanford CS234
- MIT OpenCourseWare
- OpenAI Spinning Up

---

## Final Note ✨

This repository represents a **full theoretical and practical journey through Reinforcement Learning**, from probabilistic foundations to modern policy optimization.
