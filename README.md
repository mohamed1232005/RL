# Reinforcement Learning Projects Portfolio 🤖📚

This repository contains a complete and rigorous collection of Reinforcement Learning (RL) projects developed in an academic setting. The projects are organized to reflect the **conceptual, mathematical, and algorithmic progression of Reinforcement Learning**, starting from stochastic processes and ending with policy gradient optimization.

---

## Table of Contents 📑

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Markov Chain State Simulation](#markov-chain-state-simulation)
- [Multi-Armed Bandit Simulation](#multi-armed-bandit-simulation)
- [Policy Evaluation and Improvement in MDPs](#policy-evaluation-and-improvement-in-mdps)
- [Policy Iteration Solution for the Tower of Hanoi](#policy-iteration-solution-for-the-tower-of-hanoi)
- [Monte Carlo and Temporal-Difference Value Prediction](#monte-carlo-and-temporal-difference-value-prediction)
- [SARSA and Q-Learning for Temporal-Difference Control](#sarsa-and-q-learning-for-temporal-difference-control)
- [REINFORCE Monte Carlo Policy Gradient Learning](#reinforce-monte-carlo-policy-gradient-learning)
- [Learning Progression Summary](#learning-progression-summary)
- [References](#references)

---

## Overview 🌍

Reinforcement Learning studies how an agent learns optimal behavior through interaction with an environment. At each time step, the agent:
- observes the current state
- selects an action
- receives a reward
- transitions to a new state

The objective is to learn a policy that maximizes expected cumulative reward over time.

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
| Python | Core programming language |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| NetworkX | Graph modeling |
| Gymnasium | RL environment framework |

---

## Markov Chain State Simulation 🔗

### Description

This project models a **discrete-time Markov chain**, where the system evolves between states according to fixed transition probabilities. The next state depends only on the current state, not on past history.

The system is visualized as a directed weighted graph to improve interpretability.

---

### Mathematical Formulation (Code Cell)

```text
Markov Property:

P(X_{t+1} = s' | X_t = s, X_{t-1}, ..., X_0)
= P(X_{t+1} = s' | X_t = s)

Transition Matrix:

P_ij = P(X_{t+1} = j | X_t = i)

Row normalization:

Sum_j P_ij = 1
```

---

### Results 📈

- State trajectories follow theoretical probabilities
- Visualization highlights dominant transitions
- Correct stochastic behavior observed

---

## Multi-Armed Bandit Simulation 🎰

### Description

This project studies the **exploration–exploitation trade-off** in a stateless environment. At each step, the agent selects one arm and receives a stochastic reward.

---

### Mathematical Formulation (Code Cell)

```text
Objective:

Maximize E[ sum_{t=1}^T r_t ]

Epsilon-Greedy Policy:

With probability ε: select random action
With probability 1 - ε: select argmax_a Q(a)

Incremental Update:

Q_{t+1}(a) = Q_t(a) + α ( r_t - Q_t(a) )
```

---

### Results 📊

- Exploration prevents premature convergence
- Optimal arm identified over time
- Learning curves show reward stabilization

---

## Policy Evaluation and Improvement in MDPs 🧩

### Description

This project introduces **Markov Decision Processes (MDPs)** and applies **dynamic programming** to evaluate and improve policies when the environment model is fully known.

---

### Mathematical Formulation (Code Cell)

```text
MDP Tuple:

(S, A, P, R, γ)

Policy Evaluation:

V_pi(s) =
Sum_a pi(a|s) * Sum_{s'}
P(s'|s,a) * [ R(s,a,s') + γ V_pi(s') ]

Policy Improvement:

pi'(s) =
argmax_a Sum_{s'}
P(s'|s,a) * [ R(s,a,s') + γ V(s') ]
```

---

### Results ✅

- Guaranteed convergence
- Stable optimal policy obtained
- Matches theoretical DP guarantees

---

## Policy Iteration Solution for the Tower of Hanoi 🗼

### Description

The Tower of Hanoi puzzle is formulated as a **deterministic MDP** with a combinatorial state space. A custom environment is implemented, and policy iteration is used to compute the optimal solution.

---

### Mathematical Formulation (Code Cell)

```text
State Space Size:

|S| = 3^n

Policy Iteration Algorithm:

1. Initialize arbitrary policy
2. Evaluate policy until convergence
3. Improve policy greedily
4. Repeat until policy is stable
```

---

### Reward Structure 🎯

| Event | Reward |
|-----|--------|
| Valid move | -1 |
| Invalid move | -10 |
| Goal reached | +100 |

---

### Results 🏆

- Optimal policy discovered
- Minimal number of moves
- Verified through visualization

---

## Monte Carlo and Temporal-Difference Value Prediction 📐

### Description

This project compares **Monte Carlo** and **Temporal-Difference (TD)** learning methods for estimating value functions under a fixed policy.

---

### Mathematical Formulation (Code Cell)

```text
Monte Carlo Update:

V(s) ← V(s) + α ( G_t - V(s) )

Return:

G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ...

TD(0) Update:

V(s) ← V(s) + α ( r_{t+1} + γ V(s_{t+1}) - V(s) )
```

---

### Results 📉📈

- TD converges faster
- MC has higher variance
- Both converge to true values

---

## SARSA and Q-Learning for Temporal-Difference Control 🔀

### Description

This project implements **on-policy** and **off-policy** TD control algorithms to learn optimal action-value functions.

---

### Mathematical Formulation (Code Cell)

```text
SARSA Update (On-Policy):

Q(s_t, a_t) ← Q(s_t, a_t) +
α [ r_{t+1} + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) ]

Q-Learning Update (Off-Policy):

Q(s_t, a_t) ← Q(s_t, a_t) +
α [ r_{t+1} + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t) ]
```

---

### Results 🧪

- Q-learning converges to optimal policy
- SARSA learns safer trajectories
- Clear on-policy vs off-policy distinction

---

## REINFORCE Monte Carlo Policy Gradient Learning 🚀

### Description

This project implements the **REINFORCE algorithm**, a Monte Carlo policy gradient method that directly optimizes a parameterized policy.

---

### Mathematical Formulation (Code Cell)

```text
Objective Function:

J(θ) = E [ sum_{t=0}^{T-1} γ^t r_{t+1} ]

Policy Gradient Update:

θ ← θ + α G_t ∇_θ log π_θ(a_t | s_t)
```

---

### Results 🌱

- Direct policy optimization
- Reward improves across episodes
- Correct gradient-based learning dynamics

---

## Learning Progression Summary 🧭

| Order | Project | Core Concept |
|----|--------|-------------|
| 1 | Markov Chains | Stochastic Processes |
| 2 | Bandits | Exploration |
| 3 | Dynamic Programming | Planning |
| 4 | Policy Iteration | Optimal Control |
| 5 | MC & TD | Prediction |
| 6 | SARSA & Q-Learning | Control |
| 7 | REINFORCE | Policy Gradients |

---

## References 📚

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*
- Stanford CS234
- MIT OpenCourseWare
---

## Final Note ✨

This repository represents a **full and rigorous journey through Reinforcement Learning**, designed for clarity, correctness, and long-term maintainability.
