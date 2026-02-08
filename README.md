# Reinforcement Learning Projects Portfolio 🤖📚

This repository contains a comprehensive collection of Reinforcement Learning (RL) projects developed as part of an advanced academic course. The projects are structured to reflect the **conceptual and historical progression of Reinforcement Learning**, starting from probabilistic foundations and culminating in policy gradient methods.

Each project is implemented from first principles using Python, with a strong emphasis on:
- theoretical correctness 🧠
- algorithmic clarity ⚙️
- reproducibility 🔁
- interpretability 🔍
- visualization of learning dynamics 📊

---

## Table of Contents 📑

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [Project 1: Markov Chain State Simulation](#project-1-markov-chain-state-simulation)
5. [Project 2: Multi-Armed Bandit Simulation](#project-2-multi-armed-bandit-simulation)
6. [Project 3: Policy Evaluation and Improvement in MDPs](#project-3-policy-evaluation-and-improvement-in-mdps)
7. [Project 4: Policy Iteration Solution for the Tower of Hanoi](#project-4-policy-iteration-solution-for-the-tower-of-hanoi)
8. [Project 5: Monte Carlo and Temporal-Difference Value Prediction](#project-5-monte-carlo-and-temporal-difference-value-prediction)
9. [Project 6: SARSA and Q-Learning for Temporal-Difference Control](#project-6-sarsa-and-q-learning-for-temporal-difference-control)
10. [Project 7: REINFORCE: Monte Carlo Policy Gradient Learning](#project-7-reinforce-monte-carlo-policy-gradient-learning)
11. [Learning Progression Summary](#learning-progression-summary)
12. [References](#references)

---

## Overview 🌍

Reinforcement Learning is a computational framework for modeling and solving **sequential decision-making problems under uncertainty**. An RL agent interacts with an environment over discrete time steps, receiving observations and rewards, and learning a strategy (policy) to maximize cumulative reward.

This repository follows the standard RL taxonomy:
- **Prediction vs Control**
- **Model-based vs Model-free**
- **On-policy vs Off-policy**
- **Value-based vs Policy-based methods**

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

| Technology | Purpose |
|----------|--------|
| Python | Core programming language |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| NetworkX | Graph-based modeling |
| Gymnasium | RL environment framework |

---

# Project 1: Markov Chain State Simulation 🔗

## Theoretical Background

A **Markov Chain** is a stochastic process that satisfies the **Markov property**:

\[
\mathbb{P}(X_{t+1} = s' \mid X_t = s, X_{t-1}, \dots, X_0)
=
\mathbb{P}(X_{t+1} = s' \mid X_t = s)
\]

The system is fully defined by a **transition probability matrix** \( \mathbf{P} \):

\[
\mathbf{P}_{ij} = \mathbb{P}(X_{t+1} = j \mid X_t = i)
\]

Each row of \( \mathbf{P} \) satisfies:

\[
\sum_{j} \mathbf{P}_{ij} = 1
\]

---

## Results 📈

- Correct stochastic transitions
- Empirical trajectories match transition probabilities
- Directed graph visualization improves interpretability

---

# Project 2: Multi-Armed Bandit Simulation 🎰

## Problem Definition

At each time step \( t \), the agent selects an action (arm) \( a_t \) and receives a reward:

\[
r_t \sim \mathcal{R}(a_t)
\]

The goal is to maximize:

\[
\mathbb{E}\left[\sum_{t=1}^{T} r_t \right]
\]

---

## ε-Greedy Action Selection 🎯

\[
a_t =
\begin{cases}
\text{random action}, & \text{with probability } \varepsilon \\\\
\arg\max_a Q_t(a), & \text{with probability } 1 - \varepsilon
\end{cases}
\]

### Incremental Action-Value Update

\[
Q_{t+1}(a) = Q_t(a) + \alpha \left( r_t - Q_t(a) \right)
\]

---

## Results 📊

- Exploration accelerates convergence
- Average reward converges to optimal arm
- Clear exploration–exploitation trade-off

---

# Project 3: Policy Evaluation and Improvement in MDPs 🧩

## Markov Decision Process

An MDP is defined by the tuple:

\[
\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle
\]

---

## Policy Evaluation 🔄

\[
V^{\pi}(s)
=
\sum_{a \in \mathcal{A}} \pi(a \mid s)
\sum_{s' \in \mathcal{S}}
\mathcal{P}(s' \mid s, a)
\left[
\mathcal{R}(s,a,s') + \gamma V^{\pi}(s')
\right]
\]

---

## Policy Improvement ⬆️

\[
\pi'(s)
=
\arg\max_{a}
\sum_{s'}
\mathcal{P}(s' \mid s, a)
\left[
\mathcal{R}(s,a,s') + \gamma V(s')
\right]
\]

---

## Results ✅

- Guaranteed convergence
- Stable optimal policy
- Exact match with dynamic programming theory

---

# Project 4: Policy Iteration Solution for the Tower of Hanoi 🗼

## Algorithm: Policy Iteration

1. Initialize arbitrary policy \( \pi_0 \)
2. **Policy Evaluation** until convergence
3. **Policy Improvement**
4. Repeat until policy is stable

---

## Reward Structure 🎯

| Event | Reward |
|-----|--------|
| Valid move | −1 |
| Invalid move | −10 |
| Goal reached | +100 |

---

## Results 🏆

- Optimal solution obtained
- Minimal number of moves
- Visual verification through animation

---

# Project 5: Monte Carlo and Temporal-Difference Value Prediction 📐

## Monte Carlo Update 🔁

\[
V(s) \leftarrow V(s) + \alpha \left( G_t - V(s) \right)
\]

where:

\[
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}
\]

---

## TD(0) Update ⚡

\[
V(s) \leftarrow V(s) + \alpha
\left(
r_{t+1} + \gamma V(s_{t+1}) - V(s)
\right)
\]

---

## Results 📉📈

- TD converges faster
- MC exhibits higher variance
- Both converge to true value function

---

# Project 6: SARSA and Q-Learning for Temporal-Difference Control 🔀

## SARSA (On-Policy) 🔵

\[
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
\]

---

## Q-Learning (Off-Policy) 🔴

\[
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
\]

---

## Results 🧪

- Q-learning converges to optimal policy
- SARSA learns safer trajectories
- Clear on-policy vs off-policy behavior

---

# Project 7: REINFORCE: Monte Carlo Policy Gradient Learning 🚀

## Objective Function

\[
J(\theta) = \mathbb{E}_{\pi_\theta}
\left[
\sum_{t=0}^{T-1} \gamma^t r_{t+1}
\right]
\]

---

## Gradient Update Rule 🧮

\[
\theta \leftarrow \theta
+
\alpha
\,
G_t
\,
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
\]

---

## Results 🌱

- Policy improves over episodes
- Reward convergence observed
- Correct gradient-based learning dynamics

---

## Learning Progression Summary 🧭

| Order | Project | Core Concept |
|----|--------|-------------|
| 1 | Markov Chain State Simulation | Stochastic Processes |
| 2 | Multi-Armed Bandit Simulation | Exploration vs Exploitation |
| 3 | Policy Evaluation and Improvement in MDPs | Dynamic Programming |
| 4 | Policy Iteration Solution for the Tower of Hanoi | Planning |
| 5 | Monte Carlo and TD Value Prediction | Model-Free Prediction |
| 6 | SARSA and Q-Learning | TD Control |
| 7 | REINFORCE | Policy Gradient Methods |

---

## References 📚

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press  
   http://incompleteideas.net/book/the-book-2nd.html

2. Stanford CS234 – Reinforcement Learning  
   https://web.stanford.edu/class/cs234/

3. OpenAI Spinning Up  
   https://spinningup.openai.com

---

## Final Note ✨

This repository represents a **complete and rigorous journey through Reinforcement Learning**, emphasizing mathematical foundations, algorithmic correctness, and conceptual clarity.
