# Reinforcement Learning Projects Portfolio

This repository contains a comprehensive collection of Reinforcement Learning (RL) projects developed as part of an advanced academic course. The projects are structured to reflect the **conceptual and historical progression of Reinforcement Learning**, starting from probabilistic foundations and culminating in policy gradient methods.

Each project is implemented from first principles using Python, with a strong emphasis on:
- theoretical correctness
- algorithmic clarity
- reproducibility
- interpretability
- visualization of learning dynamics

---

## Table of Contents

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

## Overview

Reinforcement Learning is a computational framework for modeling and solving sequential decision-making problems under uncertainty. An RL agent interacts with an environment over discrete time steps, receiving observations and rewards, and learning a strategy (policy) to maximize cumulative reward.

This repository follows the standard RL taxonomy:
- **Prediction vs Control**
- **Model-based vs Model-free**
- **On-policy vs Off-policy**
- **Value-based vs Policy-based methods**

---

## Project Structure

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

Each folder contains:
- Python notebooks or scripts
- Visualizations
- Experimental results
- Algorithm implementations

---

## Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Core programming language |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| NetworkX | Graph-based modeling |
| Gymnasium | RL environment framework |
| Python Standard Library | Control flow and data structures |

---

# Project 1: Markov Chain State Simulation

## Problem Definition

A **Markov Chain** is a stochastic process defined by the **Markov property**:

\[
P(X_{t+1} = s' \mid X_t = s, X_{t-1}, \dots, X_0) = P(X_{t+1} = s' \mid X_t = s)
\]

The system is fully characterized by a **transition probability matrix**:

\[
P_{ij} = P(X_{t+1} = j \mid X_t = i)
\]

---

## Implementation Details

- Finite discrete state space (bus stops)
- Transition matrix defined explicitly
- Random sampling based on categorical distributions
- State trajectory simulation
- Directed weighted graph visualization

---

## Visualization

- Nodes represent states
- Directed edges represent transitions
- Edge weights represent transition probabilities

---

## Results

- Correct stochastic transitions
- Empirical trajectories consistent with defined probabilities
- Intuitive visualization of probabilistic dynamics

---

# Project 2: Multi-Armed Bandit Simulation

## Problem Definition

The **Multi-Armed Bandit** problem models a stateless RL scenario where an agent must balance **exploration vs exploitation**.

At each time step \( t \), the agent selects an arm \( a_t \) and receives reward \( r_t \sim R(a_t) \).

---

## Algorithms Implemented

### ε-Greedy Action Selection

\[
a_t =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon
\end{cases}
\]

### Incremental Value Update

\[
Q_{n+1}(a) = Q_n(a) + \alpha (R_n - Q_n(a))
\]

---

## Results

- Exploration accelerates convergence
- Higher ε increases early variance
- Average reward converges to optimal arm

---

# Project 3: Policy Evaluation and Improvement in MDPs

## Markov Decision Process (MDP)

An MDP is defined by the tuple:

\[
(S, A, P, R, \gamma)
\]

---

## Policy Evaluation

Iterative Bellman expectation update:

\[
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\]

---

## Policy Improvement

\[
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V(s') \right]
\]

---

## Results

- Guaranteed convergence
- Stable optimal policy obtained
- Exact match with theoretical DP guarantees

---

# Project 4: Policy Iteration Solution for the Tower of Hanoi

## Environment Design

- Custom Gym-compatible environment
- State space size: \( 3^n \)
- Deterministic transitions
- Invalid action penalties

---

## Algorithm: Policy Iteration

1. Initialize random policy
2. Policy evaluation until convergence
3. Policy improvement
4. Repeat until policy stable

---

## Reward Structure

| Event | Reward |
|-----|--------|
| Valid move | −1 |
| Invalid move | −10 |
| Goal reached | +100 |

---

## Results

- Optimal solution found
- Minimal number of moves
- Correct policy verified via visualization

---

# Project 5: Monte Carlo and Temporal-Difference Value Prediction

## Environment

- Random Walk (Sutton & Barto benchmark)
- Episodic
- Known true value function

---

## Monte Carlo Update

\[
V(s) \leftarrow V(s) + \alpha (G_t - V(s))
\]

---

## TD(0) Update

\[
V(s) \leftarrow V(s) + \alpha (r + \gamma V(s') - V(s))
\]

---

## Results

- TD converges faster
- MC shows higher variance
- Both converge to true values

---

# Project 6: SARSA and Q-Learning for Temporal-Difference Control

## SARSA (On-policy)

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma Q(s',a') - Q(s,a) \right)
\]

---

## Q-Learning (Off-policy)

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_a Q(s',a) - Q(s,a) \right)
\]

---

## Results

- Q-learning converges to optimal policy
- SARSA shows safer learning behavior
- Clear on-policy vs off-policy distinction

---

# Project 7: REINFORCE: Monte Carlo Policy Gradient Learning

## Policy Gradient Objective

\[
J(\theta) = \mathbb{E}_\pi \left[ \sum_t \gamma^t r_t \right]
\]

---

## Gradient Update Rule

\[
\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t | s_t)
\]

---

## Characteristics

- No value function
- High variance
- Unbiased gradient estimates
- Stochastic policies

---

## Results

- Policy improvement over episodes
- Reward convergence
- Correct gradient-based learning behavior

---

## Learning Progression Summary

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

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press  
   http://incompleteideas.net/book/the-book-2nd.html

2. MIT OpenCourseWare – Introduction to Probability  
   https://ocw.mit.edu

3. Stanford CS234 – Reinforcement Learning  
   https://web.stanford.edu/class/cs234/

4. OpenAI Spinning Up  
   https://spinningup.openai.com

---

## Final Note

This repository reflects a **complete and rigorous journey through Reinforcement Learning**, emphasizing correctness, clarity, and depth over shortcuts or black-box solutions.
