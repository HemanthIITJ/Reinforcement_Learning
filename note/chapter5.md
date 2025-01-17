# Chapter 5: Temporal-Difference (TD) Learning

## Introduction

Temporal-Difference (TD) learning is a fundamental concept in reinforcement learning that combines elements of dynamic programming and Monte Carlo methods. This chapter delves into the intricacies of TD learning, exploring its various forms and applications in prediction and control tasks.

## Temporal-Difference Prediction (TD(0))

### Concept

TD(0) is the simplest form of TD learning, used for prediction tasks. It updates value estimates based on the difference between successive predictions.

### Algorithm

1. Initialize V(s) arbitrarily for all states s
2. For each episode:
   - Initialize S
   - For each step of episode:
     - A ← action given by policy π for S
     - Take action A, observe R, S'
     - V(S) ← V(S) + α[R + γV(S') - V(S)]
     - S ← S'
   - Until S is terminal

### Mathematical Formulation

The TD(0) update rule is given by:

$$
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
$$

Where:
- $V(S_t)$ is the value estimate of state $S_t$
- $α$ is the learning rate
- $R_{t+1}$ is the reward received
- $γ$ is the discount factor
- $V(S_{t+1})$ is the value estimate of the next state

### Advantages

- Learns from incomplete episodes
- Updates estimates based on other estimates (bootstrapping)
- Generally converges faster than Monte Carlo methods

## TD Control: SARSA

### Concept

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm that learns Q-values for state-action pairs.

### Algorithm

1. Initialize Q(s,a) arbitrarily for all s, a
2. For each episode:
   - Initialize S
   - Choose A from S using policy derived from Q (e.g., ε-greedy)
   - For each step of episode:
     - Take action A, observe R, S'
     - Choose A' from S' using policy derived from Q
     - Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
     - S ← S'; A ← A'
   - Until S is terminal

### Mathematical Formulation

The SARSA update rule is:

$$
Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γQ(S_{t+1},A_{t+1}) - Q(S_t,A_t)]
$$

### Key Features

- On-policy: learns action-values relative to the current policy
- Considers the action taken in the next state when updating Q-values

## Q-Learning

### Concept

Q-Learning is an off-policy TD control algorithm that learns the optimal action-value function directly.

### Algorithm

1. Initialize Q(s,a) arbitrarily for all s, a
2. For each episode:
   - Initialize S
   - For each step of episode:
     - Choose A from S using policy derived from Q (e.g., ε-greedy)
     - Take action A, observe R, S'
     - Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
     - S ← S'
   - Until S is terminal

### Mathematical Formulation

The Q-Learning update rule is:

$$
Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ \max_a Q(S_{t+1},a) - Q(S_t,A_t)]
$$

### Key Features

- Off-policy: learns about the optimal policy while following an exploratory policy
- Uses the maximum Q-value of the next state in updates

## On-Policy vs Off-Policy Learning

### On-Policy Learning (e.g., SARSA)

- Learns the value of the policy being followed
- More stable in some environments
- May be slower to converge to optimal policy

### Off-Policy Learning (e.g., Q-Learning)

- Learns about the optimal policy while following a different policy
- Can be more sample efficient
- May be less stable in some cases

## Convergence and Stability in TD Learning

### Convergence Properties

- TD(0) converges to the true value function under certain conditions
- Q-Learning converges to the optimal action-value function with probability 1

### Factors Affecting Convergence

1. Learning Rate (α):
   - Must satisfy the Robbins-Monro conditions:
     $$
     \sum_{t=1}^{\infty} α_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} α_t^2 < \infty
     $$

2. Exploration Strategy:
   - Sufficient exploration is necessary for convergence

3. Function Approximation:
   - Can affect convergence properties, especially in off-policy learning

### Stability Considerations

- On-policy methods like SARSA are generally more stable
- Off-policy methods like Q-Learning can be less stable, especially with function approximation

## Conclusion

Temporal-Difference learning represents a powerful class of algorithms in reinforcement learning, bridging the gap between Monte Carlo and dynamic programming methods. TD(0), SARSA, and Q-Learning offer different approaches to prediction and control tasks, each with its own strengths and considerations. Understanding the nuances of on-policy and off-policy learning, as well as the factors affecting convergence and stability, is crucial for effectively applying these methods in various reinforcement learning scenarios.