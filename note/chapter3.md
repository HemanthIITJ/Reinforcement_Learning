# Chapter 3: Dynamic Programming in Reinforcement Learning

## Overview of Dynamic Programming (DP)

Dynamic Programming (DP) is a powerful method for solving complex problems by breaking them down into simpler subproblems. In the context of Reinforcement Learning (RL), DP algorithms are used to compute optimal policies given a perfect model of the environment.

Key characteristics of DP in RL:
1. Requires complete knowledge of the environment (model-based)
2. Solves Markov Decision Processes (MDPs) with finite state and action spaces
3. Provides a foundation for understanding more advanced RL algorithms

## Policy Evaluation

Policy Evaluation is the process of computing the state-value function V^π(s) for a given policy π.

### Iterative Policy Evaluation

The iterative policy evaluation algorithm:

1. Initialize V(s) arbitrarily for all s ∈ S
2. Repeat until convergence:
   For each s ∈ S:
   
   $$
   V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s', r} p(s',r|s,a)[r + \gamma V(s')]
   $$
   
   Where:
   - π(a|s) is the probability of taking action a in state s under policy π
   - p(s',r|s,a) is the probability of transitioning to state s' and receiving reward r
   - γ is the discount factor

## Policy Improvement

Policy Improvement aims to find a better policy π' given the value function V^π of the current policy π.

### Policy Iteration Algorithm

1. Initialization: Choose an arbitrary policy π
2. Policy Evaluation: Compute V^π
3. Policy Improvement: For each s ∈ S:
   
   $$
   \pi'(s) = \arg\max_{a} \sum_{s', r} p(s',r|s,a)[r + \gamma V^{\pi}(s')]
   $$
   
4. If π' ≠ π, set π ← π' and go to step 2; otherwise, terminate

## Value Iteration Algorithm

Value Iteration combines policy evaluation and improvement into a single update:

1. Initialize V(s) arbitrarily for all s ∈ S
2. Repeat until convergence:
   For each s ∈ S:
   
   $$
   V(s) \leftarrow \max_{a} \sum_{s', r} p(s',r|s,a)[r + \gamma V(s')]
   $$
   

## Limitations of Dynamic Programming in Large State Spaces

1. Curse of dimensionality: Computational complexity grows exponentially with state space size
2. Memory requirements: Storing value functions for large state spaces becomes infeasible
3. Model dependency: Requires complete knowledge of transition probabilities and rewards