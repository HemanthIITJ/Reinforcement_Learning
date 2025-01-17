

#### Chapter 2: Markov Decision Processes (MDPs)

**2.1 Definition of MDP**

An MDP is a discrete-time stochastic control process that provides a mathematical framework for modeling decision-making in `situations where outcomes are partly random and partly under the control of a decision-maker`.



Key properties:
1. Markov Property: The future state depends only on the current state and action, not on the history of previous states.
2. Time-discreteness: Decisions are made at discrete time steps.
3. Stochasticity: Transitions between states are probabilistic.
4. Reward structure: Each state-action-next state triple is associated with a reward.

**2.2 Formal Representation of MDP**

Detailed explanation of MDP components:

1. S (State Space): 
   - Finite set of all possible states in the environment.
   - Example: In a grid world, S could be all possible (x,y) coordinates.

2. A (Action Space): 
   - Finite set of all possible actions an agent can take.
   - Example: In a robot navigation task, A could be {move_forward, turn_left, turn_right}.

3. P (Transition Probability Function):
   - P : S × A × S → [0, 1]
   - P(s' | s, a) represents the probability of transitioning to state s' when action a is taken in state s.
   - Must satisfy: ∑s'∈S P(s' | s, a) = 1 for all s ∈ S and a ∈ A.

4. R (Reward Function):
   - R : S × A × S → ℝ
   - R(s, a, s') represents the immediate reward received after transitioning from state s to s' due to action a.
   - Can also be represented as R(s, a) if the reward depends only on the current state and action.

5. γ (Discount Factor):
   - γ ∈ [0, 1]
   - Represents the importance of future rewards.
   - γ = 0: Myopic evaluation (only immediate rewards matter).
   - γ = 1: Far-sighted evaluation (future rewards are as important as immediate ones).

**2.3 Bellman Equation**

The Bellman equation is central to solving MDPs. It expresses the relationship between the value of a state and the values of its successor states.

1. State-Value Function (V^π(s)):
   Represents the expected return when starting in state s and following policy π thereafter.

   Bellman Expectation Equation for V^π(s):

   $$
   V^\pi(s) = \sum_{a \in A} \pi(a|s) \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V^\pi(s')\right) 
   $$ 

2. Action-Value Function (Q^π(s,a)):
   Represents the expected return of taking action a in state s and following policy π thereafter.

   Bellman Expectation Equation for Q^π(s,a):

   $$
   Q^\pi(s,a) = R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) \sum_{a' \in A} \pi(a'|s') Q^\pi(s',a')
   $$

3. Optimal Value Functions:
   V* and Q* represent the maximum possible expected return.

   Bellman Optimality Equation for V*(s):

   $$
   V^*(s) = \max_{a \in A} \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V^*(s')\right)
   $$

   Bellman Optimality Equation for Q*(s,a):

   $$
   Q^*(s,a) = R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) \max_{a' \in A} Q^*(s',a')
   $$

**2.4 Finite vs Infinite Horizon MDPs**

1. Finite Horizon MDPs:
   - Has a predetermined terminal time step T.
   - Value function and optimal policy depend on the time step: V_t(s), π_t(s).
   - Backward induction can be used to solve:
     For t = T, T-1, ..., 1:
     $$V_t(s) = \max_{a \in A} \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V_{t+1}(s')\right)$$

2. Infinite Horizon MDPs:
   - No predetermined end time.
   - Stationary optimal policy exists: π(s) is independent of time.
   - Value iteration or policy iteration can be used to solve.

**2.5 Solving an MDP: Policy Evaluation, Policy Iteration, Value Iteration**

1. Policy Evaluation:
   - Goal: Compute V^π for a given policy π.
   - Iterative algorithm:
     Initialize V(s) = 0 for all s ∈ S
     Repeat until convergence:
       For each s ∈ S:
         $$V(s) \leftarrow \sum_{a \in A} \pi(a|s) \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V(s')\right)$$

2. Policy Iteration:
   - Alternates between policy evaluation and policy improvement.
   - Algorithm:
     1. Initialize π arbitrarily
     2. Policy Evaluation: Compute V^π
     3. Policy Improvement:
        For each s ∈ S:
          $$\pi'(s) \leftarrow \arg\max_{a \in A} \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V^\pi(s')\right)$$
     4. If π' ≠ π, set π ← π' and go to step 2; otherwise, terminate.

3. Value Iteration:
   - Combines policy evaluation and improvement in a single step.
   - Algorithm:
     Initialize V(s) arbitrarily for all s ∈ S
     Repeat until convergence:
       For each s ∈ S:
         $$V(s) \leftarrow \max_{a \in A} \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V(s')\right)$$
     Extract policy:
       $$\pi(s) \leftarrow \arg\max_{a \in A} \left(R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a)V(s')\right)$$

These methods form the foundation for solving MDPs and are crucial in the development of reinforcement learning algorithms. Understanding these concepts is essential for researchers and scientists working in the fields of artificial intelligence, operations research, and decision theory.