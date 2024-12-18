# Lecture 1

*   **Return** is the sum of future rewards:

    *   $G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$ 
*   **Value** is the expected cumulative reward from a state *s*:

    *   $v(s) = E[G_t | S_t = s]$ 
    *   $ = E[R_{t+1} + R_{t+2} + R_{t+3} + ... | S_t = s]$ 
*   **Returns and values** can be defined **recursively**:

    *   $G_t = R_{t+1} + G_{t+1}$ 
    *   $v(s) = E[R_{t+1} + v(S_{t+1}) | S_t = s]$ 
*   A **mapping from states to actions** is called a **policy**.
*   It is also possible to **condition the value on actions:**

    *   $q(s, a) = E[G_t | S_t = s, A_t = a]$
    *   $ = E[R_{t+1} + R_{t+2} + R_{t+3} + ... | S_t = s, A_t = a]$
*   The **actual value function** is the **expected return**:

    *   $v^\pi(s) = E[G_t | S_t = s, \pi]$ 
    *   $= E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s, \pi]$
*   The **value function** also has a **recursive form** because the return has a recursive form:

    *   $v^\pi(s) = E[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t \sim \pi (s)]$ 
    *   $= E[R_{t+1} + \gamma v^\pi(S_{t+1}) | S_t = s, A_t \sim \pi (s)]$ 
    *   where $a \sim \pi(s)$ means *a* is chosen by policy $\pi$ in state *s*
*   A **similar equation** holds for the **optimal value**: 

    *   $v^*(s) = \max\limits_{a} E[R_{t+1} + \gamma v^*(S_{t+1}) | S_t = s, A_t = a]$ 
        *   This does not depend on a policy.

# Lecture 2

The following are mathematical equations from the sources, written step-by-step:

### **Values and Regret**

*   The **action-value** for an action *a* is the expected reward:
    $q(a) = E[R_t | A_t = a]$
*   The **optimal value** is:
    $v^* = max_{a ∈ A} q(a) = max_a E[R_t | A_t = a]$
*   **Regret** of an action *a* is:
    $Δ_a = v^* - q(a)$
*   The **total regret** is: 
    $L_t = \sum_{n=1}^{t} v^* - q(A_n) = \sum_{n=1}^{t} Δ_{A_n}$

### **Action Values**

*   A simple estimate of the action value is the average of the sampled rewards:
    $Q_t(a) = \frac{\sum_{n=1}^{t} I(A_n = a) R_n}{\sum_{n=1}^{t} I(A_n = a)}$
    where:
        *   *I(·)* is the indicator function:  *I(True) = 1* and *I(False) = 0*
        *   The count for action *a* is:
            $N_t(a) = \sum_{n=1}^{t} I(A_n = a)$
*   This can also be updated **incrementally**:
    $Q_t(A_t) = Q_{t-1}(A_t) + α_t (R_t - Q_{t-1}(A_t))$
    $∀_a, A_t: Q_t(a) = Q_{t-1}(a)$
    with:
        *   $α_t = \frac{1}{N_t(A_t)}$ and $N_t(A_t) = N_{t-1}(A_t) + 1$
        *   where $N_0(a) = 0$.

### **Policy Search**

*   Define **action preferences** $H_t(a)$ and a **policy**:
    $\pi(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}$ 
*   Update **policy parameters** such that the **expected value increases**:
    $θ_{t+1} = θ_t + α∇_θE[R_t | \pi_{θ_t}]$
    where $θ_t$ are the current policy parameters

### **Gradient Bandits**

*   **Log-likelihood trick**:
    $∇_θE[R_t|\pi_θ] = ∇_θ\sum_a \pi_θ(a)q(a)$
    $= \sum_a q(a)∇_θ\pi_θ(a)$
    $= \sum_a q(a)\pi_θ(a) \frac{∇_θ\pi_θ(a)}{\pi_θ(a)}$
    $= \sum_a \pi_θ(a)q(a) \frac{∇_θ\pi_θ(a)}{\pi_θ(a)}$
    $= E[R_t \frac{∇_θ\pi_θ(A_t)}{\pi_θ(A_t)}] = E[R_t ∇_θlog\pi_θ(A_t)]$
*   **Stochastic gradient ascent**:
    $θ = θ + αR_t ∇_θlog\pi_θ(A_t)$
*   For **softmax**:
    $H_{t+1}(a) = H_t(a) + αR_t \frac{∂log\pi_t(A_t)}{∂H_t(a)}$
    $= H_t(a) + αR_t(I(a = A_t) - \pi_t(a))$
    $⇒ H_{t+1}(A_t) = H_t(A_t) + αR_t(1 - \pi_t(A_t))$
    $H_{t+1}(a) = H_t(a) - αR_t\pi_t(a)$ if $a ≠ A_t$ 

### **How Well Can We Do?**

*   **Theorem (Lai and Robbins)**:
    $lim_{t→∞} L_t ≥ log t \sum_{a | Δ_a > 0} \frac{Δ_a}{KL(R_a || R_{a^*})}$

### **Counting Regret**

*   **Total regret**:
    $L_t = \sum_{n=1}^{t} Δ_{A_n} = \sum_{a ∈ A}N_t(a) Δ_a$

### **Upper Confidence Bounds**

*   **Hoeffding's Inequality**:
    $P( \bar{X}_n + u ≤ \mu) ≤ e^{-2nu^2}$
*   Applying **Hoeffding's Inequality** to bandits with bounded rewards:
    $P(Q_t(a) + U_t(a) ≤ q(a)) ≤ e^{-2N_t(a)U_t(a)^2}$
*   **Upper confidence bound**:
    $U_t(a) = \sqrt{\frac{-log p}{2N_t(a)}}$
*   Reduce *p* as we observe more rewards, e.g., $p = 1/t$:
    $U_t(a) = \sqrt{\frac{log t}{2N_t(a)}}$
*   **UCB algorithm**:
    $a_t = argmax_{a∈A} Q_t(a) + c\sqrt{\frac{log t}{N_t(a)}}$
*   **Theorem (Auer et al., 2002)**:
    $L_t ≤ 8\sum_{a | Δ_a > 0}\frac{log t}{Δ_a} + O(\sum_a Δ_a), ∀_t$.

### **Bayesian Bandits: Example**

*   **Posterior is a Beta distribution**:
    $Beta(x_a, y_a)$
    with initial parameters $x_a = 1$ and $y_a = 1$ for each action *a*
*   **Updating the posterior**:
    $x_{A_t} ← x_{A_t} + 1$ when $R_t = 0$
    $y_{A_t} ← y_{A_t} + 1$ when $R_t = 1$

### **Probability Matching**

*   Select action *a* according to the probability that *a* is optimal:
    $\pi_t(a) = P(q(a) = max_{a'} q(a') | H_{t-1})$

### **Thompson Sampling**

*   Sample $Q_t(a) \sim p_t(q(a)), ∀_a$
*   Select action maximizing sample:
    $A_t = argmax_a Q_t(a)$

### **Information State Space**

*   **Information state**:
    $I = (α, β)$

### **Example: Bernoulli Bandits**

*   **Bernoulli bandit**:
    $P(R_t = 1 | A_t = a) = \mu_a$
    $P(R_t = 0 | A_t = a) = 1 - \mu_a$ 

# Lecture 3

### **Markov Decision Process**

*   A **Markov Decision Process (MDP)** is defined as a tuple $(S, A, p, \gamma)$, where:
    *   $S$ is the set of all possible states.
    *   $A$ is the set of all possible actions.
    *   $p(r, s' | s, a)$ is the joint probability of a reward *r* and next state *s'*, given a state *s* and action *a*.
    *   $\gamma \in$ is a discount factor that trades off later rewards to earlier ones.

*   It can also be defined as a tuple $(S, A, p, r, \gamma)$, where:
    *   $S$ is the set of all possible states.
    *   $A$ is the set of all possible actions.
    *   $p(s' | s, a)$ is the probability of transitioning to *s'*, given a state *s* and action *a*.
    *   $r: S \times A \rightarrow R$ is the expected reward, achieved on a transition starting in (*s*, *a*):  $r = E[R | s, a]$.
    *   $\gamma \in$ is a discount factor that trades off later rewards to earlier ones.

*   The **dynamics** of the problem are defined by *p*.
*   Marginalized state transitions:
    $p(s' | s, a) = \sum_r p(s', r | s, a)$
*   Marginalized expected reward:
    $E[R | s, a] = \sum_r r\sum_{s'}p(r, s' | s, a)$.

### **Markov Property**

*   A state *s* has the **Markov property** when for states $\forall s' \in S$:
    $p(S_{t+1}=s' | S_t = s) = p(S_{t+1}=s' | h_{t-1}, S_t=s)$ for all possible histories $h_{t-1} = \{S_1,...,S_{t-1},A_1,...,A_{t-1},R_1,...,R_{t-1}\}$.

### **Returns**

*   **Undiscounted return** (episodic/finite horizon problem):
    $G_t = R_{t+1} + R_{t+2} + ... + R_T = \sum_{k=0}^{T-t-1} R_{t+k+1}$
*   **Discounted return** (finite or infinite horizon problem):
    $G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-t}R_T = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$
*   **Average return** (continuing, infinite horizon problem):
    $G_t = \frac{1}{T-t-1}(R_{t+1} + R_{t+2} + ... + R_T) = \frac{1}{T-t-1}\sum_{k=0}^{T-t-1} R_{t+k+1}$
*   **Discounted returns** $G_t$ for infinite horizon $T \rightarrow \infty$:
    $G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

### **Policies**

*   A **policy** $\pi : S \times A \rightarrow$ maps every state *s* to the probability of taking action *a* $\in A$ in state *s*, denoted by $\pi(a | s)$.
*   For **deterministic policies**, $a_t = \pi(s_t)$ denotes the action taken by the policy.

### **Value Functions**

*   The **value function** $v(s)$ gives the long-term value of state *s*:
    $v^\pi(s) = E[G_t | S_t = s, \pi]$
*   **State-action values**:
    $q^\pi(s, a) = E[G_t | S_t = s, A_t = a, \pi]$
*   **Connection** between the value function and state-action values:
    $v^\pi(s) = \sum_a \pi(a | s) q^\pi(s, a) = E[q^\pi(S_t, A_t) | S_t = s, \pi], \forall s$

### **Optimal Value Function**

*   The **optimal state-value function** $v^*(s)$ is the maximum value function over all policies:
    $v^*(s) = \max_\pi v^\pi(s)$
*   The **optimal action-value function** $q^*(s, a)$ is the maximum action-value function over all policies:
    $q^*(s, a) = \max_\pi q^\pi(s, a)$

### **Optimal Policy**

*   **Partial ordering over policies**:
    $\pi \geq \pi' \iff v^\pi(s) \geq v^{\pi'}(s), \forall s$
*   An **optimal policy** can be found by maximizing over $q^*(s, a)$:
    $\pi^*(s, a) = \begin{cases}
      1 & \text{if $a = argmax_{a \in A} q^*(s, a)$}\\
      0 & \text{otherwise}
    \end{cases}$

### **Bellman Equations**

*   The value function can be defined **recursively**:
    $v^\pi(s) = E[R_{t+1} + \gamma G_{t+1} | S_t = s, \pi]$
    $ = E[R_{t+1} + \gamma v^\pi(S_{t+1}) | S_t = s, A_t \sim \pi(S_t)]$
    $= \sum_a \pi(a | s) \sum_r \sum_{s'} p(r, s' | s, a)(r + \gamma v^\pi(s'))$

*   **State-action values**:
    $q^\pi(s, a) = E[R_{t+1} + \gamma v^\pi(S_{t+1}) | S_t = s, A_t = a]$
    $= E[R_{t+1} + \gamma q^\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$
    $= \sum_r \sum_{s'} p(r, s' | s, a)(r + \gamma \sum_{a'} \pi(a' | s')q^\pi(s', a'))$

*   **Bellman Expectation Equation**:
    $v^\pi(s) = \sum_a \pi(s, a) [r(s, a) + \gamma \sum_{s'} p(s' | a, s) v^\pi(s')]$
    $q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} p(s' | a, s) \sum_{a' \in A} \pi(a' | s')q^\pi(s', a')$

*   **Bellman Optimality Equations**:
    $v^*(s) = \max_a [r(s, a) + \gamma \sum_{s'} p(s' | a, s) v^*(s')]$
    $q^*(s, a) = r(s, a) + \gamma \sum_{s'} p(s' | a, s) \max_{a' \in A} q^*(s', a')$

### **Bellman Equation in Matrix Form**

*   The Bellman value equation for a given $\pi$ can be expressed using matrices:
    $v = r^\pi + \gamma P^\pi v$
    where:
        *   $v_i = v(s_i)$
        *   $r^\pi_i = E[R_{t+1} | S_t = s_i, A_t \sim \pi(S_t)]$
        *   $P^\pi_{ij} = p(s_j | s_i) = \sum_a \pi(a | s_i) p(s_j | s_i, a)$

*   The Bellman equation can be solved directly:
    $v = r^\pi + \gamma P^\pi v$
    $(I - \gamma P^\pi)v = r^\pi$
    $v = (I - \gamma P^\pi)^{-1}r^\pi$

### **Policy Evaluation**

*   Iterative update:
    $\forall s: v_{k+1}(s) \leftarrow E[R_{t+1} + \gamma v_k(S_{t+1}) | s, \pi]$

### **Policy Improvement**

*   Iterative update:
    $\forall s: \pi_{new}(s) = argmax_a q^\pi(s, a)$
    $= argmax_a E[R_{t+1} + \gamma v^\pi(S_{t+1}) | S_t = s, A_t = a]$

### **Value Iteration**

*   Iterative update:
    $\forall s: v_{k+1}(s) \leftarrow \max_a E[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s, A_t = s]$
***

# Lecture 4

### **Preliminaries**

*   A mapping $T: X\rightarrow X$ is an **$\alpha$-contraction mapping** if for any $x_1, x_2 \in X$, $\exists \alpha \in [0, 1)$ s.t.:
    $$\|Tx_1 - Tx_2\| \leq \alpha \|x_1 - x_2\|$$
*   If $\alpha \in$, then  *T*  is called **non-expanding**.

### **Bellman Expectation Equations**

Given an MDP,  $M = \langle S, A, p, r, \gamma \rangle$, for any policy $\pi$, the value functions obey the following expectation equations:

*   **State-value function**:
    $$v^\pi(s) = \sum_a \pi(s, a)[r(s, a) + \gamma \sum_{s'}p(s'|a,s)v^\pi(s')]$$
*   **Action-value function**:
    $$q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|a, s)\sum_{a' \in A}\pi(a'|s')q^\pi(s', a')$$

### **Bellman Optimality Equations**

Given an MDP, $M = \langle S, A, p, r, \gamma \rangle$, the optimal value functions obey the following expectation equations:

*   **Optimal state-value function**:
    $$v^*(s) = \max_a [r(s, a) + \gamma \sum_{s'} p(s'|a, s)v^*(s')]$$
*   **Optimal action-value function**:
    $$q^*(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|a, s) \max_{a' \in A} q^*(s', a')$$

### **Bellman Optimality Operator**

Given an MDP, $M = \langle S, A, p, r, \gamma \rangle$, let $V \equiv V_S$ be the space of bounded real-valued functions over *S*. The **Bellman Optimality Operator** $T^*_V: V \rightarrow V$ is defined point-wise as:

$$(T^*_V f)(s) = \max_a [r(s, a) + \gamma \sum_{s'} p(s'|a, s)f(s')], \forall f \in V$$

As a common convention, the index *V* is dropped, so $T^* = T^*_V$.

### **Properties of the Bellman Operator** $T^*$

*   It has one unique fixed point $v^*$:
    $$T^* v^* = v^*$$

*   $T^*$ is a $\gamma$-contraction w.r.t.  $\|\cdot\|_\infty$:
    $$\|T^*v - T^*u\|_\infty \leq \gamma \|v - u\|_\infty, \forall u, v \in V$$

*   $T^*$ is monotonic:
    $$\forall u, v \in V \text{ s.t. } u \leq v \text{, component-wise, then } T^* u \leq T^* v$$

### **Proof that** $T^*$ **is a** $\gamma$**-contraction w.r.t.** $\|\cdot\|_\infty$

$$|T^*v(s)-T^*u(s)| = |\max_a[r(s, a) + \gamma E_{s'|s,a}v(s')] - \max_b[r(s, b) + \gamma E_{s''|s, b}u(s'')]|$$

$$\leq \max_a |[r(s, a) + \gamma E_{s'|s, a}v(s')] - [r(s, a) + \gamma E_{s'|s, a}u(s')]|$$

$$= \gamma \max_a |E_{s'|s, a}[v(s') - u(s')]|$$

$$\leq \gamma \max_{s'}|[v(s') - u(s')]|$$

Thus, we get:

$$\|T^* v - T^* u\|_\infty \leq \gamma \|v - u\|_\infty, \forall u, v \in V$$

**Note**: Step (6)-(7) uses: $|\max_a f(a) - \max_b g(b)| \leq \max_a|f(a) - g(a)|$

### **Proof that** $T^*$ **is monotonic**

Given $v(s) \leq u(s), \forall s \implies r(s, a) + E_{s'|s, a} u(s') \leq r(s, a) + E_{s'|s,a}v(s')$

$$T^*v(s) - T^*u(s) = \max_a [r(s, a) + \gamma E_{s'|s, a} v(s')] - \max_b [r(s, b) + \gamma E_{s''|s, b} u(s'')]$$

$$\leq \max_a ([r(s, a) + \gamma E_{s'|s, a} v(s')] - [r(s, a) + \gamma E_{s'|s, a} u(s')])$$

$$\leq 0, \forall s$$

Thus, 

$$T^*v(s) \leq T^* u(s), \forall s \in S$$

### **Value Iteration through the Lens of the Bellman Operator**

**Value Iteration**

*   Start with $v_0$.
*   Update values: $v_{k+1} = T^* v_k$.

As $k \rightarrow \infty, v_k \rightarrow^{\|\cdot\|_\infty} v^*$.

**Proof:** Direct application of the Banach Fixed Point Theorem.

$$\|v_k - v^*\|_\infty = \|T^* v_{k-1} - v^*\|_\infty = \|T^* v_{k-1} - T^* v^*\|_\infty \text{ (fixed point property)}$$

$$\leq \gamma \|v_{k-1} - v^*\|_\infty \text{ (contraction property)}$$

$$\leq \gamma^k \|v_0 - v^*\|_\infty \text{ (iterative application)}$$

### **Bellman Expectation Operator**

Given an MDP, $M = \langle S, A, p, r, \gamma \rangle$, let $V \equiv V_S$ be the space of bounded real-valued functions over *S*. For any policy $\pi: S \times A \rightarrow$, the **Bellman Expectation Operator** $T^\pi_V: V \rightarrow V$ is defined point-wise as:

$$(T^\pi_V f)(s) = \sum_a \pi(s, a)[r(s, a) + \gamma \sum_{s'} p(s'|a, s)f(s')], \forall f \in V$$

### **Properties of the Bellman Operator** $T^\pi$

*   It has one unique fixed point $v^\pi$:
    $$T^\pi v^\pi = v^\pi$$
*   $T^\pi$ is a $\gamma$-contraction w.r.t. $\|\cdot\|_\infty$:
    $$\|T^\pi v - T^\pi u\|_\infty \leq \gamma \|v - u\|_\infty, \forall u, v \in V$$
*   $T^\pi$ is monotonic:
    $$\forall u, v \in V \text{ s.t. } u \leq v \text{, component-wise, then } T^\pi u \leq T^\pi v$$

### **Proof that** $T^\pi$ **is a** $\gamma$**-contraction w.r.t.** $\|\cdot\|_\infty$

$$T^\pi v(s) - T^\pi u(s) = \sum_a \pi(a|s)[r(s, a) + \gamma E_{s'|s, a} v(s') - r(s, a) - \gamma E_{s'|s, a}u(s')]$$

$$= \gamma \sum_a \pi(a|s) E_{s'|s, a}[v(s') - u(s')]$$

$$\implies |T^\pi v(s) - T^\pi u(s)| \leq \gamma \max_{s'} |[v(s') - u(s')]|$$

Thus, we get:

$$\|T^\pi v - T^\pi u\|_\infty \leq \gamma \|v - u\|_\infty, \forall u, v \in V$$

**Note**: Equation (14) also gives the monotonicity of $T^\pi$.

### **Policy Evaluation**

**(Iterative) Policy Evaluation**

*   Start with $v_0$.
*   Update values: $v_{k+1} = T^\pi v_k$.

As $k \rightarrow \infty, v_k \rightarrow^{\|\cdot\|_\infty} v^\pi$.

**Proof**: Direct application of the Banach Fixed Point Theorem.

### **Dynamic Programming with Bellman Operators**

**Value Iteration**

*   Start with $v_0$.
*   Update values: $v_{k+1} = T^* v_k$.

**Policy Iteration**

*   Start with $\pi_0$.

*   Iterate:

    *   Policy Evaluation: $v^{\pi_i}$ (e.g., for instance, by iterating $T^\pi: v_k = T^{\pi_i} v_{k-1} \implies v_k \rightarrow v^{\pi_i} \text{ as } k \rightarrow \infty$)
    *   Greedy Improvement: $\pi_{i+1} = \argmax_a q^{\pi_i}(s, a)$

Similarly for $q^\pi: S \times A \rightarrow R$ functions:

Given an MDP, $M = \langle S, A, p, r, \gamma \rangle$, let $Q \equiv Q_{S,A}$ be the space of bounded real-valued functions over $S \times A$. For any policy $\pi: S \times A \rightarrow$, the **Bellman Expectation Operator** $T^\pi_Q: Q \rightarrow Q$ is defined point-wise as:

$$(T^\pi_Q f)(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|a, s) \sum_{a' \in A} \pi(a'|s')f(s', a'), \forall f \in Q$$

This operator has a unique fixed point which corresponds to the action-value function $q^\pi$ in our MDP *M*. It has the same properties as $T^\pi$:  $\gamma$-contraction and monotonicity.

Similarly for $q^*: S \times A \rightarrow R$ functions:

Given an MDP, $M = \langle S, A, p, r, \gamma \rangle$, let $Q \equiv Q_{S, A}$ be the space of bounded real-valued functions over $S \times A$. The **Bellman Optimality Operator** $T^*_Q: Q \rightarrow Q$ is defined as:

$$(T^*_Q f)(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|a, s) \max_{a' \in A} f(s', a'), \forall f \in Q$$

This operator has a unique fixed point which corresponds to the action-value function $q^*$ in our MDP *M*. It has the same properties as $T^*$:  $\gamma$-contraction and monotonicity.

### **Approximate Value Iteration**

**Approximate Value Iteration**

*   Start with $v_0$.
*   Update values: $v_{k+1} = A T^* v_k (v_{k+1} \approx T^* v_k)$.

**Question**: As $k \rightarrow \infty$, does $v_k \rightarrow^{\|\cdot\|_\infty} v^*$?

**Answer**: In general, no.

### **ADP: Approximating the Value Function**

Using a function approximator $v_\theta(s)$ with a parameter vector $\theta \in R^m$:

*   The estimated value function at iteration *k* is $v_k = v_{\theta_k}$

*   Use dynamic programming to compute $v_{\theta_{k+1}}$ from $v_{\theta_k}$

    $$T^* v_k(s) = \max_a E[R_{t+1} + \gamma v_k(S_{t+1})|S_t = s]$$

*   Fit $\theta_{k+1}$ s.t. $v_{\theta_{k+1}} \approx T^*v_k(s)$

*   For instance, with respect to a squared loss over the state-space:

    $$\theta_{k+1} = \argmin_{\theta_{k+1}} \sum_s (v_{\theta_{k+1}}(s) - T^*v_k(s))^2$$

### **Example of Divergence with Dynamic Programming**

$$\theta_{k+1} = \argmin_\theta \sum_{s \in S}(v_\theta(s) - E[v_{\theta_k}(S_{t+1})|S_t=s])^2$$

$$= \argmin_\theta (\theta - \gamma 2 \theta_k)^2 + (2\theta - \gamma(1 - \epsilon)2\theta_k)^2$$

$$= \frac{2(3-2\epsilon)\gamma}{5}\theta_k$$

### **Performance of a Greedy Policy**

Consider an MDP. Let $q: S \times A \rightarrow R$ be an arbitrary function and let $\pi$ be the greedy policy associated with *q*, then:

$$\|q^* - q^\pi\|_\infty \leq \frac{2 \gamma}{1 - \gamma}\|q^* - q\|_\infty$$

where $q^*$ is the optimal value function associated with this MDP.

### **Proof of Performance of a Greedy Policy**

**Statement**:

$$\|q^* - q^\pi\|_\infty \leq \frac{2\gamma}{1 - \gamma}\|q^* - q\|_\infty$$

**Proof**:

$$\|q^* - q^\pi\|_\infty = \|q^* - T^\pi q + T^\pi q - q^\pi\|_\infty$$

$$\leq \|q^* - T^\pi q\|_\infty + \|T^\pi q - q^\pi\|_\infty$$

$$= \|T^* q^* - T^*q\|_\infty + \|T^\pi q - T^\pi q^\pi\|_\infty$$

$$\leq \gamma \|q^* - q\|_\infty + \gamma \underbrace{\|q - q^\pi\|_\infty}_{\leq \|q - q^*\|_\infty + \|q^* - q^\pi\|_\infty}$$

$$\leq 2\gamma \|q^* - q\|_\infty + \gamma \|q^* - q^\pi\|_\infty$$

Re-arranging: $(1 - \gamma)\|q^* - q^\pi\|_\infty \leq 2\gamma\|q^* - q\|_\infty$.

### **Approximate Policy Iteration**

**Approximate Policy Iteration**

*   Start with $\pi_0$.
*   Iterate:
    *   Policy Evaluation: $q_i = Aq^{\pi_i} (q_i \approx q^{\pi_i})$
    *   Greedy Improvement: $\pi_{i+1} = \argmax_a q_i(s, a)$

**Question 1**: As $i \rightarrow \infty$, does $q_i \rightarrow^{\|\cdot\|_\infty} q^*$?

**Answer**: In general, no.

**Question 2**: Or, does $\pi_i$ converge to the optimal policy?

**Answer**: In general, no.

### **Approximate Dynamic Programming**

**Approximate Value Iteration**

*   Start with $v_0$.
*   Update values: $v_{k+1} = AT^*v_k (v_{k+1} \approx T^* v_k)$.

**Approximate Policy Iteration**

*   Start with $\pi_0$.
*   Iterate:
    *   Policy Evaluation: $q_i = Aq^{\pi_i} (q_i \approx q^{\pi_i})$
    *   Greedy Improvement: $\pi_{i+1} = \argmax_a q_i(s, a)$
***
# Lecture 5


### **Monte Carlo: Bandits**

For each action, the average reward samples is:

  

$q_t(a) = \frac{\sum_{i=0}^t I(A_i = a)R_{i+1}}{\sum_{i=0}^t I(A_i=a)} \approx E[R_{t+1}|A_t = a] = q(a)$

  

Equivalently:

  

$q_{t+1}(A_t) = q_t(A_t) + \alpha_t(R_{t+1}-q_t(A_t))$

$q_{t+1}(a) = q_t(a)  \forall a \neq A_t$

  

with $\alpha_t = \frac{1}{N_t(A_t)} = \frac{1}{\sum_{i=0}^t I(A_i = a)}$

  

Note: we changed notation $R_t \rightarrow R_{t+1}$ for the reward after $A_t$.

In MDPs, the reward is said to arrive on the time step after the action.

  

### **Monte Carlo: Bandits with States**

We want to estimate:

  

$q(s, a) = E[R_{t+1}|S_t = s, A_t = a]$

  

*q* could be a parametric function, e.g., neural network, and we could use loss:

  

$L(w) = \frac{1}{2}E[(R_{t+1} - q_w(S_t, A_t))^2]$

  

Then the gradient update is:

  

$w_{t+1} = w_t - \alpha \nabla_{w_t}L(w_t)$

$= w_t - \alpha \nabla_{w_t} \frac{1}{2}E[(R_{t+1}-q_{w_t}(S_t, A_t))^2]$

$= w_t + \alpha E[(R_{t+1}-q_{w_t}(S_t, A_t))\nabla_{w_t} q_{w_t}(S_t, A_t)]$

  

We can sample this to get a stochastic gradient update (SGD).

  

When using linear functions, $q(s, a) = w^Tx(s, a)$ and $\nabla_{w_t}q_{w_t}(S_t, A_t) = x(s, a)$.

  

Then the SGD update is:

  

$w_{t+1} = w_t + \alpha(R_{t+1}-q_{w_t}(S_t, A_t))x(s, a)$.

  

### **Monte-Carlo Policy Evaluation**

The return is the total discounted reward (for an episode ending at time *T* &gt; *t*):

  

$G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-t-1}R_T$

  

The value function is the expected return:

  

$v^\pi(s) = E[G_t|S_t = s, \pi]$

  

### **Temporal-Difference Learning**

Bellman equations:

  

$v^\pi(s) = E[R_{t+1} + \gamma v^\pi(S_{t+1})|S_t = s, A_t \sim \pi(S_t)]$

  

Approximate by iterating:

  

$v_{k+1}(s) = E[R_{t+1} + \gamma v_k(S_{t+1})|S_t = s, A_t \sim \pi(S_t)]$

  

Sampling this:

  

$v_{t+1}(S_t) = R_{t+1} + \gamma v_t(S_{t+1})$

  

Taking a small step (with parameter $\alpha$):

  

$v_{t+1}(S_t) = v_t(S_t) + \alpha_t (\underbrace{R_{t+1}+\gamma v_t(S_{t+1})}_{target} - v_t(S_t))$

(Note: tabular update)

  

**Temporal difference learning**

  

*   **Monte-Carlo**:
    $v_{n+1}(S_t) = v_n(S_t) + \alpha(G_t - v_n(S_t))$

*   **Temporal-difference learning**:
    $v_{t+1}(S_t) \leftarrow v_t(S_t) + \alpha(\underbrace{R_{t+1} + \gamma v_t(S_{t+1})}_{target} - v_t(S_t))$

$\delta_t = R_{t+1} + \gamma v_t(S_{t+1}) - v_t(S_t)$ is called the TD error

  

**Temporal difference learning for action values**:

*   Update value $q_t(S_t, A_t)$ towards estimated return $R_{t+1} + \gamma q(S_{t+1}, A_{t+1})$:

$q_{t+1}(S_t, A_t) \leftarrow q_t(S_t, A_t) + \alpha(\underbrace{R_{t+1} + \gamma q_t(S_{t+1}, A_{t+1})}_{target} - q_t(S_t, A_t))$

  

### **Batch MC and TD**

Consider a fixed batch of experience:

  

episode 1: $S_1^1, A_1^1, R_2^1, ..., S_{T_1}^1$
...
episode *K*: $S_1^K, A_1^K, R_2^K, ..., S_{T_K}^K$

  

### **Differences in batch solutions**

  

*   **MC** converges to best mean-squared fit for the observed returns:

$\sum_{k=1}^K \sum_{t=1}^{T_k}(G_t^k - v(S_t^k))^2$

  

*   **TD** converges to solution of max likelihood Markov model, given the data.

  

### **Multi-Step Updates**

  

TD target look *n* steps into the future

  

### **Multi-Step Prediction**

  

Consider the following n-step returns for *n* = 1, 2, ∞:

  

*   *n* = 1 (TD) $G_t^{(1)} = R_{t+1} + \gamma v(S_{t+1})$
*   *n* = 2 $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 v(S_{t+2})$
*   ...
*   *n* = ∞ (MC) $G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-t-1} R_T$

  

In general, the n-step return is defined by:

  

$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^n v(S_{t+n})$

  

**Multi-step temporal-difference learning**:

  

$v(S_t) \leftarrow v(S_t) + \alpha(G_t^{(n)} - v(S_t))$

  

### **Mixing multi-step returns**

  

Multi-step returns bootstrap on one state, $v(S_{t+n})$:

  

$G_t^{(n)} = R_{t+1} + \gamma G_{t+1}^{(n-1)}$ (while *n* &gt; 1, continue)

$G_t^{(1)} = R_{t+1} + \gamma v(S_{t+1})$. (truncate &amp; bootstrap)

  

You can also bootstrap a little bit on multiple states:

  

$G_t^\lambda = R_{t+1} + \gamma ((1-\lambda)v(S_{t+1}) + \lambda G_{t+1}^\lambda)$

  

This gives a weighted average of n-step returns:

  

$G_t^\lambda = \sum_{n=1}^\infty (1-\lambda) \lambda^{n-1} G_t^{(n)}$

(Note, $\sum_{n=1}^\infty (1-\lambda)\lambda^{n-1} = 1$)

  

**Special cases**:

  

$G_{t}^{\lambda=0} = R_{t+1} + \gamma v(S_{t+1})$ (TD)

$G_{t}^{\lambda=1} = R_{t+1} + \gamma G_{t+1}$ (MC)

  

### **Eligibility Traces**

  

Recall linear function approximation.

  

The Monte Carlo and TD updates to $v_w(s) = w^Tx(s)$ for a state $s = S_t$ is:

  

$\Delta w_t = \alpha(G_t - v(S_t))x_t$ (MC)

$\Delta w_t = \alpha(R_{t+1} + \gamma v(S_{t+1}) - v(S_t))x_t$ (TD)

  

MC updates all states in episode *k* at once:

  

$\Delta w_{k+1} = \sum_{t=0}^{T-1} \alpha(G_t - v(S_t))x_t$

  

where $t \in \{0, ..., T-1\}$ enumerate the time steps in this specific episode

  

Accumulating a whole episode of updates:

  

$\Delta w_t \equiv \alpha \delta_t e_t$ (one time step)

where $e_t = \gamma \lambda e_{t-1} + x_t$

  

We can rewrite the MC error as a sum of TD errors:

  

$G_t - v(S_t) = R_{t+1} + \gamma G_{t+1} - v(S_t)$

$= \underbrace{R_{t+1} + \gamma v(S_{t+1}) - v(S_t)}_{= \delta_t} + \gamma(G_{t+1} - v(S_{t+1}))$

$= \delta_t + \gamma(G_{t+1} - v(S_{t+1}))$

$= ...$

$= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - v(S_{t+2}))$

$= ...$

$= \sum_{k=t}^T \gamma^{k-t}\delta_k$ (used in the next slide)

  

Now consider accumulating a whole episode (from time *t* = 0 to *T*) of updates:

  

$\Delta w_k = \sum_{t=0}^{T-1} \alpha(G_t - v(S_t))x_t$

$= \sum_{t=0}^{T-1} \alpha(\sum_{k=t}^{T-1} \gamma^{k-t}\delta_k)x_t$ (Using result from previous slide)

$= \sum_{k=0}^{T-1} \alpha (\sum_{t=0}^k \gamma^{k-t} \delta_k x_t)$ (Using $\sum_{i=0}^m \sum_{j=i}^m z_{ij} = \sum_{j=0}^m \sum_{i=0}^j z_{ij}$)

$= \sum_{k=0}^{T-1} \alpha \delta_k \underbrace{\sum_{t=0}^k \gamma^{k-t}x_t}_{\equiv e_k}$

$= \sum_{k=0}^{T-1} \alpha \delta_k e_k = \underbrace{\sum_{t=0}^{T-1} \alpha \delta_t e_t}_{\text{renaming k}\rightarrow \text{t}}$.

  

Accumulating a whole episode of updates:

  

$\Delta w_k = \sum_{t=0}^{T-1} \alpha \delta_t e_t$

  

where $e_t = \sum_{j=0}^t \gamma^{t-j}x_j$

$= \sum_{j=0}^{t-1} \gamma^{t-j} x_j + x_t$

$= \gamma \underbrace{\sum_{j=0}^{t-1} \gamma^{t-1-j}x_j}_{=e_{t-1}} + x_t$

$= \gamma e_{t-1} + x_t$

  

Accumulating a whole episode of updates:

  

$\Delta w_t \equiv \alpha \delta_t e_t$ (one time step)

$\Delta w_k = \sum_{t=0}^{T-1} \Delta w_t$ (whole episode)

  

where $e_t = \gamma e_{t-1} + x_t$.

(And then apply $\Delta w$ at the end of the episode)

  

### **Mixing multi-step returns &amp; traces**

  

Reminder: mixed multi-step return:

  

$G_t^\lambda = R_{t+1} + \gamma((1-\lambda)v(S_{t+1}) + \lambda G_{t+1}^\lambda)$

  

The associated error and trace update are:

  

$G_t^\lambda = \sum_{k=0}^{T-t} \lambda^k \gamma^k \delta_{t+k}$ (same as before, but with $\lambda \gamma$ instead of $\gamma$)

$\implies e_t = \gamma \lambda e_{t-1} + x_t$ and $\Delta w_t = \alpha \delta_t e_t$.
***
# Lecture 6


### **Model-Free Policy Evaluation**

*   **Monte Carlo (MC):**
    
    ${G^{MC}}_t = R_{t+1} + γR_{t+2} + γ^2 R_{t+3} + ...$

    ${G^{MC}}_t = R_{t+1} + γ{G^{MC}}_{t+1}$

*   **Temporal Difference (TD(0)):**

    ${G^{(1)}}_t = R_{t+1} + γv_t(S_{t+1})$

*   **n-step TD:**

    ${G^{(n)}}_t = R_{t+1} + γR_{t+2} + ... + γ^{n-1} R_{t+n} + γ^n v_t (S_{t+n})$

    ${G^{(n)}}_t = R_{t+1} + γ {G^{(n-1)}}_{t+1}$

*   **TD(λ):**

    ${G^{λ}}_t = R_{t+1} + γ[(1-λ)v_t(S_{t+1}) + λ{G^λ}_{t+1}]$

### **Model-Free Policy Iteration Using Action-Value Function**

*   **Greedy policy improvement over v(s):**

    $π'(s) = argmax_a E[R_{t+1} + γv(S_{t+1}) | S_t = s, A_t = a]$

*   **Greedy policy improvement over q(s, a):**

    $π'(s) = argmax_a q(s, a)$

### **Monte-Carlo Generalized Policy Iteration**

*   **Update rule for action-value function:**

    $q(S_t, A_t) ← q(S_t, A_t) + α_t (G_t - q(S_t, A_t))$

*   **Possible values for α:**

    *   $α_t = \frac{1}{N(S_t, A_t)}$

    *   $α_t = \frac{1}{k}$

*   **Policy improvement:**

    *   $ε ← \frac{1}{k}$

    *   $π ← ε-greedy(q)$

### **GLIE (Greedy in the Limit with Infinite Exploration)**

*   **Condition for exploring all state-action pairs:**

    $∀s, a \lim_{t→∞} N_t (s, a) = ∞$

*   **Convergence to greedy policy:**

    $\lim_{t→∞} π_t (a|s) = \mathbb{I} (a = argmax_{a'} q_t (s, a'))$

*   **Example:**

    $ε_k = \frac{1}{k}$

### **Updating Action-Value Functions with SARSA**

*   **SARSA update rule:**

    $q_{t+1}(S_t, A_t) = q_t(S_t, A_t) + α_t (R_{t+1} + γq(S_{t+1}, A_{t+1}) - q(S_t, A_t))$

### **Dynamic Programming and TD Learning**

*   **Policy evaluation (dynamic programming):**

    *   For value function: $v_{k+1}(s) = E[R_{t+1} + γv_k (S_{t+1}) | S_t = s, A_t ∼ π(S_t)]$

    *   For action-value function: $q_{k+1}(s, a) = E[R_{t+1} + γq_k(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$

*   **Value iteration (dynamic programming):**

    *   For value function: $v_{k+1}(s) = max_a E[R_{t+1} + γv_k(S_{t+1}) | S_t = s, A_t = a]$

    *   For action-value function: $q_{k+1}(s, a) = E[R_{t+1} + γ max_{a'} q_k (S_{t+1}, a') | S_t = s, A_t = a]$

*   **TD learning:**

    *   **TD:** $v_{t+1}(S_t) = v_t (S_t) + α_t (R_{t+1} + γ v_t (S_{t+1}) - v_t(S_t))$

    *   **SARSA:** $q_{t+1}(s, a) = q_t (S_t, A_t) + α_t (R_{t+1} + γ q_t (S_{t+1}, A_{t+1}) - q_t(S_t, A_t))$

    *   **Q-learning:** $q_{t+1}(s, a) = q_t (S_t, A_t) + α_t (R_{t+1} + γ max_{a'} q_t (S_{t+1}, a') - q_t (S_t, A_t))$

### **Q-Learning Control Algorithm**

*   **Q-learning update rule:**

    $q_{t+1}(s, a) = q_t (S_t, A_t) + α_t (R_{t+1} + γ max_{a'} q_t (S_{t+1}, a') - q_t(S_t, A_t))$

*   **Conditions for step-sizes:**

    $∑_t α_t = ∞$

    $∑_t {α_t}^2 < ∞$

*   **Example:**

    $α = \frac{1}{t^ω}$, with $ω ∈ (0.5, 1)$

### **Double Q-learning**

*   **Update rules:**

    *   For $q$: $R_{t+1} + γ {q'}_t (S_{t+1}, argmax_a q_t (S_{t+1}, a))$

    *   For $q'$: $R_{t+1} + γ q_t (S_{t+1}, argmax_a {q'}_t (S_{t+1}, a))$

### **Importance Sampling Corrections**

*   **Estimating expectation under target distribution d:**

    $E_{x∼d}[f(x)] = ∑ d(x) f(x)$

    $= ∑ d'(x) \frac{d(x)}{d'(x)} f(x)$

    $=E_{x∼d'} [\frac{d(x)}{d'(x)} f(x)]$

*   **Estimating one-step reward:**

    $E[R_{t+1} | S_t = s, A_t ∼ π] = ∑_a π(a|s) r(s, a)$

    $=∑_{µ(a|s)} \frac{π(a|s)}{µ(a|s)} r(s, a)$

    $=E[\frac{π(A_t|S_t)}{µ(A_t|S_t)} R_{t+1} | S_t = s, A_t ∼ µ]$

### **Importance Sampling for Off-Policy Monte-Carlo**

*   **Correcting the return:**

    $\frac{p(τ_t | π)}{p(τ_t | µ)} G(τ_t)$

    $= \frac{p(A_t | S_t, π) p(R_{t+1}, S_{t+1} | S_t, A_t) p(A_{t+1} | S_{t+1}, π) ...}{p(A_t | S_t, µ) p(R_{t+1}, S_{t+1} | S_t, A_t) p(A_{t+1} | S_{t+1}, µ) ...} G_t$

    $= \frac{p(A_t | S_t, π) p(A_{t+1} | S_{t+1}, π) ...}{p(A_t | S_t, µ) p(A_{t+1} | S_{t+1}, µ) ...} G_t$

    $=\frac{π(A_t | S_t)}{µ(A_t | S_t)} \frac{π(A_{t+1} | S_{t+1})}{µ(A_{t+1} | S_{t+1})} ... G_t$

### **Importance Sampling for Off-Policy TD Updates**

*   **TD update with importance sampling correction:**

    $v(S_t) ← v(S_t) + α (\frac{π(A_t | S_t)}{µ(A_t | S_t)} (R_{t+1} + γ v(S_{t+1})) - v(S_t))$

*   **Proof:**

    $E_µ [\frac{π(A_t|S_t)}{µ(A_t|S_t)}(R_{t+1} + γv(S_{t+1})) - v(S_t) | S_t = s]$

    $= ∑_a µ(a|s)(\frac{π(a|s)}{µ(a|s)} E[R_{t+1} + γv(S_{t+1})| S_t = s, A_t = a] - v(s))$

    $= ∑_a π(a|s)E[R_{t+1} + γv(S_{t+1}) | S_t = s, A_t = a] - ∑_a µ(a|s) v(s)$

    $=∑_a π(a|s) E[R_{t+1} + γv(S_{t+1}) | S_t = s, A_t = a] - ∑_a π(a|s) v(s)$

    $=∑_a π(a|s) (E[R_{t+1} + γv(S_{t+1})| S_t = s, A_t = a] - v(s))$

    $= E_π [R_{t+1} + γv(S_{t+1}) - v(s) | S_t = s]$

### **Expected SARSA**

*   **Update rule:**

    $q(S_t, A_t) ← q(S_t, A_t) + α (R_{t+1} + γ ∑_a π(a|S_{t+1}) q(S_{t+1}, a) - q(S_t, A_t))$

### **Off-Policy Control with Q-Learning**

*   **Target policy:**

    $π(S_{t+1}) = argmax_{a'} q(S_{t+1}, a')$

*   **Q-learning target:**

    $R_{t+1} + γ ∑_a π_{greedy}(a | S_{t+1}) q(S_{t+1}, a)$

    $= R_{t+1} + γ max_a q(S_{t+1}, a)$

### **On-Policy Control with SARSA**

*   **SARSA target:**

    $target = R_{t+1} + γq(S_{t+1}, A_{t+1})$

# lecture 7

### **Gradient Descent**

Let $J(w)$ be a differentiable function of parameter vector $w$.

The gradient of $J(w)$ is:

*   $\nabla_w J(w) = \begin{pmatrix} \frac{\partial J(w)}{\partial w_1} \\ ... \\ \frac{\partial J(w)}{\partial w_n} \end{pmatrix}$

The goal is to minimize $J(w)$.

The method is to move $w$ in the direction of the negative gradient:

*   $\Delta w = - \frac{1}{2} \alpha \nabla_w J(w)$

Where $\alpha$ is a step-size parameter.

### **Approximate Values by Stochastic Gradient Descent**

The goal is to find $w$ that minimizes the difference between $v_w(s)$ and $v_\pi (s)$.

*   $J(w) = E_{S~d}[(v_\pi (S) - v_w(S))^2]$

Where $d$ is a distribution over states (typically induced by the policy and dynamics).

Gradient descent is:

*   $\Delta w = - \frac{1}{2}\alpha \nabla_w J(w) = \alpha E_d (v_\pi (S) - v_w(S)) \nabla_w v_w(S)$

Stochastic gradient descent (SGD) samples the gradient:

*   $\Delta w = \alpha (G_t - v_w(S_t))\nabla_w v_w(S_t)$

Note: Monte Carlo return $G_t$ is a sample for $v_\pi (S_t)$.

The shorthand for $\nabla_w v_w (S_t) |_{w=w_t}$ is often written as $\nabla v(S_t)$.

### **Feature Vectors**

Represent state by a feature vector.

*   $x(s) = \begin{pmatrix} x_1(s) \\ ... \\ x_n(s) \end{pmatrix}$

$x : S \rightarrow R^n$ is a fixed mapping from state (e.g., observation) to features.

Shorthand: $x_t = x(S_t)$.

### **Linear Value Function Approximation**

Approximate the value function by a linear combination of features.

*   $v_w (s) = w^T x(s) = \sum_{j=1}^n x_j(s)w_j$

The objective function (“loss”) is quadratic in $w$.

*   $J(w) = E_{S~d}[(v_\pi (S) - w^T x(S))^2]$

Stochastic gradient descent converges on the global optimum.

The update rule is simple.

*   $\nabla_w v_w (S_t) = x(S_t) = x_t \implies \Delta w = \alpha (v_\pi (S_t) - v_w(S_t))x_t$

Update = step-size $\times$ prediction error $\times$ feature vector

### **Incremental Prediction Algorithms**

Updating towards the true value function $v_\pi(s)$ is not possible, so a target is substituted for $v_\pi(s)$.

*   For MC, the target is the return $G_t$:
    *   $\Delta w_t = \alpha (G_t - v_w(s))\nabla_w v_w(s)$
*   For TD, the target is the TD target $R_{t+1} + \gamma v_w (S_{t+1})$:
    *   $\Delta w_t = \alpha (R_{t+1} + \gamma v_w (S_{t+1}) - v_w(S_t)) \nabla_w v_w (S_t)$
*   For TD($\lambda$):
    *   $\Delta w_t = \alpha (G_t^\lambda - v_w(S_t))\nabla_w v_w(S_t)$
    *   $G_t^\lambda = R_{t+1} + \gamma ((1-\lambda)v_w(S_{t+1}) + \lambda G_{t+1}^\lambda)$

### **Monte-Carlo with Value Function Approximation**

The return $G_t$ is an unbiased sample of $v_\pi (s)$.

Supervised learning can be applied to online training data:

*   ${(S_0, G_0),...,(S_t, G_t)}$

For example, using linear Monte-Carlo policy evaluation:

*   $\Delta w_t = \alpha (G_t - v_w(S_t))\nabla_w v_w(S_t) = \alpha (G_t - v_w(S_t))x_t$

Linear Monte-Carlo evaluation converges to the global optimum.

It converges even when using non-linear value function approximation (but perhaps to a local optimum).

### **TD Learning with Value Function Approximation**

The TD target $R_{t+1} + \gamma v_w(S_{t+1})$ is a biased sample of the true value $v_\pi(S_t)$.

Supervised learning can still be applied to training data:

*   ${(S_0, R_1 + \gamma v_w(S_1)),...,(S_t, R_{t+1} + \gamma v_w(S_{t+1}))}$

For example, using linear TD:

*   $\Delta w_t = \alpha \underbrace{(R_{t+1} + \gamma v_w (S_{t+1} - v_w(S_t)))}_{= \delta_t, ‘TD \text{ } error’} \nabla_w v_w(S_t) = \alpha \delta_t x_t$

This is similar to a non-stationary regression problem, but the target depends on the parameters.

### **Control with Value Function Approximation**

**Policy evaluation**: Approximate policy evaluation, $q_w \approx q_\pi$.

**Policy improvement**: For example, $\epsilon$-greedy policy improvement.

### **Action-Value Function Approximation**

Approximate the action-value function $q_w(s, a) \approx q_\pi (s, a)$.

For instance, with linear function approximation with state-action features:

*   $q_w(s, a) = x(s, a)^T w$

Stochastic gradient descent update:

*   $\Delta w = \alpha (q_\pi(s, a) - q_w(s, a)) \nabla_w q_w(s, a) = \alpha (q_\pi (s, a) - q_w(s, a))x(s, a)$

Approximate the action-value function $q_w(s, a) \approx q_\pi (s, a)$.

For instance, with linear function approximation with state features:

*   $q_w(s) = Wx(s) (W \in R^{m \times n}, x(s) \in R^n \implies q \in R^m)$
*   $q_w(s, a) = q_w(s)[a] = x(s)^T w_a$ (where $w_a = W_a \cdot$)

Stochastic gradient descent update:

*   $\Delta w_a = \alpha (q_\pi (s, a) - q_w(s, a)) \nabla_w q_w(s, a) = \alpha (q_\pi (s, a) - q_w(s, a))x(s) \forall a, b: \Delta w_b = 0$

Equivalently:

*   $\Delta W = \alpha (q_\pi (s, a) - q_w(s, a))i_a x(s)^T$

where $i_a = (0,...,0,1,0,...,0)$ with $i_a[a] = 1, i_a[b] = 0$ for $b \neq a$.

### **Action-Value Function Approximation**

**Action in**: $q_w(s, a) = w^T x(s, a)$

**Action out**: $q_w(s) = Wx(s)$ such that $q_w(s, a) = q_w(s)[a]$

It is unclear which is better in general.

Action-in is easier if continuous actions are desired (later lecture).

Action-out is common for (small) discrete action spaces (e.g., DQN).

### **Convergence of MC**

With linear functions (and suitably decaying step size), MC converges to:

*   $w^{MC} = argmin_w E_\pi [(G_t - v_w(S_t))^2] = E_\pi [x_t x_t^T]^{-1} E_\pi[G_t x_t]$

(Notation: the state distribution implicitly depends on $\pi$ here.)

Verifying the fixed point:

*   $\nabla_{w^{MC}} E[(G_t - v_{w^{MC}}(S_t))^2] = E[(G_t - v_{w^{MC}} (S_t))x_t] = 0$
*   $E[(G_t - x_t^T w^{MC})x_t] = 0$
*   $E[G_t x_t - x_t x_t^T w^{MC}] = 0$
*   $E[x_t x_t^T] w^{MC} = E[G_t x_t]$
*   $w^{MC} = E[x_t x_t^T]^{-1} E[G_t x_t]$

Agent state $S_t$ does not have to be Markov: the fixed point only depends on observed data (and features).

### **Convergence of TD**

With linear functions, TD converges to:

*   $w^{TD} = E[x_t(x_t - \gamma x_{t+1})^T]^{-1}E[R_{t+1} x_t]$

(in continuing problems with fixed $\gamma < 1$, and with appropriately decaying $\alpha_t \rightarrow 0$).

Verify (assuming $\alpha_t$ does not correlate with $R_{t+1}, x_t, x_{t+1}$):

*   $E[\Delta w^{TD}] = 0 = E[\alpha_t(R_{t+1} + \gamma x_{t+1}^T w^{TD} - x_t^T w^{TD}) x_t]$
*   $0 = E[\alpha_t R_{t+1} x_t] + E[\alpha_t x_t (\gamma x_{t+1}^T - x_t^T) w^{TD}]$
*   $E[\alpha_t x_t(x_t - \gamma x_{t+1})^T] w^{TD} = E[\alpha_t R_{t+1}x_t]$
*   $w^{TD} = E[x_t (x_t - \gamma x_{t+1})^T]^{-1} E[R_{t+1} x_t]$

This differs from the MC solution.

Typically, the asymptotic MC solution is preferred (smallest prediction error).

TD often converges faster (especially intermediate $\lambda \in$ or $n \in {1,..., \infty}$).

Let $VE(w)$ denote the value error:

*   $VE(w) = ||v_\pi - v_w||_{d_\pi} = \sum_{s \in S} d_\pi (s)(v_\pi (s) - v_w(s))^2$

The Monte Carlo solution minimizes the value error.

**Theorem:**

*   $VE(w^{TD}) \le \frac{1}{1-\gamma}VE(w^{MC}) = \frac{1}{1 - \gamma} min_w VE(w)$

### **TD is Not a Gradient**

The TD update is not a true gradient update:

*   $w \leftarrow w + \alpha (r + \gamma v_w(s') - v_w(s)) \nabla v_w (s)$

It is a stochastic approximation update.

Stochastic approximation algorithms are a broader class than SGD.

SGD always converges (with bounded noise, decaying step size, stationarity, ...).

TD does not always converge...

### **Example of Divergence**

Consider $w_t > 0$. If $\gamma > \frac{1}{2}$, then $w_{t+1} > w_t$.

$\implies \lim_{t \rightarrow \infty} w_t = \infty$

Algorithms that combine bootstrapping, off-policy learning, and function approximation may diverge. This is sometimes called the **deadly triad**.

Consider sampling on-policy, over an episode. Update:

*   $\Delta w = \alpha(0 + 2\gamma w - w) + \alpha (0 + \gamma 0 - 2w) = \alpha (2\gamma -3)w$

The multiplier is negative, for all $\gamma \in$.

$\implies$ convergence ($w$ goes to zero, which is optimal here).

With tabular features, this is just regression. The answer may be sub-optimal, but no divergence occurs. Specifically, if only $v(s)$ (= left-most state) is updated:

*   $v(s) = w$ will converge to $\gamma v(s')$
*   $v(s') = w$ will stay where it was initialized

What if multi-step returns are used? Still only consider updating the left-most state.

*   $\Delta w = \alpha (r + \gamma (G_t^\lambda - v(s))$
*   $= \alpha (r + \gamma ((1 - \lambda) v(s') + \lambda (r' + v(s'')) - v(s))$ $(r = r' = v(s'') = 0)$
*   $= \alpha (2 \gamma (1-\lambda) - 1)w$

The multiplier is negative when $2\gamma(1-\lambda) < 1 \implies \lambda > 1 - \frac{1}{2\gamma}$.

For example, when $\gamma = 0.9$, then $\lambda > \frac{4}{9} \approx 0.45$ is needed.

### **Residual Bellman Updates**

TD: $\Delta w_t = \alpha \delta \nabla v_w(S_t)$ where $\delta_t = R_{t+1} + \gamma v_w (S_{t+1}) - v_w (S_t)$.

This update ignores dependence of $v_w (S_{t+1})$ on $w$.

Alternative: Bellman residual gradient update:

*   Loss: $E[\delta_t^2]$
*   Update: $\Delta w_t = \alpha \delta_t \nabla_w (v_w(S_t) - \gamma v_w(S_{t+1}))$

This tends to work worse in practice. Bellman residuals smooth, whereas TD methods predict. Smoothed values may lead to suboptimal decisions.

Alternative: minimize the Bellman error:

*   Loss: $E[\delta_t]^2$
*   Update: $\Delta w_t = \alpha \delta_t \nabla_w (v_w(S_t) - \gamma v_w (S_{t+1}')$

...but this requires a second independent sample $S_{t+1}'$ which could (randomly) differ from $S_{t+1}$ (so this can't be used online).

### **Batch Reinforcement Learning**

Batch methods seek to find the best-fitting value function for a given set of past experience (“training data”).

### **Least Squares Temporal Difference**

Which parameters $w$ give the best fitting linear value function $v_w(s) = w^T x(s)$?

Recall:

*   $E[(R_{t+1} + \gamma v_w(S_{t+1}) - v_w(S_t))x_t] = 0$
*   $\implies w^{TD} = E[x_t (x_t - \gamma x_{t+1})^T]^{-1} E[R_{t+1}x_t]$

A closed-form empirical loss can be used:

*   $\frac{1}{t} \sum_{i=0}^t (R_{i+1} + \gamma v_w(S_{i+1}) - v_w(S_i))x_i = 0$
*   $\implies w^{LSTD} = (\sum_{i=0}^t x_i (x_i - \gamma x_{i+1})^T)^{-1} (\sum_{i=0}^t R_{i+1} x_i)$

This is called least-squares TD (LSTD).

*   $w_t = \underbrace{(\sum_{i=0}^t x_i (x_i - \gamma x_{i+1})^T)^{-1}}_{= A_t^{-1}} \underbrace{(\sum_{i=0}^t R_{i+1}x_i)}_{= b_t}$ (LSTD estimate)

$b_t$ and $A_t^{-1}$ can be updated incrementally online.

*   **Naive approach ($O(n^3)$):**
    *   $A_{t+1} = A_t + x_t (x_t - \gamma x_{t+1})^T$
    *   $b_{t+1} = b_t + R_{t+1} x_t$
*   **Faster approach ($O(n^2)$):** directly update $A^{-1}$ with Sherman-Morrison:
    *   $A_{t+1}^{-1} = A_t^{-1} - \frac{A_t^{-1} x_t(x_t - \gamma x_{t+1})^T A_t^{-1}}{1 + (x_t - \gamma x_{t+1})^T A_t^{-1} x_t}$
    *   $b_{t+1} = b_t + R_{t+1} x_t$

Still more compute per step than TD ($O(n)$).

In the limit, LSTD and TD converge to the same fixed point.

LSTD can be extended to multi-step returns: LSTD($\lambda$).

LSTD can be extended to action values: LSTDQ.

It can also be interlaced with policy improvement: least-squares policy iteration (LSPI).

### **Experience Replay**

Given experience consisting of trajectories of experience:

*   $D = {S_0, A_0, R_1, S_1,...,S_t}$

Repeat:

1.  Sample transition(s), e.g., $(S_n, A_n, R_{n+1}, S_{n+1})$ for $n \le t$
2.  Apply stochastic gradient descent update:
    *   $\Delta w = \alpha (R_{n+1} + \gamma v_w(S_{n+1}) - v_w(S_n)) \nabla_w v_w (S_n)$
3.  Old data can be reused.

This is also a form of batch learning.

Beware: the data may be off-policy if the policy changes.

### **Example: Neural Q-Learning**

Online neural Q-learning may include:

*   Neural network: $O_t \rightarrow q_w$ (action-out)
*   Exploration policy: $\pi_t = \epsilon - greedy(q_t)$, and then $A_t \sim \pi_t$
*   Weight update: for instance Q-learning
    *   $\Delta w \propto (R_{t+1} + \gamma max_a q_w (S_{t+1}, a) - q_w (S_t, A_t)) \nabla_w q_w (S_t, A_t)$
*   An optimizer to minimize the loss (e.g., SGD, RMSprop, Adam)

Often, the weight update is implemented via a 'loss'.

*   $L(w) = \frac{1}{2} (R_{t+1} + \gamma \begin{bmatrix} max_a q_w (S_{t+1}, a)\end{bmatrix} - q_w(S_t, A_t))^2$ where $n\cdot o$ denotes stopping the gradient, so that the semi-gradient is $\Delta w$.

Note that $L(w)$ is not a real loss; it just happens to have the right gradient.

### **Example: DQN**

DQN (Mnih et al. 2013, 2015) includes:

*   A neural network: $O_t \rightarrow q_w$ (action-out)
*   An exploration policy: $\pi_t = \epsilon - greedy(q_t)$, and then $A_t \sim \pi_t$
*   A replay buffer to store and sample past transitions $(S_i, A_i, R_{i+1}, S_{i+1})$
*   Target network parameters $w^-$
*   A Q-learning weight update on $w$ (uses replay and target network)
    *   $\Delta w = (R_{i+1} + \gamma max_a q_{w^-}(S_{i+1}, a) - q_w(S_i, A_i)) \nabla_w q_w(S_i, A_i)$
*   Update $w_t^- \leftarrow w_t$ occasionally (e.g., every 10000 steps)
*   An optimizer to minimize the loss (e.g., SGD, RMSprop, or Adam)

Replay and target networks make RL look more like supervised learning. Neither is strictly necessary, but they helped for DQN.

“DL-aware RL”

# lecture 8



### **Model Learning**

The goal of model learning is to estimate a model *M<sub>η</sub>* from experience {*S<sub>1</sub>, A<sub>1</sub>, R<sub>2</sub>,..., S<sub>T</sub>*}. This is a supervised learning problem over a dataset of state transitions observed in the environment.  To learn a suitable function *f<sub>η</sub>(s, a) = r, s'*, one can choose a functional form for *f*, pick a loss function (e.g., mean-squared error), and then find parameters *η* that minimize empirical loss.

**If *f<sub>η</sub>(s, a) = r, s'*, then we would hope *s' ≈ E[S<sub>t+1</sub> | s = S<sub>t</sub>, a = A<sub>t</sub>]*.**

### **Expectation Models**

Considering an expectation model *f<sub>η</sub>(ϕ<sub>t</sub>) = E[ϕ<sub>t+1</sub>]* and value function *v<sub>θ</sub>(ϕ<sub>t</sub>) = θ<sup>></sup>ϕ<sub>t</sub>*:

*   **E[*v<sub>θ</sub>(ϕ<sub>t+1</sub>*)| S<sub>t</sub> = s] = E[θ<sup>></sup>ϕ<sub>t+1</sub> | S<sub>t</sub> = s]**
*   **= θ<sup>></sup>E[ϕ<sub>t+1</sub> | S<sub>t</sub> = s]**
*   **= *v<sub>θ</sub>(E[ϕ<sub>t+1</sub> | S<sub>t</sub> = s])*.**

If the model is also linear: *f<sub>η</sub>(ϕ<sub>t</sub>) = Pϕ<sub>t</sub>* for some matrix *P*. Then we can unroll an expectation model multiple steps into the future, and still have **E[*v<sub>θ</sub>(ϕ<sub>t+n</sub>*) | S<sub>t</sub> = s] = *v<sub>θ</sub>(E[ϕ<sub>t+n</sub> | S<sub>t</sub> = s])*.**

### **Stochastic Models**

In stochastic models:

**R̂<sub>t+1</sub>, Ŝ<sub>t+1</sub> = p̂(S<sub>t</sub> ,A<sub>t</sub> , ω)**, 

where ω is a noise term.

### **Full Models**

The expected value in full models can be represented as:

**E[*v(S<sub>t+1</sub>) | S<sub>t</sub> = s] = ∑ <sub>a</sub>π(a | s) ∑ <sub>s′</sub>p̂(s, a, s′)(r̂(s, a, s′) + γv(s′))**

The expected value unrolled multiple steps into the future:

**E[*v(S<sub>t+n</sub>) | S<sub>t</sub> = s] = ∑ <sub>a</sub>π(a | s) ∑ <sub>s′</sub>p̂(s, a, s′)( r̂(s, a, s′) + γ ∑ <sub>a′</sub>π(a′ | s′) ∑ <sub>s′′</sub>p̂(s′, a′, s′′)( r̂(s′, a′, s′′) + γ<sup>2</sup> ∑ <sub>a′′</sub>π(a′′ | s′′) ∑ <sub>s′′′</sub>p̂(s′′, a′′, s′′′)( r̂(s′′, a′′, s′′′) + ... )))**

### **Table Lookup Models**

In table lookup models, the model is an explicit MDP.

**p̂<sub>t</sub>(s′ | s, a) = 1/N(s, a)  <sup>t−1</sup>∑ <sub>k=0</sub> I (S<sub>k</sub> = s,A<sub>k</sub> = a, S<sub>k+1</sub> = s′)**

**Ep̂<sub>t</sub> [R<sub>t+1</sub> | S<sub>t</sub> = s,A<sub>t</sub> = a] = 1/N(s, a)  <sup>t−1</sup>∑ <sub>k=0</sub> I (S<sub>k</sub> = s,A<sub>k</sub> = a)R<sub>k+1</sub>**

### **Linear Expectation Models**

In linear expectation models, the expected next states are parameterized by a square matrix *T<sub>a</sub>*, for each action *a*.

**ŝ′(s, a) = T<sub>a</sub>ϕ(s)**

The rewards are parameterized by a vector *w<sub>a</sub>*, for each action *a*.

**r̂(s, a) = w<sup>T</sup> <sub>a</sub>ϕ(s)**

On each transition (*s, a, r, s'*), a gradient descent step can be applied to update *w<sub>a</sub>* and *T<sub>a</sub>* to minimize the loss:

**L(s, a, r, s′) = (s′ − T<sub>a</sub>ϕ(s))<sup>2</sup> + (r − w<sup>T</sup><sub>a</sub> ϕ(s))<sup>2</sup>**

### **Sample-Based Planning with a Learned Model**

In sample-based planning with a learned model, sample experience is taken from the model:

**S,R ∼ p̂<sub>η</sub>(· | s, a)**

### **Monte-Carlo Learning**

In the AB example using Monte-Carlo learning: **V(A) = 1, V(B) = 0.75**.

### **Real and Simulated Experience**

Real experience is sampled from the environment (true MDP):

**r, s′ ∼ p**

Simulated experience is sampled from the model (approximate MDP):

**r, s′ ∼ p̂<sub>η</sub>**

### **Prediction via Monte-Carlo Simulation**

Given a parameterized model *M<sub>η</sub>* and a simulation policy *π*, *K* episodes from the current state *S<sub>t</sub>* are simulated.

**{*S<sup>k</sup> <sub>t</sub> = S<sub>t</sub>, A<sup>k</sup><sub>t</sub>, R<sup>k</sup> <sub>t+1</sub>, S<sup>k</sup> <sub>t+1</sub>,..., S<sup>k</sup><sub>T</sub>*}<sup>K</sup><sub>k=1</sub> ∼ p̂<sub>η</sub>, π**

The state is evaluated by mean return (Monte-Carlo evaluation):

**v(S<sub>t</sub>) = 1/K  <sup>K</sup>∑ <sub>k=1</sub> G<sup>k</sup> <sub>t</sub>  v<sub>π</sub>(S<sub>t</sub>)**

### **Control via Monte-Carlo Simulation**

Given a model *M<sub>η</sub>* and a simulation policy *π*:

*   For each action *a ∈ A*, simulate *K* episodes from current (real) state *s*:

    **{*S<sup>k</sup> <sub>t</sub> = s, A<sup>k</sup><sub>t</sub> = a, R<sup>k</sup><sub>t+1</sub>, S<sup>k</sup><sub>t+1</sub>, A<sup>k</sup><sub>t+1</sub>,..., S<sup>k</sup><sub>T</sub>*}<sup>K</sup><sub>k=1</sub> ∼ M<sub>ν</sub>, π**
*   Evaluate actions by mean return (Monte-Carlo evaluation):

    **q(s, a) = 1/K  <sup>K</sup>∑ <sub>k=1</sub> G<sup>k</sup><sub>t</sub>  q<sub>π</sub>(s, a)**
*   Select current (real) action with maximum value:

    **A<sub>t</sub> = argmax<sub>a∈A</sub> q(S<sub>t</sub> , a)**

### **Monte-Carlo Tree Search**

The estimated action values *q(s, a)* are calculated as follows:

**q(s, a) = 1/N(s, a)  <sup>K</sup>∑ <sub>k=1</sub><sup>T</sup>∑ <sub>u=t</sub> 1(S<sup>k</sup><sub>u</sub>, A<sup>k</sup> <sub>u</sub> = s, a)G<sup>k</sup> <sub>u</sub> q<sub>π</sub>(s, a)**

# lecture 9


### **Policy Objective Functions**

**1. Episodic-return Objective:**

*   **J<sub>G</sub>(θ) = E<sub>S<sub>0</sub>∼d<sub>0</sub>,π<sub>θ</sub></sub> [<sup>∞</sup>∑<sub>t=0</sub> γ<sup>t</sup>R<sub>t+1</sub>]**

    This equation represents the expected total return per episode, where:

    *   *J<sub>G</sub>(θ)* is the episodic-return objective function.
    *   *θ* represents the policy parameters.
    *   *E* denotes the expectation.
    *   *S<sub>0</sub>* is the starting state, sampled from the initial state distribution *d<sub>0</sub>*.
    *   *π<sub>θ</sub>* is the policy parameterized by *θ*.
    *   *γ* is the discount factor.
    *   *R<sub>t+1</sub>* is the reward received at time step *t+1*.

*   **J<sub>G</sub>(θ) = E<sub>S<sub>0</sub>∼d<sub>0</sub>,π<sub>θ</sub></sub> \[G<sub>0</sub>\]** 

    This is a simplified representation of the previous equation, where *G<sub>0</sub>* represents the total return from the start state *S<sub>0</sub>*.

*   **J<sub>G</sub>(θ) = E<sub>S<sub>0</sub>∼d<sub>0</sub></sub> \[E<sub>π<sub>θ</sub></sub> \[G<sub>t</sub> | S<sub>t</sub> = S<sub>0</sub>\]\]**

    This equation expresses the objective as the expected return starting from *S<sub>0</sub>* and following the policy *π<sub>θ</sub>*.

*   **J<sub>G</sub>(θ) = E<sub>S<sub>0</sub>∼d<sub>0</sub></sub> \[v<sub>π<sub>θ</sub></sub> (S<sub>0</sub>)\]**

    This equation shows that the objective is equivalent to the expected value of the starting state *S<sub>0</sub>* under the policy *π<sub>θ</sub>*.
*   *d<sub>0</sub>* is the start-state distribution. 
*   This objective equals the expected value of the start state.

**2. Average-reward Objective:**

*   **J<sub>R</sub>(θ) = E<sub>π<sub>θ</sub></sub> \[R<sub>t+1</sub>\]** 

    This equation represents the average reward per time step, where:

    *   *J<sub>R</sub>(θ)* is the average-reward objective function.

*   **J<sub>R</sub>(θ) = E<sub>S<sub>t</sub>∼d<sub>π<sub>θ</sub></sub></sub> \[E<sub>A<sub>t</sub>∼π<sub>θ</sub> (S<sub>t</sub> )</sub> \[R<sub>t+1</sub> | S<sub>t</sub> \]\]** 

    This equation expresses the average reward as the expectation over the state distribution *d<sub>π<sub>θ</sub></sub>* induced by the policy and the action distribution *π<sub>θ</sub>(S<sub>t</sub>)* at each state.

*   **J<sub>R</sub>(θ) = ∑ <sub>s</sub> d<sub>π<sub>θ</sub></sub> (s) ∑ <sub>a</sub> π<sub>θ</sub>(s, a) ∑ <sub>r</sub> p(r | s, a)r**

    This equation calculates the average reward by summing over all possible states, actions, and rewards, weighted by their respective probabilities.
*   *d<sub>π</sub>(s) = p(S<sub>t</sub> = s | π)* is the probability of being in state *s* in the long run. 
*   Think of it as the ratio of time spent in *s* under policy *π*. 

### **Policy Gradient**

*   **∆θ = α∇<sub>θ</sub>J(θ)**

    This equation represents the update rule for the policy parameters using gradient ascent, where:

    *   *∆θ* is the change in policy parameters.
    *   *α* is the learning rate or step-size parameter.
    *   *∇<sub>θ</sub>J(θ)* is the gradient of the objective function with respect to the policy parameters.

*   **∇<sub>θ</sub>J(θ) = [∂J(θ)/∂θ<sub>1</sub> ... ∂J(θ)/∂θ<sub>n</sub>]**

    This equation represents the gradient vector of the objective function.

*   **∇<sub>θ</sub>J(θ) = ∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R]**.

    This equation expresses the policy gradient as the gradient of the expected reward.

### **Contextual Bandit Policy Gradient**

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = E<sub>π<sub>θ</sub></sub> \[R(S, A)∇<sub>θ</sub> log π(A|S)\]**

    This equation uses the score function trick to express the gradient of the expected reward in terms of the log probability of the policy.
*   Expectation is over *d* (states) and *π* (actions).
*   For now, *d* does not depend on *π*.
*   Also known as **REINFORCE (Williams, 1992)**.

*   **θ <sub>t+1</sub> = θ <sub>t</sub> + αR<sub>t+1</sub>∇<sub>θ</sub> log π<sub>θ<sub>t</sub></sub> (A<sub>t</sub> |S<sub>t</sub> )**

    This equation is the update rule for the policy parameters in the contextual bandit setting.

### **The Score Function Trick**

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = ∇<sub>θ</sub> ∑ <sub>s</sub> d(s) ∑ <sub>a</sub> π<sub>θ</sub>(a|s) rsa** 

    The equation breaks down the gradient of the expected reward into a sum over states and actions, where:

    *   *rsa* represents the expected reward given state *s* and action *a*.
*   Let *rsa = E \[R(S, A) | S = s, A = s]*.

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = ∑ <sub>s</sub> d(s) ∑ <sub>a</sub> rsa ∇<sub>θ</sub>π<sub>θ</sub>(a|s)**

    This equation isolates the gradient of the policy.

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = ∑ <sub>s</sub> d(s) ∑ <sub>a</sub> rsa π<sub>θ</sub>(a|s) ∇<sub>θ</sub>π<sub>θ</sub>(a|s) / π<sub>θ</sub>(a|s)**

    This step introduces the policy probability term in both the numerator and denominator.

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = ∑ <sub>s</sub> d(s) ∑ <sub>a</sub> π<sub>θ</sub>(a|s) rsa ∇<sub>θ</sub> log π<sub>θ</sub>(a|s)**

    This equation uses the fact that the derivative of the logarithm of a function is equal to the derivative of the function divided by the function itself.

*   **∇<sub>θ</sub>E<sub>π<sub>θ</sub></sub> \[R(S, A)\] = E<sub>d,π<sub>θ</sub></sub> \[R(S, A) ∇<sub>θ</sub> log π<sub>θ</sub>(A|S)\]**

    This equation expresses the gradient in terms of an expectation over the state and action distributions.

### **Policy Gradients: Reduce Variance**

*   **E \[b∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\] = E\[∑ <sub>a</sub> π(a|S<sub>t</sub> )b∇<sub>θ</sub> log π(a|S<sub>t</sub> )\]**

    This equation expands the expectation to sum over all actions.

*   **E \[b∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\] = E\[ b∇<sub>θ</sub> ∑ <sub>a</sub> π(a|S<sub>t</sub> )\]**

    This step moves the baseline *b* outside the sum, as it doesn't depend on the action.

*   **E \[b∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\] = E \[b∇<sub>θ</sub>1\] = 0**

    This equation uses the fact that the sum of probabilities over all actions is 1, and the derivative of a constant is zero.

*   **θ <sub>t+1</sub> = θ <sub>t</sub> + α(R<sub>t+1</sub> − b(S<sub>t</sub> ))∇<sub>θ</sub> log π<sub>θ<sub>t</sub></sub> (A<sub>t</sub> |S<sub>t</sub> )**

    This equation is the updated policy parameter update rule with the baseline subtraction.

### **Example: Softmax Policy**

*   **π<sub>θ</sub>(a|s) = e<sup>h(s,a)</sup>/∑ <sub>b</sub> e<sup>h(s,b)</sup>**

    This equation represents the softmax policy, where the probability of an action is proportional to the exponentiated action preference *h(s, a)*.

*   **∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t</sub> |S<sub>t</sub> ) = ∇<sub>θ</sub>h(S<sub>t</sub>, A<sub>t</sub> ) − ∑ <sub>a</sub> π<sub>θ</sub>(a|S<sub>t</sub> )∇<sub>θ</sub>h(S<sub>t</sub>, a)**

    This equation calculates the gradient of the log probability for the softmax policy.

### **Policy Gradient Theorem (Episodic)**

*   **∇<sub>θ</sub>J(θ) = E<sub>π<sub>θ</sub></sub> \[ <sup>T</sup>∑ <sub>t=0</sub> γ<sup>t</sup>q<sub>π<sub>θ</sub></sub> (S<sub>t</sub>, A<sub>t</sub> )∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t</sub> |S<sub>t</sub> ) | S<sub>0</sub> ∼ d<sub>0</sub> \]**

    This equation gives the policy gradient theorem for episodic settings, where:

    *   *q<sub>π</sub>(s, a) = E<sub>π</sub>\[G<sub>t</sub> | S<sub>t</sub> = s, A<sub>t</sub> = a\] = E<sub>π</sub>\[R<sub>t+1</sub> + γq<sub>π</sub>(S<sub>t+1</sub>, A<sub>t+1</sub>) | S<sub>t</sub> = s, A<sub>t</sub> = a]*.

### **Episodic Policy Gradients: Proof**

*   **∇<sub>θ</sub>J<sub>θ</sub>(π) = ∇<sub>θ</sub>E \[G(τ)\] = E \[G(τ)∇<sub>θ</sub> log p(τ)\]**

    This equation uses the score function trick to express the gradient of the expected return.

*   **∇<sub>θ</sub> log p(τ) = ∇<sub>θ</sub> log \[ p(S<sub>0</sub>)π(A<sub>0</sub> |S<sub>0</sub>)p(S<sub>1</sub> |S<sub>0</sub>, A<sub>0</sub>)π(A<sub>1</sub> |S<sub>1</sub>) · · · \]**

    This equation expands the log probability of the trajectory.

*   **∇<sub>θ</sub> log p(τ) = ∇<sub>θ</sub>\[ log p(S<sub>0</sub>) + log π(A<sub>0</sub> |S<sub>0</sub>) + log p(S<sub>1</sub> |S<sub>0</sub>, A<sub>0</sub>) + log π(A<sub>1</sub> |S<sub>1</sub>) + · · · \]**

    This step applies the logarithm property to separate the terms.

*   **∇<sub>θ</sub> log p(τ) = ∇<sub>θ</sub> \[ log π(A<sub>0</sub> |S<sub>0</sub>) + log π(A<sub>1</sub> |S<sub>1</sub>) + · · · \]**

    Since the state transitions probabilities do not depend on the policy parameters, their gradients are zero.

*   **∇<sub>θ</sub>J<sub>θ</sub>(π) = E<sub>π</sub>\[G(τ)∇<sub>θ</sub> <sup>T</sup>∑ <sub>t=0</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This equation combines the previous results.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[G(τ) <sup>T</sup>∑ <sub>t=0</sub> ∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This step moves the gradient operator inside the sum.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ <sub>t=0</sub> G(τ)∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This step rearranges the terms.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ <sub>t=0</sub> ( <sup>T</sup>∑ <sub>k=0</sub> γ<sup>k</sup>R<sub>k+1</sub> ) ∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This equation substitutes the definition of the return *G(τ)*.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ <sub>t=0</sub> ( <sup>T</sup>∑ <sub>k=t</sub> γ<sup>k</sup>R<sub>k+1</sub> ) ∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This step adjusts the summation limits to account for the return from time *t* onwards.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ <sub>t=0</sub> ( γ<sup>t</sup> <sup>T</sup>∑ <sub>k=t</sub> γ<sup>k−t</sup>R<sub>k+1</sub> ) ∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This step factors out the discount factor *γ<sup>t</sup>*.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ ( γ<sup>t</sup>G<sub>t</sub> ) ∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This equation replaces the inner sum with the return from time *t*, *G<sub>t</sub>*.

*   **∇<sub>θ</sub> J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ γ<sup>t</sup>q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub> )∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This equation replaces the return with the action-value function *q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub>)*.

### **Episodic Policy Gradients Algorithm**

*   **∇<sub>θ</sub>J<sub>θ</sub>(π) = E<sub>π</sub>\[<sup>T</sup>∑ <sub>t=0</sub> γ<sup>t</sup>q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub> )∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )\]**

    This equation restates the policy gradient theorem for episodic settings.

*   **∆θ <sub>t</sub> = γ <sup>t</sup>G<sub>t</sub>∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> )**

    This equation defines the update for the policy parameters at each time step.

*   **E<sub>π</sub>\[ ∑ <sub>t</sub> ∆θ <sub>t</sub> ] = ∇<sub>θ</sub>J<sub>θ</sub>(π)**

    This equation shows that the expected sum of the individual updates is equal to the policy gradient.

### **Policy Gradient Theorem (Average Reward)**

*   **∇<sub>θ</sub>J(θ) = E<sub>π</sub>\[q<sub>π<sub>θ</sub></sub> (S<sub>t</sub> , A<sub>t</sub>)∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t</sub> |S<sub>t</sub>)\]**

    This equation gives the policy gradient theorem for average reward settings, where:

    *   *q<sub>π</sub>(s, a) = E<sub>π</sub>\[R<sub>t+1</sub> − ρ + q<sub>π</sub>(S<sub>t+1</sub>, A<sub>t+1</sub>) | S<sub>t</sub> = s, A<sub>t</sub> = a]*.
    *   *ρ = E<sub>π</sub>\[R<sub>t+1</sub>]*.
        *   Note: global average, not conditioned on state or action.
*   Expectation is over both states and actions.

*   **∇<sub>θ</sub>J(θ) = E<sub>π</sub>\[R<sub>t+1</sub> <sup>∞</sup>∑ <sub>n=0</sub> ∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t−n</sub> |S<sub>t−n</sub>)\]**

    This equation presents an alternative (but equivalent) form of the policy gradient theorem for average reward settings.
*   Expectation is over both states and actions.

### **Policy Gradients: Reduce Variance**

*   **∇<sub>θ</sub>J<sub>θ</sub>(π) = E\[∑ <sub>t=0</sub> γ<sup>t</sup> (q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub> ) − v<sub>π</sub>(S<sub>t</sub>))∇<sub>θ</sub> log π(A<sub>t</sub> |S<sub>t</sub> ) \]**

    This equation applies the baseline subtraction to the policy gradient theorem.

### **Actor-Critic**

*   **δ<sub>t</sub> = R<sub>t+1</sub> + γv<sub>w</sub>(S<sub>t+1</sub>) − v<sub>w</sub>(S<sub>t</sub> )** \[one-step TD-error, or advantage\]

    This equation calculates the one-step TD error or advantage.

*   **w ← w + β δ<sub>t</sub> ∇<sub>w</sub>v<sub>w</sub>(S<sub>t</sub> )** \[TD(0)\]

    This equation updates the critic parameters *w* using the TD(0) algorithm.

*   **θ ← θ + α δ<sub>t</sub> ∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t</sub> | S<sub>t</sub> )** \[Policy gradient update (ignoring γ<sup>t</sup> term)\]

    This equation updates the actor parameters *θ* using the policy gradient with the advantage as the weighting factor.

### **Increasing Robustness with Trust Regions**

*   **KL(π<sub>old</sub>‖π<sub>θ</sub>) = E\[∫ π<sub>old</sub>(a | S) log (π<sub>θ</sub>(a | S) / π<sub>old</sub>(a | S)) da ]** 

    This equation calculates the Kullback-Leibler (KL) divergence between the old policy and the current policy, where:
*   Expectation is over states.

### **Example: Gaussian Policy**

*   **A<sub>t</sub> ∼ N(µ<sub>θ</sub>(S<sub>t</sub> ), σ<sup>2</sup>)**

    This equation defines the Gaussian policy, where the action is sampled from a normal distribution with mean *µ<sub>θ</sub>(S<sub>t</sub> )* and variance *σ<sup>2</sup>*.

*   **∇<sub>θ</sub> log π<sub>θ</sub>(s, a) =( A<sub>t</sub> − µ<sub>θ</sub>(S<sub>t</sub> ) / σ<sup>2</sup>) ∇µ<sub>θ</sub>(s)**

    This equation calculates the gradient of the log probability for the Gaussian policy.

### **Example: Policy Gradient with Gaussian Policy**

*   **θ <sub>t+1</sub> = θ <sub>t</sub> + β(G<sub>t</sub> − v(S<sub>t</sub> ))∇<sub>θ</sub> log π<sub>θ</sub>(A<sub>t</sub> |S<sub>t</sub> )**

    This equation updates the policy parameters using the policy gradient with a Gaussian policy.

*   **θ <sub>t+1</sub> = θ <sub>t</sub> + β(G<sub>t</sub> − v(S<sub>t</sub> )) (A<sub>t</sub> − µ<sub>θ</sub>(S<sub>t</sub> ) / σ<sup>2</sup>) ∇µ<sub>θ</sub>(S<sub>t</sub> )**

    This equation substitutes the gradient of the Gaussian policy.

### **Gradient Ascent on Value**

*   **∆θ ∝ ∂Q<sub>π</sub>(s, a) / ∂θ**

    This equation updates the policy parameters by performing gradient ascent on the action-value function.

*   **∆θ ∝ ∂Q<sub>π</sub>(s, π<sub>θ</sub>(S<sub>t</sub> )) / ∂π<sub>θ</sub>(S<sub>t</sub> ) ⋅ ∂π<sub>θ</sub>(S<sub>t</sub> ) / ∂θ**

    This equation expands the gradient using the chain rule.

### **Continuous Actor-Critic Learning Automaton (Cacla)**

*   **a<sub>t</sub> = Actor<sub>θ</sub>(S<sub>t</sub> )** (get current (continuous) action proposal)
*   **A<sub>t</sub> ∼ π(·|S<sub>t</sub>, a<sub>t</sub> )** (e.g., A<sub>t</sub> ∼ N(a<sub>t</sub>, Σ)) (explore)
*   **δ<sub>t</sub> = R<sub>t+1</sub> + γv<sub>w</sub>(S<sub>t+1</sub>) − v<sub>w</sub>(S<sub>t</sub> )** (compute TD error)
*   Update v<sub>w</sub>(S<sub>t</sub> ) (e.g., using TD) (policy evaluation)
*   If δ<sub>t</sub> > 0, update Actor<sub>θ</sub>(S<sub>t</sub> ) towards A<sub>t</sub> (policy improvement)

    **θ <sub>t+1</sub> ← θ <sub>t</sub> + β(A<sub>t</sub> − a<sub>t</sub> )∇<sub>θ<sub>t</sub></sub>Actor<sub>θ<sub>t</sub></sub> (S<sub>t</sub> )**
*   If δ<sub>t</sub> ≤ 0, do not update Actor<sub>θ</sub>

    Note: update magnitude does not depend on the value magnitude.
    Note: don’t update ‘away’ from ‘bad’ actions.


# lecture 10



### The Bellman Optimality Operator

Given an MDP, M = 〈S,A, p, r , γ〉, let V ≡ VS be the space of bounded real-valued functions over S. The Bellman Expectation operator T ∗V : V → V is defined point-wise as:

**(T ∗V f )(s) = max<sub>a</sub> [ r(s, a) + γ Σ<sub>s′</sub> p(s ′|a, s)f (s ′) ], ∀f ∈ V (1)**

As a common convention, the index V is dropped, and T ∗ = T ∗V is used.

### The Bellman Expectation Operator

Given an MDP, M = 〈S,A, p, r , γ〉, let V ≡ VS be the space of bounded real-valued functions over S. For any policy π : S ×A →, the Bellman Expectation operator TπV : V → V is defined, point-wise,  as:

**(Tπ V f )(s) = Σ<sub>a</sub> π(s, a) [ r(s, a) + γ Σ<sub>s′</sub>p(s ′|a, s)f (s ′) ], ∀f ∈ V (2)**

### Value Iteration

**v<sub>k+1</sub> = T ∗v<sub>k</sub>**

As k →∞, v<sub>k</sub> →‖.‖<sub>∞</sub> v∗. This is a direct application of the Banach’s Fixed Point theorem.

### Approximate Value Iteration

**v<sub>k+1</sub> = AT ∗v<sub>k</sub>**  (v<sub>k+1</sub> ≈ T ∗v<sub>k</sub>)

The control policy is returned: **π<sub>k+1</sub> = Greedy(v<sub>k+1</sub>)**

### Approximate Value Iteration (q-value version)

**q<sub>k+1</sub> = AT ∗q<sub>k</sub>** (q<sub>k+1</sub> ≈ T ∗q<sub>k</sub>)

The control policy is returned:  **π<sub>k+1</sub> = Greedy(q<sub>k+1</sub>)**

### Performance of AVI

Consider an MDP. Let q<sub>k </sub>be the value function returned by AVI after k steps and let π<sub>k</sub> be its corresponding greedy policy, then:

**‖q∗ − q<sub>πn</sub>‖<sub>∞</sub> ≤ (2γ/(1− γ)<sup>2 </sup>) max<sub> 0≤k<n</sub> ‖T ∗q<sub>k</sub> −AT ∗q<sub>k</sub>‖<sub>∞</sub> + (2γ<sup>n+1</sup>/(1− γ)) ε<sub>0</sub>** 

where **ε<sub>0</sub> = ‖q∗ − q<sub>0</sub>‖<sub>∞</sub>**

and T ∗ is the optimal Bellman operator associated with this MDP.

### Implications of AVI Performance Equation

*   As n→∞, ⇒ 2γ<sup>n</sup>/(1− γ)→ 0
*   If q<sub>0</sub> = q∗, then ‖q∗ − q<sub>πn</sub>‖<sub>∞</sub> ≤ (2γ/(1− γ)<sup>2 </sup>) max<sub> 0≤k<n</sub> ‖T ∗q<sub>k</sub> −AT ∗q<sub>k</sub>‖<sub>∞ </sub>
*   In iteration 1: q<sub>1</sub> = AT ∗q<sub>0</sub> = Aq∗. In general ⇒ ‖q<sub>1</sub> − q<sub>0</sub>‖<sub>∞</sub> > 0.

### Projection operator in L<sub>∞</sub>

Consider a hypothesis space F. If A = Π<sub>∞</sub> is the projection operator in L<sub>∞</sub>, then:

**Π<sub>∞</sub>g := arg inf<sub>f∈F</sub> ‖g − f ‖<sub>∞</sub>**

**q<sub>k+1</sub> = Π<sub>∞</sub>T ∗q<sub>k</sub> = arg inf<sub>f∈F</sub> ‖T ∗q<sub>k</sub> − f ‖<sub>∞</sub>**

Note that AT ∗ = Π<sub>∞</sub>T ∗ is a contraction operator in L<sub>∞</sub>. The algorithm converges for its fixed point: **f = Π<sub>∞</sub>T ∗f** If q∗ ∈ F, the above will converge to q∗.

### Fitted Q-iteration with Linear Approximation

Let q<sub>k+1</sub> = Π<sub>∞</sub>T ∗q<sub>k</sub> = arg inf<sub>f ∈F</sub> ‖T ∗q<sub>k</sub> − f ‖<sub>∞</sub>

Consider a linear hypothesis space **F<sub>φ</sub> = {q<sub>w</sub> (s, a) = w<sup>T</sup>φ(s, a)|∀w ∈ B}**. Then:

**q<sub>k+1</sub> = arg inf<sub>f ∈Fφ</sub> ‖T ∗q<sub>k</sub> − f ‖<sub>∞</sub> (12)**

⇔ **w<sub>k+1</sub> = arg inf<sub>w∈B</sub> ‖T ∗(w<sup>T</sup><sub>k </sub>φ)− w<sup>T</sup>φ‖<sub>∞</sub> (13)**

### Fitted Q-iteration with Linear Approximation Proposals

To address the problems of L<sub>∞</sub> minimization being hard to carry out efficiently and T ∗ typically being unknown and approximated, the following are proposed:

*   L<sub>∞</sub> → L<sub>2</sub>, with respect to a probability distribution µ over S ×A.  **q<sub>k+1</sub> = arg inf<sub>f ∈F</sub> ‖T ∗q<sub>k</sub> − f ‖<sup>2</sup><sub>µ</sub>**
*   Sampling to approximate T ∗. Sample (S<sub>t</sub> ,A<sub>t</sub> ,R<sub>t+1</sub>,S<sub>t+1</sub>) ∼ µ,P. Approximate T ∗q<sub>k</sub>(S<sub>t</sub> ,A<sub>t</sub>) by **Y<sub>t</sub> = R<sub>t+1</sub> + γmax<sub> a</sub> q<sub>k</sub>(S<sub>t+1</sub>, a) := T̃ ∗q<sub>k</sub>**

Every iteration k:

**q<sub>k+1</sub> = arg min <sub>qw∈F</sub> (1/n<sub>samples</sub>) Σ<sup>n<sub>samples</sub></sup> <sub>i=1</sub> (Y<sub>t</sub> − q<sub>w</sub> (S<sub>t</sub> ,A<sub>t</sub>))<sup>2</sup>**

### Fitted Q-iteration with other Approximations

Every iteration k + 1:

**q<sub>k+1</sub> = arg min <sub>qθ∈F</sub> (1/n<sub>samples</sub>) Σ<sup>n<sub>samples</sub></sup> <sub>i=1 </sub>(Y<sub>t</sub> − q<sub>θ</sub>(S<sub>t</sub> ,A<sub>t</sub>))<sup>2 </sup> (14)**

= **arg min <sub>qθ∈F</sub> (1/n<sub>samples</sub>) Σ<sup>n<sub>samples</sub></sup> <sub>i=1 </sub>(T̃ ∗q<sub>k</sub>(S<sub>t</sub> ,A<sub>t</sub>)− q<sub>θ</sub>(S<sub>t</sub> ,A<sub>t</sub>))<sup>2</sup> (15)**

### Temporal difference error

**δ<sub>t</sub> = R<sub>t+1</sub> + γq<sub>wt</sub> (S<sub>t+1</sub>, π(S<sub>t+1</sub>))− q<sub>wt</sub> (S<sub>t</sub> ,A<sub>t</sub>) (37)**

### Parameters update

**w<sub>t+1</sub> = w<sub>t</sub> + α<sub>t</sub>δ<sub>t</sub>φ(s<sub>t</sub> , a<sub>t</sub>)**

### TD(λ) with Linear Approximation

**‖q<sub>w∗</sub> − q<sub>π</sub>‖<sup>2</sup><sub>,µπ</sub> ≤ ((1− λγ)/(1− γ)) inf<sub>w</sub> ‖q<sub>w</sub> − q<sub>π</sub>‖<sup>2</sup><sub>,µπ</sub> (38)**

# lecture 11



### Multi-step Off-Policy Learning 
The document introduces the concept of **importance sampling corrections** used in multi-step off-policy learning to account for the difference between the behavior policy (data generating policy) and the target policy (policy being learned).

Given a trajectory $\tau_t = \{S_t, A_t, R_{t+1},...,S_T\}$, the importance sampling ratio is calculated as follows:
$$
\begin{aligned}
\hat{G}_t &\equiv \frac{p(\tau_t|\pi)}{p(\tau_t|\mu)} \\
&= \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} \cdot ... \cdot \frac{\pi(A_T|S_T)}{\mu(A_T|S_T)}G_t
\end{aligned}
$$
This ensures that the expected return under the behavior policy is equivalent to the expected return under the target policy: $E[\hat{G}_t|\mu] = E[G_t|\pi]$.

### Issues with Off-Policy Learning
The document highlights two primary issues with off-policy learning:

*   **High variance**, especially with multi-step updates 
*   **Divergent and inefficient learning**, particularly with one-step updates 

### Mitigating Variance in Off-Policy Learning

#### Per-Decision Importance Weighting 
To mitigate high variance, the document suggests per-decision importance weighting. It leverages the fact that the expectation of a random variable $X$ that doesn't correlate with the action $A$ remains the same under both the target and behavior policies.

Mathematically:
$$
\begin{aligned}
E[X|\pi] &= E[\frac{\pi(A|s)}{\mu(A|s)}X|\mu] \\
&= E[X|\mu]
\end{aligned}
$$
This implies that for variables uncorrelated with the policy, no importance sampling correction is necessary.

#### Control Variates
Control variates are introduced as a method to reduce variance in multi-step return estimations. This involves incorporating terms that don't change the expected value but reduce the variance. The document extends this idea to multi-step returns by adding per-decision importance weights.

#### Adaptive Bootstrapping 
Adaptive bootstrapping is proposed as a mechanism to balance the trade-off between high variance from limited bootstrapping and the risk of divergence from excessive bootstrapping. This is achieved by adaptively adjusting the bootstrapping parameter $\lambda$ based on the degree of off-policyness.

One approach, known as ABTD or v-trace, sets $\lambda_t$ to the minimum of 1 and the inverse of the importance sampling ratio $\rho_t$:

$\lambda_t$= $min(1, 1/\rho_t)$.

This effectively truncates the error sum when the policies significantly diverge, thus controlling variance. 

### Tree Backup 
As an alternative to adaptive bootstrapping with $\lambda_t$, the document presents the tree backup method. This approach focuses on selectively updating the value of the actually selected action while using the current estimate for other actions. This method offers low variance and unbiased updates but might lead to premature bootstrapping, potentially triggering issues associated with the deadly triad. 

# lecture 12


**Recap: Value function approximation**

*   Goal: find  $\theta$ that minimizes the difference between $v^π$ and $v_θ$
    
    $L(θ) = E_{S∼d} [(v^π(S) − v_θ(S))^2]$ 
    
    Where $d$ is the state visitation distribution induced by $π$ and the dynamics $p$.
    
*   Solution: use gradient descent to iteratively minimize this objective
    
    $Δθ = -\frac{1}{2}α∇_θL(θ) = αE_{S∼d} [(v^π(S) − v_θ(S)∇_θv_θ(S)]$
    
*   Problem: evaluating the expectation is going to be hard in general
    
*   Solution: use stochastic gradient descent, i.e. sample the gradient update
    
    $Δθ = α(G_t − v_θ(S_t))∇_θv_θ(S_t)$
    
    where $G_t$ is a suitable sampled estimate of the return
    
    *   Monte Carlo Prediction →  $G_t = R_t + γR_{t+1} + γ^2R_{t+2} + ...$
        
    *   TD Prediction → $G_t = R_t + γv_θ(S_{t+1})$
        

**Deep value function approximation**

*   Parameterize $v_θ$ using a deep neural network, for instance as a multilayer perceptron:
    
    $v_θ(S) = W_2 \tanh(W_1 ∗ S + b_1) + b_2$
    
    where $θ = \{W_1, b_1, W_2, b_2\}$
    
    When $v_θ$ was linear $∇v_θ$ was trivial to compute. 

**Deep Q-learning**

*   Update parameters $θ$ through the stochastic update:
    
    $Δθ = α(G_t − q_θ(S_t ,A_t))∇_θq_θ(S_t , A_t),  G_t = R_{t+1} + γ \max_a q_θ(S_{t+1}, a)$
    
*   For consistency with DL notation you may write this as gradient of a pseudo-loss:
    
    $L(θ) = \frac{1}{2} (R_{t+1} + γ\max_a q_θ(S_{t+1}, a) − q_θ(S_t ,A_t))^2$
    
    *   Note: we ignore the dependency of the bootstrap target on θ
        
    *   Note: this is not a true loss!
        

**Deep double Q-learning (van Hasselt et al. 2016)**

*   Q-learning has an overestimation bias, that can be corrected by double Q-learning
    
    $L(θ) = \frac{1}{2} ( R_{i+1} + γ q_{θ−}(S_{i+1}, \argmax_a q_θ(S_{i+1}, a)) − q_θ(S_i ,A_i ))^2$
    
    *   Great combination with target networks: we can use the frozen params as $θ_−$.
        

**Prioritized replay (Schaul et al. 2016)**

*   DQN samples uniformly from replay
    
*   Idea: prioritize transitions on which we can learn much
    
*   Basic implementation: priority of sample $i = |δ_i |$, where $δ_i$ was the TD error on the last this transition was sampled
    

**Multi-step control**

*   Define the n-step Q-learning target
    
    ${G_t}^{(n)} = R_{t+1} + γR_{t+2} + ...+ γ^{n−1}R_{t+n} + γ^n q_{θ−}(S_{i+1}, \argmax_a q_θ(S_{i+1}, a))$ 
    
*   Multi-step deep Q-learning
    
    $Δθ = α({G_t}^{(n)} − q_θ(S_t ,A_t))∇_θq_θ(S_t ,A_t)$
    

**Dueling networks (Wang et al. 2016)**

*   We can decompose $q_θ(s, a) = v_ξ(s) + A_χ(s, a)$, where $θ = ξ ∪ χ$ 
    
    *   Here $A_χ(s, a)$ is the advantage for taking action a
        



# lecture 13

**General Value Functions**

*   A GVF is conditioned on more than just state and actions:
    
    $q_{c,γ,π}(s, a) = E[C_{t+1} + γ_{t+1}C_{t+2} + γ_{t+1}γ_{t+2}C_{t+3} + ... | S_t = s, A_{t+i} \sim π(S_{t+i})]$
    
    where  $C_t = c(S_t)$ and $γ_t = γ(S_t)$ where $S_t$ could be the environment state
    
    *   $c: S → R$ is the cumulant
        
    *   $γ: S → R$ is the discount or termination
        
    *   $π: S → A$ is the target policy
        

**Adaptive target normalization (van Hasselt et al. 2016)**

1.  Observe target, e.g., $T_{t+1} = R_{t+1} + γ \max_a q_θ(S_{t+1}, a)$
    
2.  Update normalization parameters:
    
    *   $µ_{t+1} = µ_t + η(T_{t+1} − µ_t)$ (first moment / mean)
        
    *   $ν_{t+1} = ν_t + η(T^2_{t+1} − ν_t)$ (second moment)
        
    *   $σ_{t+1} = \sqrt{ν_t − µ^2_t}$ (variance)
        
        where η is a step size (e.g., η = 0.001)
        
3.  Network outputs $\tilde{q}_θ(s, a)$, update with
    
    $Δθ_t ∝ (\frac{T_{t+1} − µ_{t+1}}{σ_{t+1}} − \tilde{q}_θ(S_t,A_t))∇_θ\tilde{q}_θ(S_t,A_t)$
    
4.  Recover unnormalized value: $q_θ(s, a) = σ_t \tilde{q}_θ(s, a) + µ_t$ (used for bootstrapping)
    

**Preserve outputs**

*   Every update to the normalization changes all outputs.
*   To avoid this, typically:

$\tilde{q}_{W,b,θ}(s) = Wϕ_θ(s) + b$.

*   Idea: define

$W_t' = \frac{σ_t}{σ_{t+1}}W$

$b_t' = \frac{σ_tb_t + µ_t - µ_{t+1}}{σ_{t+1}}$

Then $σ_{t+1}\tilde{q}_{W_t',b_t',θ_t}(s) + µ_{t+1} = σ_t\tilde{q}_{W_t,b_t,θ_t}(s) + µ_t$

*   Then update $W_t'$, $b_t'$ and $θ_t$ as normal (e.g., SGD)

**Categorical Return Distributions (Bellemare et al, 2017)**

*   Consider a fixed ‘comb’ distribution on $z = (−10,−9.9, . . . , 9.9, 10)$
    
*   For each point of support, we assign a ‘probability’ $p_{iθ}(S_t, A_t)$
    
*   The approximate distribution of the return s and a is the tuple $(z, p_θ(s, a))$
    
*   Our estimate of the expectation is: $z>p_θ(s, a) ≈ q(s, a)$ – use this to act
    

**Categorical Return Distributions**

1.  Find max action: $a^* = \argmax_a z>p_θ(S_{t+1}, a)$ (use, e.g., $θ_−$ for double Q)
    
2.  Update support:
    
    $z' = R_{t+1} + γz$
    
3.  Project distribution $(z', p_θ(S_{t+1}, a^*))$ onto support z
    
    $d' = (z, p') = Π(z', p_θ(S_{t+1}, a^*))$ 
    
    where Π denotes projection
    
4.  Minimize divergence
    
    $KL(d'||d) = -Σ_ip_i' \frac{\log p_i'}{\log p_{iθ}(S_t, A_t)}$
    
    Bottom-right: target distribution $Π(R_{t+1} + γz, p_θ(S_{t+1}, a^*))$ 
    
    Update $p_θ(S_t, A_t)$ towards this
    

