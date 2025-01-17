import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence as torch_kl_div
from typing import Any, Callable, List, Tuple, Optional
import numpy as np
import gym
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def initialize_weights(m: nn.Module) -> None:
    """
    Initialize weights of linear layers using Xavier uniform initialization.

    Args:
        m (nn.Module): The module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class Actor(nn.Module):
    """
    Actor network for policy approximation.
    Accepts any PyTorch model architecture.
    """

    def __init__(self, base_model: nn.Module, action_space: gym.Space) -> None:
        """
        Initialize the Actor.

        Args:
            base_model (nn.Module): The base neural network model.
            action_space (gym.Space): The action space of the environment.
        """
        super(Actor, self).__init__()
        self.base = base_model
        if isinstance(action_space, gym.spaces.Discrete):
            # Use out_features instead of in_features
            if hasattr(self.base, 'fc') and hasattr(self.base.fc, 'out_features'):
                self.action_head = nn.Linear(self.base.fc.out_features, action_space.n)
            else:
                raise AttributeError("The base model must have 'fc.out_features' for Discrete action spaces.")
        else:
            raise NotImplementedError("Only Discrete action spaces are supported.")

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> Categorical:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Categorical: Action distribution.
        """
        x = self.base(x)
        logits = self.action_head(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """
    Critic network for value function approximation.
    Accepts any PyTorch model architecture.
    """

    def __init__(self, base_model: nn.Module) -> None:
        """
        Initialize the Critic.

        Args:
            base_model (nn.Module): The base neural network model.
        """
        super(Critic, self).__init__()
        self.base = base_model
        if hasattr(self.base, 'fc') and hasattr(self.base.fc, 'out_features'):
            self.value_head = nn.Linear(self.base.fc.out_features, 1)
        else:
            raise AttributeError("The base model must have 'fc.out_features' for Critic.")

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: State value.
        """
        x = self.base(x)
        value = self.value_head(x)
        return value


def conjugate_gradient(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    cg_iters: int = 10,
    residual_tol: float = 1e-10
) -> torch.Tensor:
    """
    Conjugate gradient algorithm to solve Ax = b.

    Args:
        A (Callable[[torch.Tensor], torch.Tensor]): Function to compute the matrix-vector product.
        b (torch.Tensor): Right-hand side vector.
        cg_iters (int, optional): Maximum number of iterations. Defaults to 10.
        residual_tol (float, optional): Tolerance for convergence. Defaults to 1e-10.

    Returns:
        torch.Tensor: Solution vector x.
    """
    x = torch.zeros_like(b, device=b.device)
    r = b.clone()
    p = b.clone()
    rsold = torch.dot(r, r)

    for i in range(cg_iters):
        Ap = A(p)
        pAp = torch.dot(p, Ap)
        if pAp == 0:
            print("Early termination of CG: pAp == 0")
            break  # Prevent division by zero
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if rsnew < residual_tol:
            print(f"Conjugate Gradient converged in {i+1} iterations.")
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def fisher_vector_product(
    kl: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    vector: torch.Tensor,
    damping: float
) -> torch.Tensor:
    """
    Compute the Fisher-vector product.

    Args:
        kl (Callable[[], torch.Tensor]): KL divergence function.
        params (List[torch.Tensor]): List of model parameters.
        vector (torch.Tensor): Vector to multiply with the Fisher Information Matrix.
        damping (float): Damping factor.

    Returns:
        torch.Tensor: Resulting tensor.
    """
    kl_loss = kl()
    kl_grad = torch.autograd.grad(kl_loss, params, create_graph=True)
    flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
    kl_v = (flat_kl_grad * vector).sum()
    kl_grad2 = torch.autograd.grad(kl_v, params, retain_graph=True)
    flat_kl_grad2 = torch.cat([g.contiguous().view(-1) for g in kl_grad2])
    return flat_kl_grad2 + damping * vector


class TRPOAgent:
    """
    Trust Region Policy Optimization (TRPO) agent.
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        env: gym.Env,
        device: torch.device,
        max_kl: float = 1e-2,
        damping: float = 0.1,  # Increased damping for stability
        gamma: float = 0.99,
        lam: float = 0.95,
        cg_iters: int = 10,
        cg_tol: float = 1e-10,
        critic_lr: float = 1e-3,
        max_step_size: float = 0.1,  # Reduced step size for more controlled updates
        batch_size: int = 5  # Number of episodes per update
    ) -> None:
        """
        Initialize the TRPO agent.

        Args:
            actor (Actor): The actor network.
            critic (Critic): The critic network.
            env (gym.Env): The environment.
            device (torch.device): The device to run computations on.
            max_kl (float, optional): Maximum KL divergence. Defaults to 1e-2.
            damping (float, optional): Damping factor for Fisher vector product. Defaults to 0.1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lam (float, optional): GAE lambda. Defaults to 0.95.
            cg_iters (int, optional): Conjugate gradient iterations. Defaults to 10.
            cg_tol (float, optional): Conjugate gradient tolerance. Defaults to 1e-10.
            critic_lr (float, optional): Learning rate for critic optimizer. Defaults to 1e-3.
            max_step_size (float, optional): Maximum step size for parameter updates. Defaults to 0.1.
            batch_size (int, optional): Number of episodes to collect before an update. Defaults to 5.
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.env = env
        self.device = device
        self.max_kl = max_kl
        self.damping = damping
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.max_step_size = max_step_size
        self.batch_size = batch_size

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            Tuple[int, float]: The selected action and its log probability.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_advantages(
        self,
        rewards: List[float],
        masks: List[float],
        values: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (List[float]): List of rewards.
            masks (List[float]): List indicating if the episode is done.
            values (List[float]): List of state values.

        Returns:
            Tuple[List[float], List[float]]: Advantages and returns.
        """
        advantages = []
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return advantages, returns

    def update(self, trajectories: List[dict]) -> None:
        """
        Update the policy using TRPO.

        Args:
            trajectories (List[dict]): Collected trajectories.
        """
        # Flatten all trajectories
        all_steps = [t for traj in trajectories for t in traj["steps"]]
        states = torch.tensor(
            np.array([t["state"] for t in all_steps]),
            dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            [t["action"] for t in all_steps],
            dtype=torch.int64
        ).to(self.device)
        rewards = [t["reward"] for t in all_steps]
        masks = [1 - t["done"] for t in all_steps]

        # Compute values with bootstrap
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy().tolist()
            last_state = all_steps[-1]["state"]
            last_state_tensor = torch.from_numpy(np.array([last_state], dtype=np.float32)).to(self.device)
            last_value = self.critic(last_state_tensor).squeeze().item()
            values.append(last_value)  # bootstrap value

        advantages, returns = self.compute_advantages(rewards, masks, values)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Old policy
        with torch.no_grad():
            old_dist = self.actor(states)
            old_log_probs = old_dist.log_prob(actions)

        def compute_kl() -> torch.Tensor:
            new_dist = self.actor(states)
            kl = torch_kl_div(old_dist, new_dist).mean()
            return kl

        # Objective
        def get_loss() -> torch.Tensor:
            dist = self.actor(states)
            ratio = torch.exp(dist.log_prob(actions) - old_log_probs)
            return -(ratio * advantages).mean()

        loss = get_loss()

        # Compute gradient
        grads = torch.autograd.grad(loss, self.actor.parameters(), retain_graph=True)
        grad = torch.cat([g.view(-1) for g in grads]).data

        # Gradient Clipping
        grad_norm = torch.norm(grad)
        max_grad_norm = 0.5  # Adjust this value as needed
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (grad_norm + 1e-8))

        # Compute step direction
        def fisher_vector_product_closure(v: torch.Tensor) -> torch.Tensor:
            return fisher_vector_product(compute_kl, list(self.actor.parameters()), v, self.damping)

        step_dir = conjugate_gradient(fisher_vector_product_closure, grad, self.cg_iters, self.cg_tol)

        # Check for NaNs in step_dir
        if torch.isnan(step_dir).any():
            print("Conjugate gradient produced NaN. Skipping this update.")
            return

        shs = 0.5 * (step_dir * fisher_vector_product_closure(step_dir)).sum()
        if shs <= 0:
            print("SHS is non-positive. Skipping this update.")
            return
        step_size = torch.sqrt(2 * self.max_kl / (shs + 1e-10))
        step_size = min(step_size.item(), self.max_step_size)  # Limit step_size to prevent large updates
        full_step = step_dir * step_size

        # Save old parameters
        old_params = torch.cat([p.data.view(-1) for p in self.actor.parameters()])

        # Perform line search
        def set_params(new_params: torch.Tensor) -> None:
            pointer = 0
            for p in self.actor.parameters():
                num = p.numel()
                p.data.copy_(new_params[pointer:pointer + num].view_as(p))
                pointer += num

        success = False
        for iteration in range(10):
            new_params = old_params + full_step
            set_params(new_params)
            new_loss = get_loss()
            kl = compute_kl()

            # Check for NaNs
            if torch.isnan(new_loss) or torch.isnan(kl):
                print(f"NaN detected in loss or KL during line search at iteration {iteration}.")
                break

            if new_loss < loss and kl < self.max_kl:
                success = True
                print(f"Line search succeeded at iteration {iteration}.")
                break
            else:
                full_step = full_step * 0.5  # Reduce step size
                print(f"Line search iteration {iteration}: Reducing step size.")

        if not success:
            print("Line search failed. Restoring old parameters.")
            set_params(old_params)
        else:
            # Check if any parameter is NaN
            nan_detected = False
            for p in self.actor.parameters():
                if torch.isnan(p).any():
                    nan_detected = True
                    print("NaN detected in actor parameters after update. Restoring old parameters.")
                    set_params(old_params)
                    break

            if nan_detected:
                return

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_predictions = self.critic(states).squeeze()
        critic_loss = F.mse_loss(critic_predictions, returns)
        critic_loss.backward()
        # Gradient Clipping for Critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()


# def main() -> None:
#     """
#     Main function to train the TRPO agent.
#     """
#     try:
#         env = gym.make("CartPole-v1")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Use a simple MLP for the actor and critic
#         class SimpleModel(nn.Module):
#             def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
#                 super(SimpleModel, self).__init__()
#                 self.fc = nn.Linear(input_dim, hidden_dim)
#                 self.relu = nn.ReLU()

#             def forward(self, x: torch.Tensor) -> torch.Tensor:
#                 return self.relu(self.fc(x))

#         input_dim = env.observation_space.shape[0]
#         action_space = env.action_space

#         base_actor = SimpleModel(input_dim)
#         base_actor.apply(initialize_weights)
#         actor = Actor(base_actor, action_space)

#         base_critic = SimpleModel(input_dim)
#         base_critic.apply(initialize_weights)
#         critic = Critic(base_critic)

#         agent = TRPOAgent(actor, critic, env, device, batch_size=5)  # Collect 5 episodes per update

#         num_episodes = 1000
#         batch_trajectories = []
#         for episode in range(1, num_episodes + 1):
#             # Handle env.reset() for different Gym versions
#             reset_output = env.reset()
#             if isinstance(reset_output, tuple):
#                 state, _ = reset_output
#             else:
#                 state = reset_output

#             trajectories = []
#             done = False
#             total_reward = 0.0
#             while not done:
#                 action, _ = agent.select_action(state)
#                 step_output = env.step(action)

#                 # Handle env.step() for different Gym versions
#                 if isinstance(step_output, tuple) and len(step_output) == 5:
#                     next_state, reward, terminated, truncated, _ = step_output
#                     done = terminated or truncated
#                 elif isinstance(step_output, tuple) and len(step_output) == 4:
#                     next_state, reward, done, _ = step_output
#                 else:
#                     raise ValueError("Unexpected number of outputs from env.step()")

#                 trajectories.append({
#                     "state": state,
#                     "action": action,
#                     "reward": reward,
#                     "done": done
#                 })
#                 state = next_state
#                 total_reward += reward

#             batch_trajectories.append({"steps": trajectories})

#             # Perform update every 'batch_size' episodes
#             if episode % agent.batch_size == 0:
#                 agent.update(batch_trajectories)
#                 batch_trajectories = []

#             # Logging
#             print(f"Episode {episode}: Total Reward: {total_reward}")

#     except Exception as e:
#         print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()

def main() -> None:
    """
    Main function to train the TRPO agent.
    """
    
    
    
    gym_version = gym.__version__
    major_version = int(gym_version.split('.')[0])
    minor_version = int(gym_version.split('.')[1])
    if major_version > 0 or (major_version == 0 and minor_version >= 26):
        render_mode = 'human'
    else:
        render_mode = None
        
        # Choose environment
    env_name = 'MountainCar-v0'  # You can change this to other environments like 'MountainCar-v0', 'Acrobot-v1', etc.
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a simple MLP for the actor and critic
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.fc(x))

    input_dim = env.observation_space.shape[0]
    action_space = env.action_space

    base_actor = SimpleModel(input_dim)
    actor = Actor(base_actor, action_space)

    base_critic = SimpleModel(input_dim)
    critic = Critic(base_critic)

    agent = TRPOAgent(actor, critic, env, device)

    num_episodes = 100
    for episode in range(num_episodes):
        # Handle env.reset() for different Gym versions
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output

        trajectories = []
        done = False
        total_reward = 0.0
        while not done:
            action, log_prob = agent.select_action(state)
            step_output = env.step(action)

            # Handle env.step() for different Gym versions
            if isinstance(step_output, tuple) and len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            elif isinstance(step_output, tuple) and len(step_output) == 4:
                next_state, reward, done, _ = step_output
            else:
                raise ValueError("Unexpected number of outputs from env.step()")

            trajectories.append({
                "state": state,
                "action": action,
                "reward": reward,
                "done": done
            })
            state = next_state
            total_reward += reward

        agent.update([{"steps": trajectories}])
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    


if __name__ == "__main__":
    main()