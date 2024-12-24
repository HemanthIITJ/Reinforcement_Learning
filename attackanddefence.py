import gymnasium as gym
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import imageio

class AttackDefenseEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.attacks = [
            {"id": "brute_force", "potential_value": 5, "emoji": "💣"},
            {"id": "sql_injection", "potential_value": 8, "emoji": "💉"},
            {"id": "ddos", "potential_value": 7, "emoji": "💥"}
        ]
        self.defenses = [
            {"id": "firewall", "potential_value": 3, "emoji": "🛡️"},
            {"id": "antivirus", "potential_value": 2, "emoji": "🦠❌"},
            {"id": "rate_limiting", "potential_value": 1, "emoji": "🚦"}
        ]
        self.action_space = gym.spaces.Discrete(len(self.defenses))
        self.observation_space = gym.spaces.Discrete(len(self.attacks))
        self.current_attack = None
        self.render_mode = render_mode
        self.mapping_attack_to_index = {attack['id']: i for i, attack in enumerate(self.attacks)}

    def step(self, action):
        defense = self.defenses[action]
        reward = self.current_attack["potential_value"] - defense["potential_value"]
        terminated = True
        truncated = False
        info = {}
        return self.mapping_attack_to_index[self.current_attack['id']], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_attack = random.choice(self.attacks)
        return self.mapping_attack_to_index[self.current_attack['id']], {}

    def render(self):
        if self.render_mode == 'rgb_array':
            attack_index = self.mapping_attack_to_index[self.current_attack['id']]
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :50, :] = [255, 0, 0] if attack_index % 2 == 0 else [0, 255, 0]
            image[:, 50:, :] = [0, 0, 255] if attack_index % 2 == 1 else [255, 255, 0]
            return image

    def close(self):
        pass

class PlottingCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.rewards = []
        self.losses = []
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if 'rewards' in self.locals:
          self.rewards.extend(self.locals['rewards'])
        return True

    def _on_rollout_end(self) -> None:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "loss" in info:
                    self.losses.append(info["loss"])

    def _on_training_end(self) -> None:
        plt.figure()
        plt.plot(self.rewards)
        plt.title("Rewards")
        plt.savefig(f"{self.save_path}/rewards.png")
        plt.close()
        if self.losses:
            plt.figure()
            plt.plot(self.losses)
            plt.title("Losses")
            plt.savefig(f"{self.save_path}/losses.png")
            plt.close()
        np.save(f"{self.save_path}/rewards.npy", self.rewards)
        if self.losses:
           np.save(f"{self.save_path}/losses.npy", self.losses)

env = DummyVecEnv([lambda: AttackDefenseEnv()])
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_tensorboard/", verbose=1)
callback = PlottingCallback(save_path="./train_results")

model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
model.save("./ppo_attack_defense")

model = PPO.load("./ppo_attack_defense")
test_callback = PlottingCallback(save_path="./test_results")
obs = env.reset()
all_images = []
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated= env.step(action)
    all_images.append(env.render())
    test_callback.locals['rewards'] = [reward]
    # test_callback.on_step()
    if terminated or truncated:
        obs= env.reset()
test_callback._on_training_end()

import imageio
imageio.mimsave("./inference.gif", all_images, duration=100)