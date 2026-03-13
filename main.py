from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.ntn_env import NTNEnv
import config
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Custom Callback for Live Plot
# ------------------------------
class LivePlotCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_title("Live PPO Training - NTN Resource Allocation")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        plt.ion()   # Interactive mode ON

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]["r"]
            self.rewards.append(reward)

            self.line.set_xdata(np.arange(len(self.rewards)))
            self.line.set_ydata(self.rewards)

            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        return True


# ------------------------------
# Environment Setup
# ------------------------------
env = Monitor(NTNEnv())

model = PPO("MlpPolicy", env, verbose=0)

print("🔥 Live Training Started...")

callback = LivePlotCallback()
model.learn(total_timesteps=config.EPISODES * config.STEPS_PER_EPISODE,
            callback=callback)

print("✅ Training Finished")

plt.ioff()
plt.show()
# ------------------------------
# Inspect Learned Policy
# ------------------------------

print("\nInspecting learned power allocation:\n")

obs, _ = env.reset()

for i in range(5):
    action, _ = model.predict(obs, deterministic=True)

    print(f"Channel Gains: {obs}")
    print(f"AI Power Allocation: {action}\n")

    obs, _, terminated, truncated, _ = env.step(action)
    # ------------------------------
# Baseline Comparison
# ------------------------------
import numpy as np

def evaluate_policy(env, model=None, episodes=50):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards)

print("\nEvaluating policies...")

ppo_reward = evaluate_policy(env, model=model)
random_reward = evaluate_policy(env, model=None)

print(f"Average PPO Reward: {ppo_reward}")
print(f"Average Random Reward: {random_reward}")