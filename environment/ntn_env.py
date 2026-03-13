import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
import pandas as pd


class NTNEnv(gym.Env):

    def __init__(self):
        super(NTNEnv, self).__init__()

        # -----------------------------
        # Action space (power allocation per tower)
        # -----------------------------
        self.action_space = spaces.Box(
            low=0.0,
            high=config.MAX_POWER,
            shape=(config.NUM_CHANNELS,),
            dtype=np.float32
        )

        # -----------------------------
        # Observation space
        # -----------------------------
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.NUM_CHANNELS,),
            dtype=np.float32
        )

        # Episode control
        self.current_step = 0
        self.max_steps = config.STEPS_PER_EPISODE

        # -----------------------------
        # Satellite position
        # -----------------------------
        self.sat_x = 0.0
        self.sat_y = 8.0

        # -----------------------------
        # Ground tower positions
        # -----------------------------
        self.ground_positions = np.array([
            [2, 1],
            [4, 1],
            [6, 1],
            [8, 1],
            [5, 1]
        ])

        # -----------------------------
        # Load dataset (12000 samples)
        # -----------------------------
        self.dataset = pd.read_csv("ntn_dataset.csv")
        self.dataset_index = 0

        self.state = None

    # --------------------------------------------------
    # RESET FUNCTION
    # --------------------------------------------------
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.sat_x = 0.0

        # Pick random dataset sample
        self.dataset_index = np.random.randint(len(self.dataset))

        row = self.dataset.iloc[self.dataset_index]

        # Create state from dataset
        self.state = np.array([
            row["channel_gain"],
            row["sinr"],
            row["bandwidth"],
            row["power"],
            row["delay"]
        ], dtype=np.float32)

        return self.state, {}

    # --------------------------------------------------
    # STEP FUNCTION
    # --------------------------------------------------
    def step(self, action):

        # Move satellite
        self.sat_x += 0.1
        if self.sat_x > 10:
            self.sat_x = 0

        # Move dataset index
        self.dataset_index += 1
        if self.dataset_index >= len(self.dataset):
            self.dataset_index = 0

        row = self.dataset.iloc[self.dataset_index]

        channel_gain = row["channel_gain"]

        self.state = np.array([
            row["channel_gain"],
            row["sinr"],
            row["bandwidth"],
            row["power"],
            row["delay"]
        ], dtype=np.float32)

        # -----------------------------
        # Compute SNR
        # -----------------------------
        snr = (action * channel_gain) / config.NOISE_POWER

        # -----------------------------
        # Throughput (Shannon capacity)
        # -----------------------------
        throughput = np.sum(np.log2(1 + snr))

        # -----------------------------
        # Energy penalty
        # -----------------------------
        energy_penalty = np.sum(action)

        # -----------------------------
        # Reward
        # -----------------------------
        reward = throughput - 0.02 * energy_penalty

        # -----------------------------
        # Episode termination
        # -----------------------------
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}
