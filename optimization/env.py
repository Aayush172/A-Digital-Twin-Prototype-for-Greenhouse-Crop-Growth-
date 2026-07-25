"""
Gymnasium-compatible Reinforcement Learning Environment for Greenhouse Climate Control.

Agents learn to manipulate heating, ventilation, humidification, and lighting systems
to maximize plant growth while minimizing energy resource consumption.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any
from digital_twin.state import DigitalTwinState
from utils.constants import OPTIMAL_TEMPERATURE, OPTIMAL_HUMIDITY, OPTIMAL_CO2, OPTIMAL_LIGHT


class GreenhouseEnv(gym.Env):
    """
    A Gymnasium environment for optimizing greenhouse environmental conditions.
    
    State Space:
        - Temperature [°C]
        - Relative Humidity [%]
        - CO₂ concentration [ppm]
        - Light intensity [μmol/m²/s]
        - Plant height [cm]
    
    Action Space (Continuous):
        - Heating control [-1, 1] (negative = cooling, positive = heating)
        - Humidification control [-1, 1] (negative = dehumidify, positive = humidify)
        - CO₂ injection control [0, 1] (0 = off, 1 = max injection)
        - Lighting control [0, 1] (0 = off, 1 = max lighting)
    
    Reward: Positive for maintaining optimal conditions & plant growth, negative for energy waste.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        initial_state: DigitalTwinState,
        episode_length: int = 168,  # 1 week in hours
        reward_weight_growth: float = 0.5,
        reward_weight_conditions: float = 0.3,
        reward_weight_energy: float = 0.2,
    ):
        """
        Initialize the Greenhouse RL Environment.
        
        Args:
            initial_state: Starting Digital Twin state
            episode_length: Number of timesteps per episode
            reward_weight_growth: Weight for plant growth reward
            reward_weight_conditions: Weight for condition maintenance reward
            reward_weight_energy: Weight for energy efficiency penalty
        """
        super().__init__()
        
        self.initial_state = initial_state
        self.episode_length = episode_length
        self.current_step = 0
        
        # Reward weights
        self.w_growth = reward_weight_growth
        self.w_conditions = reward_weight_conditions
        self.w_energy = reward_weight_energy
        
        # Optimal setpoints (from constants)
        self.optimal_temp = OPTIMAL_TEMPERATURE
        self.optimal_humidity = OPTIMAL_HUMIDITY
        self.optimal_co2 = OPTIMAL_CO2
        self.optimal_light = OPTIMAL_LIGHT
        
        # Observation space: [temp, humidity, co2, light, plant_height]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([40.0, 100.0, 2000.0, 2000.0, 200.0]),
            dtype=np.float32
        )
        
        # Action space: [heating, humidification, co2_injection, lighting]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_state = initial_state
        self.history = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_state = self.initial_state
        self.history = []
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment with the given action.
        
        Args:
            action: Control signals [heating, humidification, co2, lighting]
            
        Returns:
            obs: New observation
            reward: Scalar reward signal
            terminated: Episode end flag
            truncated: Time limit flag
            info: Additional info dictionary
        """
        self.current_step += 1
        
        # Apply action: simulate environmental response
        new_temp = self.current_state.temperature + action[0] * 2.0  # ±2°C per action
        new_humidity = np.clip(self.current_state.humidity + action[1] * 10.0, 20, 95)  # ±10% per action
        new_co2 = np.clip(self.current_state.co2 + action[2] * 200.0, 300, 1500)  # 0-200 ppm injection
        new_light = np.clip(self.current_state.light_intensity + action[3] * 400.0, 0, 1500)  # 0-400 μmol/m²/s
        
        # Update plant height based on conditions (simplified growth model)
        growth_factor = self._calculate_growth_factor(new_temp, new_humidity, new_co2, new_light)
        new_plant_height = self.current_state.plant_height + growth_factor * 0.05  # ~5cm/week optimal
        
        # Update state
        self.current_state = DigitalTwinState(
            timestamp=self.current_state.timestamp,
            temperature=new_temp,
            humidity=new_humidity,
            co2=new_co2,
            light_intensity=new_light,
            plant_height=new_plant_height,
            step=self.current_state.step + 1
        )
        
        # Calculate multi-objective reward
        reward = self._calculate_reward(growth_factor, action)
        
        # Episode termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Penalty for extreme conditions
        if new_temp < 10 or new_temp > 35 or new_humidity < 20 or new_humidity > 95:
            reward -= 5.0
            terminated = True
        
        obs = self._get_observation()
        info = {
            "temperature": float(new_temp),
            "humidity": float(new_humidity),
            "co2": float(new_co2),
            "light": float(new_light),
            "plant_height": float(new_plant_height),
            "growth_factor": float(growth_factor),
            "energy_usage": float(np.sum(np.abs(action[:3]))),  # Proxy for energy
        }
        
        self.history.append(info)
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Return current observation as numpy array."""
        return np.array(
            [
                self.current_state.temperature,
                self.current_state.humidity,
                self.current_state.co2,
                self.current_state.light_intensity,
                self.current_state.plant_height,
            ],
            dtype=np.float32
        )
    
    def _calculate_growth_factor(self, temp: float, humidity: float, co2: float, light: float) -> float:
        """
        Calculate plant growth factor based on environmental conditions [0, 1].
        Implements a simplified crop growth model using Growing Degree Days (GDD) concept.
        """
        # Temperature stress factor (optimal: 22-26°C)
        if 15 <= temp <= 30:
            temp_factor = 1.0 - 0.05 * abs(temp - 24.0)
        else:
            temp_factor = 0.0
        
        # Humidity stress factor (optimal: 65-75%)
        humidity_factor = 1.0 - 0.01 * abs(humidity - 70.0)
        
        # CO₂ stress factor (optimal: 800-1200 ppm)
        if 400 <= co2 <= 1500:
            co2_factor = 1.0 - 0.0005 * abs(co2 - 1000.0)
        else:
            co2_factor = 0.5
        
        # Light stress factor (optimal: 800-1500 μmol/m²/s)
        if light >= 200:
            light_factor = min(1.0, light / 1000.0)
        else:
            light_factor = 0.1
        
        # Combined growth factor (weighted average)
        growth_factor = np.clip(
            0.4 * temp_factor + 0.2 * humidity_factor + 0.2 * co2_factor + 0.2 * light_factor,
            0.0,
            1.0
        )
        
        return growth_factor
    
    def _calculate_reward(self, growth_factor: float, action: np.ndarray) -> float:
        """
        Multi-objective reward function balancing growth, conditions, and energy.
        
        Args:
            growth_factor: Plant growth factor [0, 1]
            action: Control actions [heating, humidification, co2, lighting]
        
        Returns:
            Scalar reward signal
        """
        # Growth reward
        growth_reward = growth_factor * 10.0
        
        # Condition maintenance reward (penalize deviation from setpoints)
        temp_penalty = abs(self.current_state.temperature - self.optimal_temp) / 20.0
        humidity_penalty = abs(self.current_state.humidity - self.optimal_humidity) / 50.0
        co2_penalty = abs(self.current_state.co2 - self.optimal_co2) / 500.0
        light_penalty = abs(self.current_state.light_intensity - self.optimal_light) / 1000.0
        
        condition_penalty = (temp_penalty + humidity_penalty + co2_penalty + light_penalty) / 4.0
        condition_reward = -condition_penalty * 5.0
        
        # Energy penalty (discourage excessive control)
        energy_penalty = np.sum(np.abs(action[:3])) * 2.0
        energy_reward = -energy_penalty
        
        # Combined reward
        total_reward = (
            self.w_growth * growth_reward
            + self.w_conditions * condition_reward
            + self.w_energy * energy_reward
        )
        
        return total_reward
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if self.current_step == 0:
            print(f"{'Step':<6} {'Temp(°C)':<12} {'Humid(%)':<12} {'CO₂(ppm)':<12} {'Light(μmol)':<12} {'Height(cm)':<12}")
            print("-" * 66)
        
        if self.current_step % 24 == 0:  # Print every day
            print(
                f"{self.current_step:<6} "
                f"{self.current_state.temperature:<12.2f} "
                f"{self.current_state.humidity:<12.2f} "
                f"{self.current_state.co2:<12.2f} "
                f"{self.current_state.light_intensity:<12.2f} "
                f"{self.current_state.plant_height:<12.2f}"
            )
    
    def close(self):
        """Clean up environment resources."""
        pass
