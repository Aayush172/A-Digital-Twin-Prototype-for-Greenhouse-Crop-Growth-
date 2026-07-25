"""
Reinforcement Learning Agent Trainer for Greenhouse Environmental Optimization.

Implements PPO (Proximal Policy Optimization) for learning optimal control policies
for heating, ventilation, humidification, and lighting systems.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EarlyStopping, BaseCallback
from stable_baselines3.common.logger import configure
from optimization.env import GreenhouseEnv
from digital_twin.state import DigitalTwinState
from utils.logger import get_logger


logger = get_logger("RLAgent")


class CustomLoggingCallback(BaseCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        self.current_episode_reward += self.locals.get("rewards", 0)
        self.current_episode_length += 1
        
        # Check for episode termination
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(
                    f"Episode {len(self.episode_rewards)}: "
                    f"Reward={self.current_episode_reward:.2f}, "
                    f"Avg(last 10)={avg_reward:.2f}, "
                    f"Length={self.current_episode_length}"
                )
            
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        return True


class RLAgent:
    """
    Reinforcement Learning Agent for Greenhouse Climate Control.
    
    Uses PPO (Proximal Policy Optimization) to learn optimal environmental control policies.
    """
    
    def __init__(
        self,
        env: GreenhouseEnv,
        model_dir: Path = Path("results/models"),
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize RL Agent.
        
        Args:
            env: Gymnasium environment
            model_dir: Directory to save trained models
            learning_rate: PPO learning rate
            batch_size: Training batch size
            n_epochs: Number of training epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
        """
        self.env = env
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            verbose=0,
            device="cpu"
        )
        
        logger.info(f"Initialized PPO Agent with learning_rate={learning_rate}, batch_size={batch_size}")
    
    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        patience: int = 5,
    ) -> Dict:
        """
        Train the RL agent using PPO.
        
        Args:
            total_timesteps: Total number of environment steps
            eval_freq: Evaluation frequency (steps between evaluations)
            patience: Early stopping patience
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting RL training for {total_timesteps} timesteps...")
        
        # Custom logging callback
        logging_callback = CustomLoggingCallback(verbose=0)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[logging_callback],
            progress_bar=True
        )
        
        logger.info("RL training completed.")
        
        # Save trained model
        model_path = self.model_dir / "ppo_greenhouse_agent"
        self.model.save(str(model_path))
        logger.info(f"Saved trained PPO model to {model_path}.zip")
        
        results = {
            "episode_rewards": logging_callback.episode_rewards,
            "episode_lengths": logging_callback.episode_lengths,
            "total_timesteps": total_timesteps,
            "final_avg_reward": np.mean(logging_callback.episode_rewards[-20:]) if logging_callback.episode_rewards else 0.0,
        }
        
        return results
    
    def evaluate(self, n_eval_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent on test episodes.
        
        Args:
            n_eval_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating agent over {n_eval_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_growth = []
        
        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_info = {"plant_height": []}
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_info["plant_height"].append(info["plant_height"])
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            growth = episode_info["plant_height"][-1] - self.env.initial_state.plant_height
            episode_growth.append(growth)
            
            logger.info(
                f"Episode {episode + 1}/{n_eval_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Length={episode_length}, "
                f"Growth={growth:.2f} cm"
            )
        
        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "mean_growth": float(np.mean(episode_growth)),
            "total_growth": float(np.sum(episode_growth)),
        }
        
        logger.info(f"Evaluation Results: {results}")
        
        return results
    
    def get_policy(self):
        """Return the learned policy."""
        return self.model.policy
    
    @classmethod
    def load(cls, model_path: Path, env: GreenhouseEnv) -> "RLAgent":
        """
        Load a trained agent from disk.
        
        Args:
            model_path: Path to saved model
            env: Environment instance
        
        Returns:
            Loaded RLAgent instance
        """
        agent = cls(env)
        agent.model = PPO.load(str(model_path))
        logger.info(f"Loaded trained PPO model from {model_path}")
        return agent
