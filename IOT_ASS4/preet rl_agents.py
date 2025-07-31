import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import PromptOptimizationEnv
from config import settings
import os
import logging

logger = logging.getLogger(__name__)

class TrainingCallback(BaseCallback):
    """Custom callback to monitor training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log training metrics every 100 steps
        if self.n_calls % 100 == 0:
            if len(self.locals.get('rewards', [])) > 0:
                mean_reward = np.mean(self.locals['rewards'])
                logger.info(f"Step {self.n_calls}: Mean reward = {mean_reward:.3f}")
        return True

class RLAgentManager:
    """Manages multiple RL agents for prompt optimization"""
    
    def __init__(self, save_dir: str = "../models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create environment
        self.env = PromptOptimizationEnv()
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all RL agents"""
        print("Initializing RL agents...")
        
        # PPO - Good for discrete actions and stable training
        print("Creating PPO agent...")
        self.agents['PPO'] = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=settings.learning_rate,
            n_steps=2048,
            batch_size=settings.batch_size,
            verbose=1,
            device='cpu'  # Force CPU usage as recommended
        )
        
        # A2C - Simpler actor-critic method
        print("Creating A2C agent...")
        self.agents['A2C'] = A2C(
            'MlpPolicy',
            self.env,
            learning_rate=settings.learning_rate,
            verbose=1,
            device='cpu'  # Force CPU usage as recommended
        )
        
        print("All agents initialized!")
    
    def train_agent(self, agent_name: str, total_timesteps: int = 10000):
        """Train a specific agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        print(f"Training {agent_name} for {total_timesteps} timesteps...")
        
        # Create callback
        callback = TrainingCallback()
        
        # Train the agent
        self.agents[agent_name].learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model_path = f"{self.save_dir}/{agent_name.lower()}_prompt_optimizer"
        self.agents[agent_name].save(model_path)
        print(f"Model saved to {model_path}")
    
    def train_all_agents(self, total_timesteps: int = 10000):
        """Train all available agents"""
        for agent_name in self.agents.keys():
            print(f"\n=== Training {agent_name} ===")
            self.train_agent(agent_name, total_timesteps)
    
    def test_agent(self, agent_name: str, test_queries: list):
        """Test a trained agent on sample queries"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        print(f"\n=== Testing {agent_name} ===")
        
        for query in test_queries:
            print(f"\nOriginal query: '{query}'")
            
            # Reset environment with new query
            self.env.set_user_query(query)
            obs, info = self.env.reset()
            
            # Get agent's action
            action, _ = self.agents[agent_name].predict(obs, deterministic=True)
            
            # Apply action
            obs, reward, done, truncated, info = self.env.step(action)
            
            print(f"Action taken: {action}")
            print(f"Modified prompt: '{info['modified_prompt']}'")
            print(f"Reward: {reward:.3f}")
    
    def load_agent(self, agent_name: str):
        """Load a pre-trained agent"""
        model_path = f"{self.save_dir}/{agent_name.lower()}_prompt_optimizer"
        
        if not os.path.exists(f"{model_path}.zip"):
            print(f"No saved model found for {agent_name}")
            return False
        
        if agent_name == 'PPO':
            self.agents[agent_name] = PPO.load(model_path)
        elif agent_name == 'A2C':
            self.agents[agent_name] = A2C.load(model_path)
        
        print(f"Loaded {agent_name} model from {model_path}")
        return True

# Test the RL agents
if __name__ == "__main__":
    print("Creating RL Agent Manager...")
    
    # Create agent manager
    manager = RLAgentManager()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does a computer work?",
        "Explain photosynthesis"
    ]
    
    # Train agents for a short time (just to test)
    print("\n=== Quick Training Test ===")
    manager.train_agent('PPO', total_timesteps=1000)
    
    # Test the trained agent
    print("\n=== Testing Trained Agent ===")
    manager.test_agent('PPO', test_queries[:1])  # Test with one query
    
    print("\nRL agents test completed!")