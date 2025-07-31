# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from typing import Dict, List, Tuple, Any
# import asyncio
# from groq_client import GroqClient
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import logging

# logger = logging.getLogger(__name__)

# class PromptOptimizationEnv(gym.Env):
#     """
#     Reinforcement Learning Environment for Prompt Optimization
#     """
    
#     def __init__(self, max_prompt_length: int = 500):
#         super().__init__()
        
#         # Initialize components
#         self.groq_client = GroqClient()
#         self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Environment parameters
#         self.max_prompt_length = max_prompt_length
#         self.current_user_query = ""
#         self.current_modified_prompt = ""
#         self.response_history = []
        
#         # Action space: discrete actions for prompt modifications
#         # 0: Add clarity phrases, 1: Add specificity, 2: Simplify, 3: Add context
#         self.action_space = spaces.Discrete(4)
        
#         # Observation space: embedding of current prompt + metadata
#         self.observation_space = spaces.Box(
#             low=-1.0, high=1.0, 
#             shape=(384 + 10,),  # 384 for sentence embedding + 10 for metadata
#             dtype=np.float32
#         )
        
#         # Reset environment
#         self.reset()
    
#     def reset(self, seed=None, options=None):
#         """Reset the environment"""
#         super().reset(seed=seed)
        
#         self.current_user_query = ""
#         self.current_modified_prompt = ""
#         self.response_history = []
#         self.step_count = 0
        
#         # Return initial observation
#         return self._get_observation(), {}
    
#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         """Execute one step in the environment"""
#         self.step_count += 1
        
#         # Apply action to modify prompt
#         self.current_modified_prompt = self._apply_action(action, self.current_user_query)
        
#         # Calculate reward
#         reward = asyncio.run(self._calculate_reward())
        
#         # Check if episode is done
#         done = self.step_count >= 10  # Max 10 modifications per episode
#         truncated = False
        
#         # Get new observation
#         observation = self._get_observation()
        
#         info = {
#             'original_query': self.current_user_query,
#             'modified_prompt': self.current_modified_prompt,
#             'reward_components': self.last_reward_components
#         }
        
#         return observation, reward, done, truncated, info
    
#     def set_user_query(self, query: str):
#         """Set the current user query to optimize"""
#         self.current_user_query = query
#         self.current_modified_prompt = query
#         self.step_count = 0
    
#     def _apply_action(self, action: int, prompt: str) -> str:
#         """Apply modification action to prompt"""
#         if action == 0:  # Add clarity phrases
#             return f"Please provide a clear and detailed explanation of: {prompt}"
#         elif action == 1:  # Add specificity
#             return f"Specifically, {prompt}. Include examples and details."
#         elif action == 2:  # Simplify
#             return f"In simple terms: {prompt}"
#         elif action == 3:  # Add context
#             return f"Context: This is a user question that needs a comprehensive answer. Question: {prompt}"
#         else:
#             return prompt
    
#     def _get_observation(self) -> np.ndarray:
#         """Get current state observation"""
#         if not self.current_modified_prompt:
#             # Return zero observation if no prompt
#             return np.zeros(394, dtype=np.float32)
        
#         # Get sentence embedding
#         embedding = self.sentence_model.encode([self.current_modified_prompt])[0]
        
#         # Add metadata features
#         metadata = np.array([
#             len(self.current_modified_prompt) / self.max_prompt_length,  # Length ratio
#             self.current_modified_prompt.count('?') / 10,  # Question marks
#             self.current_modified_prompt.count('.') / 10,  # Periods
#             self.current_modified_prompt.count(',') / 10,  # Commas
#             len(self.current_modified_prompt.split()) / 100,  # Word count ratio
#             self.step_count / 10,  # Step progress
#             len(self.response_history) / 10,  # Response history length
#             0.0,  # Placeholder for user feedback
#             0.0,  # Placeholder for sentiment
#             0.0   # Placeholder for clarity score
#         ], dtype=np.float32)
        
#         # Combine embedding and metadata
#         observation = np.concatenate([embedding, metadata])
#         return observation
    
#     async def _calculate_reward(self) -> float:
#         """Calculate reward for current prompt modification"""
#         if not self.current_modified_prompt:
#             return 0.0
        
#         # Component 1: Clarity and specificity (using LLM evaluation)
#         clarity_score = await self.groq_client.evaluate_prompt_clarity(self.current_modified_prompt)
#         clarity_reward = (clarity_score - 5) / 5  # Normalize to [-1, 1]
        
#         # Component 2: Response consistency (if we have multiple responses)
#         consistency_reward = 0.0
#         if len(self.response_history) > 1:
#             embeddings = self.sentence_model.encode(self.response_history[-2:])
#             if len(embeddings) >= 2:
#                 similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#                 consistency_reward = similarity
        
#         # Component 3: Length penalty (avoid overly long prompts)
#         length_penalty = max(0, (len(self.current_modified_prompt) - 200) / 1000)
        
#         # Combine rewards
#         total_reward = (
#             0.6 * clarity_reward + 
#             0.3 * consistency_reward - 
#             0.1 * length_penalty
#         )
        
#         # Store reward components for debugging
#         self.last_reward_components = {
#             'clarity': clarity_reward,
#             'consistency': consistency_reward,
#             'length_penalty': length_penalty,
#             'total': total_reward
#         }
        
#         return total_reward

# # Test the environment
# if __name__ == "__main__":
#     print("Starting RL environment test...")
    
#     print("Loading sentence transformer model...")
#     try:
#         from sentence_transformers import SentenceTransformer
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Model loading error: {e}")
    
#     print("Creating environment...")
#     try:
#         env = PromptOptimizationEnv()
#         print("Environment created!")
#     except Exception as e:
#         print(f"Environment creation error: {e}")
#         exit()
    
#     print("Setting user query...")
#     env.set_user_query("What is machine learning?")
    
#     print("Resetting environment...")
#     obs, info = env.reset()
#     print(f"Initial observation shape: {obs.shape}")
    
#     print("Taking action...")
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
    
#     print(f"Action: {action}")
#     print(f"Modified prompt: {info['modified_prompt']}")
#     print(f"Reward: {reward}")
#     print("Test completed!")



import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import asyncio
from groq_client import GroqClient
import logging

logger = logging.getLogger(__name__)

class PromptOptimizationEnv(gym.Env):
    """
    Simplified Reinforcement Learning Environment for Prompt Optimization
    """
    
    def __init__(self, max_prompt_length: int = 500):
        super().__init__()
        
        # Initialize components
        self.groq_client = GroqClient()
        
        # Environment parameters
        self.max_prompt_length = max_prompt_length
        self.current_user_query = ""
        self.current_modified_prompt = ""
        self.response_history = []
        
        # Action space: discrete actions for prompt modifications
        # 0: Add clarity phrases, 1: Add specificity, 2: Simplify, 3: Add context
        self.action_space = spaces.Discrete(4)
        
        # Observation space: simplified features without embeddings
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(10,),  # 10 basic features
            dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_user_query = ""
        self.current_modified_prompt = ""
        self.response_history = []
        self.step_count = 0
        
        # Return initial observation
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Apply action to modify prompt
        self.current_modified_prompt = self._apply_action(action, self.current_user_query)
        
        # Calculate reward (simplified without async for now)
        reward = self._calculate_simple_reward()
        
        # Check if episode is done
        done = self.step_count >= 10  # Max 10 modifications per episode
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        
        info = {
            'original_query': self.current_user_query,
            'modified_prompt': self.current_modified_prompt,
            'step_count': self.step_count
        }
        
        return observation, reward, done, truncated, info
    
    def set_user_query(self, query: str):
        """Set the current user query to optimize"""
        self.current_user_query = query
        self.current_modified_prompt = query
        self.step_count = 0
    
    def _apply_action(self, action: int, prompt: str) -> str:
        """Apply modification action to prompt"""
        if action == 0:  # Add clarity phrases
            return f"Please provide a clear and detailed explanation of: {prompt}"
        elif action == 1:  # Add specificity
            return f"Specifically, {prompt}. Include examples and details."
        elif action == 2:  # Simplify
            return f"In simple terms: {prompt}"
        elif action == 3:  # Add context
            return f"Context: This is a user question that needs a comprehensive answer. Question: {prompt}"
        else:
            return prompt
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        if not self.current_modified_prompt:
            return np.zeros(10, dtype=np.float32)
        
        # Basic text features
        features = np.array([
            len(self.current_modified_prompt) / self.max_prompt_length,  # Length ratio
            self.current_modified_prompt.count('?') / 10,  # Question marks
            self.current_modified_prompt.count('.') / 10,  # Periods
            self.current_modified_prompt.count(',') / 10,  # Commas
            len(self.current_modified_prompt.split()) / 100,  # Word count ratio
            self.step_count / 10,  # Step progress
            len(self.response_history) / 10,  # Response history length
            1.0 if 'clear' in self.current_modified_prompt.lower() else 0.0,  # Has clarity words
            1.0 if 'specific' in self.current_modified_prompt.lower() else 0.0,  # Has specificity words
            len([w for w in self.current_modified_prompt.split() if len(w) > 6]) / 20  # Complex words ratio
        ], dtype=np.float32)
        
        return features
    
    def _calculate_simple_reward(self) -> float:
        """Calculate simplified reward without API calls"""
        if not self.current_modified_prompt:
            return 0.0
        
        # Reward longer, more detailed prompts
        length_reward = min(len(self.current_modified_prompt) / 200, 1.0)
        
        # Reward specific keywords
        clarity_words = ['clear', 'detailed', 'explain', 'specific', 'examples']
        keyword_bonus = sum(1 for word in clarity_words if word in self.current_modified_prompt.lower()) * 0.1
        
        # Penalty for overly long prompts
        length_penalty = max(0, (len(self.current_modified_prompt) - 300) / 1000)
        
        total_reward = length_reward + keyword_bonus - length_penalty
        return total_reward

# Test the environment
if __name__ == "__main__":
    print("Creating simplified RL environment...")
    
    env = PromptOptimizationEnv()
    print("Environment created!")
    
    env.set_user_query("What is machine learning?")
    print("User query set!")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Action taken: {action}")
    print(f"Modified prompt: {info['modified_prompt']}")
    print(f"Reward: {reward}")
    print("Test completed successfully!")