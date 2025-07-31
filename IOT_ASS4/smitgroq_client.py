import asyncio
import logging
from typing import Dict, List, Optional
from groq import Groq
from config import settings

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error: {str(e)}"
    
    async def evaluate_prompt_clarity(self, prompt: str) -> float:
        """Use LLM to evaluate prompt clarity (1-10 scale)"""
        eval_prompt = f"""Rate the clarity and specificity of this prompt on a scale of 1-10.
        Only respond with a number between 1 and 10.
        
        Prompt to evaluate: "{prompt}"
        
        Rating:"""
        
        try:
            response = await self.generate_response(eval_prompt, max_tokens=10)
            # Extract number from response
            rating = float(''.join(filter(str.isdigit, response.split()[0])))
            return min(max(rating, 1), 10)  # Clamp between 1-10
        except:
            return 5.0  # Default neutral rating
    
    async def verify_factual_accuracy(self, statement: str) -> float:
        """Use LLM to verify factual accuracy"""
        verify_prompt = f"""Verify if this statement is factually correct. 
        Respond with a confidence score between 0 and 1, where:
        - 1.0 means definitely correct
        - 0.5 means uncertain/mixed evidence  
        - 0.0 means definitely incorrect
        
        Only respond with a number between 0 and 1.
        
        Statement: "{statement}"
        
        Confidence:"""
        
        try:
            response = await self.generate_response(verify_prompt, max_tokens=10)
            confidence = float(response.strip())
            return min(max(confidence, 0), 1)  # Clamp between 0-1
        except:
            return 0.5  # Default uncertain

# Test function
async def test_groq_connection():
    client = GroqClient()
    test_prompt = "What is the capital of France?"
    response = await client.generate_response(test_prompt)
    print(f"Test response: {response}")

if __name__ == "__main__":
    asyncio.run(test_groq_connection())