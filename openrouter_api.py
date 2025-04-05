# openrouter_api.py - OpenRouter API integration for SHL Assessment Recommender

import os
import requests
import json
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in .env file")

# Model configuration
MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"
API_BASE_URL = "https://openrouter.ai/api/v1"

# Headers for OpenRouter API requests
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://shl-assessment-recommender.com",  # Replace with your actual domain
}

def generate_content(prompt, temperature=0.7):
    """
    Generate content using OpenRouter API with Mistral model
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "data_policy": {"allow_prompt_training": True}
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def create_embeddings(texts):
    """
    Generate embeddings for texts using OpenRouter API with Mistral model
    
    Since OpenRouter doesn't have a dedicated embeddings endpoint for Mistral,
    we'll use a prompt-based approach to generate vector representations.
    """
    embeddings = []
    
    for text in texts:
        try:
            # Prompt the model to generate a vector representation
            prompt = f"""
            Generate a 768-dimensional embedding vector that captures the semantic meaning of this text for SHL assessment matching.
            Focus on skills, competencies, and job roles. Return only the vector as a valid JSON array of numbers.
            
            Text: "{text}"
            
            Vector: 
            """
            
            response = generate_content(prompt)
            
            if response:
                # Extract the vector from the response
                # The response might contain explanatory text, so we need to extract just the vector
                try:
                    # Try to find a JSON array in the response
                    vector_start = response.find('[')
                    vector_end = response.rfind(']') + 1
                    
                    if vector_start >= 0 and vector_end > vector_start:
                        vector_str = response[vector_start:vector_end]
                        vector = json.loads(vector_str)
                        
                        # Ensure we have a 768-dimensional vector
                        if len(vector) != 768:
                            # If not 768 dimensions, pad or truncate
                            if len(vector) < 768:
                                vector.extend([0.0] * (768 - len(vector)))
                            else:
                                vector = vector[:768]
                        
                        embeddings.append(vector)
                    else:
                        # Fallback: generate a random vector
                        print(f"Could not extract vector from response for: {text[:50]}...")
                        embeddings.append(np.random.normal(0, 0.1, 768).tolist())
                except Exception as e:
                    print(f"Error parsing embedding: {e}")
                    embeddings.append(np.random.normal(0, 0.1, 768).tolist())
            else:
                # Fallback: generate a random vector
                print(f"No response for embedding: {text[:50]}...")
                embeddings.append(np.random.normal(0, 0.1, 768).tolist())
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embeddings.append(np.random.normal(0, 0.1, 768).tolist())
    
    return embeddings