# api.py - FastAPI endpoint for SHL Assessment Recommender

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
import os
import uvicorn
from scrape_shl2 import scrape_shl_catalog, save_to_csv
import time
import numpy as np
from openrouter_api import generate_content, create_embeddings

# Load environment variables
load_dotenv()

# Set up OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in .env file")

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

# Data models
class Assessment(BaseModel):
    name: str
    url: str
    test_type: str
    remote_testing: str
    adaptive_irt: str
    duration: Optional[str] = "N/A"
    relevance: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]

# Global variables
assessments_df = None
embedding_cache = {}

def load_or_scrape_data():
    """Load data from CSV or scrape if not available"""
    global assessments_df
    
    filename = "shl_assessments.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df.empty:
            return df
    
    print("Initial data load: Scraping SHL catalog...")
    df = scrape_shl_catalog()
    save_to_csv(df, filename)
    
    return df

def create_embeddings_for_api(texts):
    """Generate embeddings for texts using OpenRouter API with Mistral model"""
    embeddings = []
    for text in texts:
        if text in embedding_cache:
            embeddings.append(embedding_cache[text])
            continue
            
        try:
            # Use the OpenRouter API to generate embeddings
            embedding = create_embeddings([text])[0]  # Get the first (and only) embedding
            embedding_cache[text] = embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Use a zero vector as fallback with appropriate dimensionality
            embeddings.append([0.0] * 768)  # Typical embedding dimension
    return embeddings

def semantic_search(query, df, top_k=10):
    """Perform semantic search using embeddings"""
    # Generate embedding for the query
    query_embedding = create_embeddings_for_api([query])[0]
    
    # If we haven't cached the assessment embeddings, create them
    if 'embedding' not in df.columns:
        print("Generating embeddings for assessments...")
        assessment_texts = []
        for _, row in df.iterrows():
            text = f"{row['name']}. Test type: {row['test_type']}. "
            assessment_texts.append(text)
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 5
        all_embeddings = []
        for i in range(0, len(assessment_texts), batch_size):
            batch = assessment_texts[i:i+batch_size]
            batch_embeddings = create_embeddings_for_api(batch)
            all_embeddings.extend(batch_embeddings)
            time.sleep(1)  # Avoid rate limiting
        
        df['embedding'] = all_embeddings
    
    # Calculate cosine similarity
    similarities = []
    for _, row in df.iterrows():
        embedding = row['embedding']
        dot_product = np.dot(query_embedding, embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_doc = np.linalg.norm(embedding)
        
        if norm_query > 0 and norm_doc > 0:
            similarity = dot_product / (norm_query * norm_doc)
        else:
            similarity = 0
            
        similarities.append(similarity)
    
    df['similarity'] = similarities
    results = df.sort_values('similarity', ascending=False).head(top_k).copy()
    
    if 'embedding' in results.columns:
        results = results.drop('embedding', axis=1)
    
    return results

def extract_duration_requirement(query):
    """Extract duration requirement from query"""
    prompt = f"""
    Analyze this query for time/duration requirements: "{query}"
    
    If there's a specific time/duration mentioned (like "within 30 minutes", "less than 45 minutes", etc.),
    extract and return only the maximum duration in minutes as a number.
    If no specific duration is mentioned, return "None".
    
    Return only the number or "None", nothing else.
    """
    
    try:
        duration_text = generate_content(prompt)
        
        if duration_text and duration_text.lower() != "none":
            return int(duration_text)
        else:
            return None
    except Exception:
        return None

def enhance_recommendations(results, query):
    """Add relevance explanations to recommendations"""
    explanations = []
    
    for _, row in results.iterrows():
        prompt = f"""
        Explain why this SHL assessment is relevant to the following query. Focus on specific skills, competencies, and job requirements that match.
        
        Query: "{query}"
        
        Assessment: "{row['name']}"
        Test type: "{row['test_type']}"
        Remote testing: "{row['remote_testing']}"
        Adaptive/IRT: "{row['adaptive_irt']}"
        Duration: "{row.get('duration', 'N/A')}"
        
        Your explanation should:
        1. Be 1-2 concise sentences
        2. Highlight specific skills or competencies measured by this assessment
        3. Explain how these skills relate to the query
        4. Mention any relevant features (remote testing, adaptive nature) if they match query requirements
        
        Keep your explanation focused and specific.
        """
        
        try:
            explanation = generate_content(prompt)
            if explanation:
                explanations.append(explanation.strip())
            else:
                explanations.append("No explanation available.")
        except Exception:
            explanations.append("No explanation available.")
    
    results['relevance'] = explanations
    return results

@app.get("/api/recommend", response_model=RecommendationResponse)
async def recommend(
    query: str = Query(..., description="Job description or query text"), 
    top_k: int = Query(10, ge=1, le=10, description="Number of recommendations to return")
):
    global assessments_df
    
    # Ensure data is loaded
    if assessments_df is None:
        assessments_df = load_or_scrape_data()
    
    # Extract duration requirement
    max_duration = extract_duration_requirement(query)
    
    # Get recommendations
    results = semantic_search(query, assessments_df, top_k=top_k)
    
    # Filter by duration if specified
    if max_duration is not None:
        if 'duration' in results.columns and not pd.api.types.is_numeric_dtype(results['duration']):
            results['duration_min'] = results['duration'].astype(str).str.extract(r'(\d+)').astype(float)
            filtered_results = results[results['duration_min'] <= max_duration].copy()
            
            if not filtered_results.empty:
                results = filtered_results
    
    # Add relevance explanations
    results = enhance_recommendations(results, query)
    
    # Convert to response format
    recommendations = []
    for _, row in results.iterrows():
        assessment = Assessment(
            name=row['name'],
            url=row['url'],
            test_type=row['test_type'],
            remote_testing=row['remote_testing'],
            adaptive_irt=row['adaptive_irt'],
            duration=row.get('duration', 'N/A'),
            relevance=row.get('relevance', None)
        )
        recommendations.append(assessment)
    
    return RecommendationResponse(recommendations=recommendations)

@app.get("/")
async def root():
    return {"message": "SHL Assessment Recommender API is running. Use /api/recommend endpoint to get recommendations."}

# Pre-load data when starting the server
@app.on_event("startup")
async def startup_event():
    global assessments_df
    assessments_df = load_or_scrape_data()
    print(f"Loaded {len(assessments_df)} assessments")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)