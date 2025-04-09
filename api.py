# api.py - FastAPI endpoint for SHL Assessment Recommender

from fastapi import FastAPI, Query, Body, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
import os
import uvicorn
from scrape_shl2 import scrape_shl_catalog, save_to_csv
import time
import numpy as np
from openrouter_api import generate_content, create_embeddings
import re

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
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language query or job description")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return", ge=1, le=20)
    test_types: Optional[List[str]] = Field(None, description="Filter by specific test types")
    max_duration: Optional[int] = Field(None, description="Maximum duration in minutes", ge=0)

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

class AssessmentDetailRequest(BaseModel):
    url: str = Field(..., description="URL of the assessment to get details for")

class AssessmentDetailResponse(BaseModel):
    assessment: Assessment

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
        # Check if text is empty or None
        if not text or not isinstance(text, str):
            print(f"Warning: Invalid text for embedding: {text}")
            embeddings.append([0.0] * 768)
            continue
            
        # Check cache first
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
    # Validate inputs
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    if df is None or df.empty:
        raise ValueError("Assessment dataframe is empty or None")
    
    # Generate embedding for the query
    try:
        query_embedding = create_embeddings_for_api([query])[0]
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query embedding"
        )
    
    # If we haven't cached the assessment embeddings, create them
    if 'embedding' not in df.columns:
        print("Generating embeddings for assessments...")
        assessment_texts = []
        for _, row in df.iterrows():
            # Create a richer text representation for better matching
            text = f"{row['name']}. Test type: {row['test_type']}. "
            if 'duration' in row and row['duration'] != 'N/A':
                text += f"Duration: {row['duration']}. "
            assessment_texts.append(text)
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 5
        all_embeddings = []
        try:
            for i in range(0, len(assessment_texts), batch_size):
                batch = assessment_texts[i:i+batch_size]
                batch_embeddings = create_embeddings_for_api(batch)
                all_embeddings.extend(batch_embeddings)
                time.sleep(1)  # Avoid rate limiting
            
            df['embedding'] = all_embeddings
        except Exception as e:
            print(f"Error generating assessment embeddings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process assessment embeddings"
            )
    
    # Calculate cosine similarity more efficiently
    try:
        similarities = []
        for _, row in df.iterrows():
            embedding = row['embedding']
            # Handle potential serialization issues with embeddings
            if isinstance(embedding, str):
                try:
                    import json
                    embedding = json.loads(embedding)
                except:
                    embedding = [0.0] * 768
            
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
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate similarities"
        )

def extract_duration_requirement(query):
    """Extract duration requirement from query"""
    if not query or not isinstance(query, str):
        return None
        
    # First try a regex-based approach for common patterns
    duration_patterns = [
        r'(?:within|under|less than|no more than|maximum of|max)\s+(\d+)\s*(?:min|minute|minutes)',
        r'(\d+)\s*(?:min|minute|minutes)\s*(?:or less|maximum|max|limit)',
        r'(?:time limit|time constraint|time frame|duration)\s*(?:of|:)?\s*(\d+)\s*(?:min|minute|minutes)'
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    
    # If regex fails, use the LLM approach
    prompt = f"""
    Analyze this query for time/duration requirements: "{query}"
    
    If there's a specific time/duration mentioned (like "within 30 minutes", "less than 45 minutes", etc.),
    extract and return only the maximum duration in minutes as a number.
    If no specific duration is mentioned, return "None".
    
    Return only the number or "None", nothing else.
    """
    
    try:
        duration_text = generate_content(prompt)
        
        if duration_text:
            # Clean up the response
            duration_text = duration_text.strip().lower()
            if duration_text == "none":
                return None
                
            # Try to extract a number
            match = re.search(r'\d+', duration_text)
            if match:
                return int(match.group(0))
        
        return None
    except Exception as e:
        print(f"Error extracting duration: {e}")
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

@app.get("/health")
async def health_check():
    """Health check endpoint with enhanced status information"""
    global assessments_df
    
    status_info = {
        "status": "healthy",
        "api_version": "1.0.0",
        "data_loaded": assessments_df is not None and not assessments_df.empty,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": ["/recommend", "/assessment", "/health"]
    }
    
    return status_info

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest = Body(...)):
    global assessments_df
    
    # Validate input
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 3 characters long"
        )
    
    try:
        # Ensure data is loaded
        if assessments_df is None:
            assessments_df = load_or_scrape_data()
            if assessments_df is None or assessments_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Assessment data is not available. Please try again later."
                )
        
        # Extract duration requirement
        max_duration = extract_duration_requirement(request.query)
        
        # Get recommendations
        top_k = min(request.max_results, 20)  # Limit to 20 max results
        results = semantic_search(request.query, assessments_df, top_k=top_k)
        
        # Filter by duration if specified
        if request.max_duration is not None:
            if 'duration' in results.columns:
                # Handle different duration formats
                if not pd.api.types.is_numeric_dtype(results['duration']):
                    results['duration_min'] = results['duration'].astype(str).str.extract(r'(\d+)').astype(float)
                    filtered_results = results[results['duration_min'] <= request.max_duration].copy()
                    
                    if not filtered_results.empty:
                        results = filtered_results
                else:
                    # If duration is already numeric
                    filtered_results = results[results['duration'] <= request.max_duration].copy()
                    if not filtered_results.empty:
                        results = filtered_results
        
        # Filter by test types if specified
        if request.test_types and len(request.test_types) > 0:
            filtered_results = results[results.apply(
                lambda row: any(test_type in request.test_types 
                                for test_type in row['test_type'].split(',') if isinstance(row['test_type'], str)),
                axis=1
            )].copy()
            
            if not filtered_results.empty:
                results = filtered_results
        
        # Add relevance explanations
        results = enhance_recommendations(results, request.query)
        
        # Convert to response format
        recommended_assessments = []
        for _, row in results.iterrows():
            # Parse test_type string into a list
            test_types = []
            if isinstance(row['test_type'], str):
                test_types = [t.strip() for t in row['test_type'].split(',') if t.strip()]
            
            # Convert duration string to integer
            duration_str = row.get('duration', 'N/A')
            try:
                if isinstance(duration_str, (int, float)):
                    duration = int(duration_str)
                elif duration_str != 'N/A':
                    duration_match = re.search(r'(\d+)', str(duration_str))
                    duration = int(duration_match.group(1)) if duration_match else 30  # Default to 30 minutes if not found
                else:
                    duration = 30  # Default duration
            except Exception as e:
                print(f"Error parsing duration: {e}")
                duration = 30  # Default duration
            
            # Map remote_testing and adaptive_irt to Yes/No format
            remote_support = "Yes" if row.get('remote_testing', '').lower() == "yes" else "No"
            adaptive_support = "Yes" if row.get('adaptive_irt', '').lower() == "yes" else "No"
            
            # Use name and relevance to create a description
            description = f"{row['name']}"
            if 'relevance' in row and row['relevance']:
                description += f". {row['relevance']}"
            
            # Ensure URL is valid
            url = row['url']
            if not url.startswith('http'):
                url = f"https://www.shl.com{url}"
            
            assessment = Assessment(
                url=url,
                test_type=test_types if test_types else ["General"],  # Provide default if empty
                remote_support=remote_support,
                adaptive_support=adaptive_support,
                duration=duration,
                description=description
            )
            recommended_assessments.append(assessment)
        
        return RecommendationResponse(recommended_assessments=recommended_assessments)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error processing recommendation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.post("/assessment", response_model=AssessmentDetailResponse)
async def get_assessment_detail(request: AssessmentDetailRequest = Body(...)):
    """Get details for a specific assessment by URL"""
    global assessments_df
    
    try:
        # Ensure data is loaded
        if assessments_df is None:
            assessments_df = load_or_scrape_data()
            if assessments_df is None or assessments_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Assessment data is not available. Please try again later."
                )
        
        # Normalize URL for comparison
        search_url = request.url
        if not search_url.startswith("http"):
            search_url = f"https://www.shl.com{search_url}"
        
        # Find the assessment by URL
        matching_assessments = assessments_df[assessments_df['url'] == search_url]
        
        if matching_assessments.empty:
            # Try a more flexible match
            base_url = search_url.split("?")[0]  # Remove query parameters
            matching_assessments = assessments_df[assessments_df['url'].str.startswith(base_url)]
        
        if matching_assessments.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        # Get the first matching assessment
        row = matching_assessments.iloc[0]
        
        # Parse test_type string into a list
        test_types = []
        if isinstance(row['test_type'], str):
            test_types = [t.strip() for t in row['test_type'].split(',') if t.strip()]
        
        # Convert duration string to integer
        duration_str = row.get('duration', 'N/A')
        try:
            if isinstance(duration_str, (int, float)):
                duration = int(duration_str)
            elif duration_str != 'N/A':
                duration_match = re.search(r'(\d+)', str(duration_str))
                duration = int(duration_match.group(1)) if duration_match else 30
            else:
                duration = 30
        except Exception as e:
            print(f"Error parsing duration: {e}")
            duration = 30
        
        # Map remote_testing and adaptive_irt to Yes/No format
        remote_support = "Yes" if row.get('remote_testing', '').lower() == "yes" else "No"
        adaptive_support = "Yes" if row.get('adaptive_irt', '').lower() == "yes" else "No"
        
        # Create description
        description = f"{row['name']}"
        
        # Ensure URL is valid
        url = row['url']
        if not url.startswith('http'):
            url = f"https://www.shl.com{url}"
        
        assessment = Assessment(
            url=url,
            test_type=test_types if test_types else ["General"],
            remote_support=remote_support,
            adaptive_support=adaptive_support,
            duration=duration,
            description=description
        )
        
        return AssessmentDetailResponse(assessment=assessment)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error processing assessment detail request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.get("/")
async def root():
    return {
        "message": "SHL Assessment Recommender API is running",
        "endpoints": [
            {
                "path": "/recommend",
                "method": "POST",
                "description": "Get assessment recommendations based on a job description or query"
            },
            {
                "path": "/assessment",
                "method": "POST",
                "description": "Get details for a specific assessment by URL"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            }
        ],
        "version": "1.0.0"
    }

@app.get("/metadata")
async def get_metadata():
    """Get metadata about available assessments"""
    global assessments_df
    
    # Ensure data is loaded
    if assessments_df is None:
        assessments_df = load_or_scrape_data()
        if assessments_df is None or assessments_df.empty:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assessment data is not available. Please try again later."
            )
    
    try:
        # Calculate metadata
        metadata = {
            "total_assessments": len(assessments_df),
            "test_types": [],
            "duration_stats": {
                "min": 0,
                "max": 0,
                "avg": 0
            },
            "remote_testing_available": 0,
            "adaptive_testing_available": 0
        }
        
        # Get unique test types
        all_test_types = []
        for test_type in assessments_df['test_type']:
            if isinstance(test_type, str):
                types = [t.strip() for t in test_type.split(',')]
                all_test_types.extend(types)
        
        # Count unique test types
        test_type_counts = {}
        for test_type in all_test_types:
            if test_type in test_type_counts:
                test_type_counts[test_type] += 1
            else:
                test_type_counts[test_type] = 1
        
        # Format test types for response
        metadata["test_types"] = [{
            "name": test_type,
            "count": count
        } for test_type, count in test_type_counts.items()]
        
        # Calculate duration statistics
        durations = []
        for duration in assessments_df['duration']:
            try:
                if isinstance(duration, (int, float)):
                    durations.append(int(duration))
                elif duration != 'N/A':
                    duration_match = re.search(r'(\d+)', str(duration))
                    if duration_match:
                        durations.append(int(duration_match.group(1)))
            except:
                pass
        
        if durations:
            metadata["duration_stats"] = {
                "min": min(durations),
                "max": max(durations),
                "avg": round(sum(durations) / len(durations))
            }
        
        # Count remote and adaptive testing availability
        metadata["remote_testing_available"] = assessments_df['remote_testing'].str.lower().eq('yes').sum()
        metadata["adaptive_testing_available"] = assessments_df['adaptive_irt'].str.lower().eq('yes').sum()
        
        return metadata
    
    except Exception as e:
        print(f"Error generating metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing metadata"
        )

# Pre-load data when starting the server
@app.on_event("startup")
async def startup_event():
    global assessments_df
    try:
        assessments_df = load_or_scrape_data()
        print(f"Loaded {len(assessments_df)} assessments")
    except Exception as e:
        print(f"Error loading assessment data: {e}")
        # Don't raise an exception here, as it would prevent the server from starting

if __name__ == "__main__":
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)