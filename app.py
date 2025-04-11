import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import numpy as np
import plotly.express as px
from scrape_shl2 import scrape_shl_catalog, save_to_csv
import time
from openrouter_api import generate_content, create_embeddings
import subprocess
import sys

# Set page config at the very beginning
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide"
)

def start_api_server():
    """Start the FastAPI server in a separate process"""
    try:
        # Use Python executable to run api.py
        api_process = subprocess.Popen([sys.executable, "api.py"], 
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        # Wait a few seconds for the server to start
        time.sleep(3)
        return api_process
    except Exception as e:
        st.error(f"Failed to start API server: {e}")
        return None

# Load environment variables
load_dotenv()

# Set up OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Please set OPENROUTER_API_KEY in .env file")
    st.stop()

def load_or_scrape_data():
    """Load data from CSV or scrape if not available"""
    filename = "shl_assessments.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df.empty:
            return df
    
    with st.spinner("Initial data load: Scraping SHL catalog (this may take a few minutes)..."):
        df = scrape_shl_catalog()
        save_to_csv(df, filename)
    
    return df

# Set up API endpoint
API_ENDPOINT = "http://localhost:8000"

# Function to call the FastAPI backend
def get_recommendations_from_api(query, max_results=10):
    """Get assessment recommendations from the FastAPI backend"""
    try:
        response = requests.post(
            f"{API_ENDPOINT}/recommend",
            json={"query": query, "max_results": max_results}
        )
        
        if response.status_code == 200:
            return response.json()["recommended_assessments"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

def get_assessment_detail_from_api(url):
    """Get details for a specific assessment from the FastAPI backend"""
    try:
        response = requests.post(
            f"{API_ENDPOINT}/assessment",
            json={"url": url}
        )
        
        if response.status_code == 200:
            return response.json()["assessment"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_ENDPOINT}/health")
        return response.status_code == 200
    except:
        return False

def scrape_job_description(url):
    """Scrape job description from a provided URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # This is a generic approach - would need customization for specific job sites
            job_description = ""
            # Try to find main content
            main_content = soup.find("div", class_=["job-description", "description"])
            if (main_content):
                job_description = main_content.text.strip()
            else:
                # Fallback to look for paragraphs
                paragraphs = soup.find_all("p")
                job_description = " ".join([p.text.strip() for p in paragraphs])
            
            return job_description
        else:
            return f"Failed to fetch URL: {response.status_code}"
    except Exception as e:
        return f"Error processing URL: {str(e)}"

def create_embeddings_for_app(texts):
    """Generate embeddings for texts using OpenRouter API with Mistral model"""
    embeddings = []
    for text in texts:
        # Check if we have this embedding cached in session state
        if 'embedding_cache' in st.session_state and text in st.session_state.embedding_cache:
            embeddings.append(st.session_state.embedding_cache[text])
            continue
            
        try:
            # Use the OpenRouter API to generate embeddings
            embedding = create_embeddings([text])[0]  # Get the first (and only) embedding
            
            # Cache the embedding
            if 'embedding_cache' not in st.session_state:
                st.session_state.embedding_cache = {}
            st.session_state.embedding_cache[text] = embedding
            
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            # Use a zero vector as fallback
            embeddings.append([0.0] * 768)  # Typical embedding dimension
    return embeddings

def semantic_search(query, df, top_k=10):
    """Perform semantic search using embeddings"""
    # Generate embedding for the query
    query_embedding = create_embeddings_for_app([query])[0]
    
    # If we haven't cached the assessment embeddings, create them
    if 'embedding' not in df.columns:
        with st.spinner("Generating embeddings for assessments (this will take some time)..."):
            # Prepare text representation of each assessment
            assessment_texts = []
            for _, row in df.iterrows():
                text = f"{row['name']}. Test type: {row['test_type']}. "
                assessment_texts.append(text)
            
            # Generate embeddings in batches to avoid rate limits
            batch_size = 5
            all_embeddings = []
            for i in range(0, len(assessment_texts), batch_size):
                batch = assessment_texts[i:i+batch_size]
                batch_embeddings = create_embeddings_for_app(batch)
                all_embeddings.extend(batch_embeddings)
                time.sleep(1)  # Avoid rate limiting
            
            df['embedding'] = all_embeddings
    
    # Calculate cosine similarity
    similarities = []
    for _, row in df.iterrows():
        embedding = row['embedding']
        # Calculate cosine similarity
        dot_product = np.dot(query_embedding, embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_doc = np.linalg.norm(embedding)
        
        # Avoid division by zero
        if norm_query > 0 and norm_doc > 0:
            similarity = dot_product / (norm_query * norm_doc)
        else:
            similarity = 0
            
        similarities.append(similarity)
    
    df['similarity'] = similarities
    results = df.sort_values('similarity', ascending=False).head(top_k).copy()
    
    # Drop embedding column from results to make it easier to display
    if 'embedding' in results.columns:
        results = results.drop('embedding', axis=1)
    
    return results

def get_recommendations(query, df):
    """Get assessment recommendations based on query"""
    # First, let's extract the duration requirement if any
    prompt = f"""
    Analyze this query for time/duration requirements: "{query}"
    
    If there's a specific time/duration mentioned (like "within 30 minutes", "less than 45 minutes", etc.),
    extract and return only the maximum duration in minutes as a number.
    If no specific duration is mentioned, return "None".
    
    Return only the number or "None", nothing else.
    """
    
    duration_text = generate_content(prompt)
    
    try:
        if duration_text is not None and duration_text.lower() != "none":
            max_duration = int(duration_text)
        else:
            max_duration = None
    except ValueError:
        max_duration = None
        
    # Next, perform semantic search
    results = semantic_search(query, df)
    
    # If we have a duration constraint, filter results
    if max_duration is not None:
        # If duration is stored as text, we need to parse it
        if 'duration' in results.columns and not pd.api.types.is_numeric_dtype(results['duration']):
            # Extract numbers from duration strings
            results['duration_min'] = results['duration'].astype(str).str.extract(r'(\d+)').astype(float)
            
            # Filter results where duration is available and within the limit
            has_duration = results['duration_min'].notna()
            within_limit = results['duration_min'] <= max_duration
            
            # Only filter if we have some assessments with valid durations
            if has_duration.any():
                filtered_results = results[has_duration & within_limit].copy()
                
                # If we filtered out all results, return the top results anyway with a warning
                if filtered_results.empty:
                    st.warning(f"No assessments found within {max_duration} minutes. Showing all relevant results.")
                    return results.head(10)  # Return top 10 results
                return filtered_results
            else:
                # If no duration info available, just return the semantic search results
                st.info(f"Duration information not available for most assessments. Showing most relevant results.")
    
    return results

def enhance_recommendations(results, query):
    """Add relevance explanations to recommendations"""
    # For each result, add an explanation of why it's relevant
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
        except Exception as e:
            explanations.append("No explanation available.")
    
    results['relevance'] = explanations
    return results

def main():
    # Start the API server if it's not already running
    if not check_api_health():
        st.info("Starting API server...")
        api_process = start_api_server()
        if api_process:
            time.sleep(2)  # Give the server a moment to start
            if check_api_health():
                st.success("✅ API server started successfully")
            else:
                st.error("❌ Failed to start API server")
                st.stop()
    
    st.title("🧪 SHL Assessment Recommender")
    st.markdown("""
    Enter a job description or query to get personalized SHL assessment recommendations.
    You can include duration requirements like "assessments under 30 minutes" in your query.
    """)
    
    # Check API health
    api_available = check_api_health()
    if not api_available:
        st.error("⚠️ API service is not available. Please make sure the FastAPI server is running.")
        st.stop()
    else:
        st.success("✅ Connected to API service")
    
    tab1, tab2 = st.tabs(["Enter Query", "Upload Job Description URL"])
    
    with tab1:
        query = st.text_area(
            "Enter your query:",
            height=150,
            placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
        )
        search_button = st.button("Get Recommendations", key="search_button")
    
    with tab2:
        url = st.text_input(
            "Enter job description URL:",
            placeholder="https://example.com/job-posting"
        )
        url_button = st.button("Fetch & Analyze", key="url_button")
    
    if search_button and query:
        with st.spinner("Finding the best assessments for you..."):
            recommendations = get_recommendations_from_api(query)
            
            if recommendations:
                # Display prettified results
                st.subheader("📊 Recommended Assessments")
                
                # Display results in a more visual way
                for i, assessment in enumerate(recommendations):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {i+1}. [{assessment['description'].split('.')[0]}]({assessment['url']})")
                            st.markdown(f"**Test Type:** {', '.join(assessment['test_type'])}")
                            st.markdown(f"**Why it's relevant:** {'.'.join(assessment['description'].split('.')[1:])}")
                        with col2:
                            st.markdown(f"**Remote Testing:** {assessment['remote_support']}")
                            st.markdown(f"**Adaptive/IRT:** {assessment['adaptive_support']}")
                            st.markdown(f"**Duration:** {assessment['duration']} minutes")
                        st.markdown("---")
                
                # Add a visualization
                st.subheader("Visualization of Test Types")
                test_types = []
                for assessment in recommendations:
                    test_types.extend(assessment['test_type'])
                
                type_counts = pd.Series(test_types).value_counts().reset_index()
                type_counts.columns = ['Test Type', 'Count']
                
                fig = px.bar(
                    type_counts, 
                    x='Test Type', 
                    y='Count',
                    title='Distribution of Test Types in Recommendations',
                    color='Test Type'
                )
                st.plotly_chart(fig)
            else:
                st.warning("No assessments found matching your query.")
    
    elif url_button and url:
        with st.spinner("Fetching job description..."):
            job_description = scrape_job_description(url)
            if not job_description.startswith("Failed") and not job_description.startswith("Error"):
                st.text_area("Extracted Job Description", job_description, height=200)
                st.session_state.job_description = job_description
                
                # Auto-search with the extracted job description
                with st.spinner("Finding the best assessments for you..."):
                    recommendations = get_recommendations_from_api(job_description)
                    
                    if recommendations:
                        # Display prettified results
                        st.subheader("📊 Recommended Assessments")
                        
                        # Display results in a more visual way
                        for i, assessment in enumerate(recommendations):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### {i+1}. [{assessment['description'].split('.')[0]}]({assessment['url']})")
                                    st.markdown(f"**Test Type:** {', '.join(assessment['test_type'])}")
                                    st.markdown(f"**Why it's relevant:** {'.'.join(assessment['description'].split('.')[1:])}")
                                with col2:
                                    st.markdown(f"**Remote Testing:** {assessment['remote_support']}")
                                    st.markdown(f"**Adaptive/IRT:** {assessment['adaptive_support']}")
                                    st.markdown(f"**Duration:** {assessment['duration']} minutes")
                                st.markdown("---")
                        
                        # Add visualization
                        st.subheader("Visualization of Test Types")
                        test_types = []
                        for assessment in recommendations:
                            test_types.extend(assessment['test_type'])
                        
                        type_counts = pd.Series(test_types).value_counts().reset_index()
                        type_counts.columns = ['Test Type', 'Count']
                        
                        fig = px.bar(
                            type_counts, 
                            x='Test Type', 
                            y='Count',
                            title='Distribution of Test Types in Recommendations',
                            color='Test Type'
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("No assessments found matching the job description.")
            else:
                st.error(job_description)

    # Show API instructions
    with st.expander("API Usage Information"):
        st.markdown("""
        ### API Endpoints
        
        The SHL Assessment Recommender API provides the following endpoints:
        
        #### Recommend Assessments
        ```
        POST /recommend
        ```
        
        Request body:
        ```json
        {
            "query": "your query or job description",
            "max_results": 10
        }
        ```
        
        #### Get Assessment Details
        ```
        POST /assessment
        ```
        
        Request body:
        ```json
        {
            "url": "https://www.shl.com/path/to/assessment"
        }
        ```
        
        #### Health Check
        ```
        GET /health
        ```
        
        For full API documentation, visit: [API Docs](/docs)
        """)

    # Add footer
    st.markdown("---")
    st.markdown("SHL Assessment Recommender | Built with Streamlit and FastAPI")

if __name__ == "__main__":
    main()