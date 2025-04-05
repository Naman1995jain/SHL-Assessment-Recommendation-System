# SHL Assessment Recommender: Technical Approach

## Overview
The SHL Assessment Recommender is a tool designed to match job descriptions with appropriate SHL assessments using semantic search and natural language processing. The system provides both a user-friendly web interface and an API endpoint for programmatic access.

## Architecture
The solution consists of three main components:

1. **Data Collection Layer**: Scrapes SHL's product catalog to build a comprehensive assessment database
2. **Processing Layer**: Uses AI embeddings and semantic search to match queries with relevant assessments
3. **Interface Layer**: Provides both a Streamlit web UI and a FastAPI endpoint

## Technologies Used

### Core Libraries
- **FastAPI**: Powers the API endpoint with automatic documentation
- **Streamlit**: Provides the interactive web interface
- **OpenRouter API**: Integrates with Mistral-small-3.1-24b-instruct model for:
  - Generating text embeddings for semantic search
  - Extracting duration requirements from queries
  - Creating relevance explanations for recommendations

### Data Processing
- **BeautifulSoup**: Web scraping SHL's product catalog
- **Pandas**: Data manipulation and storage
- **NumPy**: Vector operations for semantic search
- **Plotly**: Data visualization in the web interface

### Implementation Details

#### Semantic Search
The system uses vector embeddings to represent both the user query and the assessment descriptions. Cosine similarity is calculated between these vectors to find the most relevant matches. This approach allows the system to understand the semantic meaning behind queries rather than just matching keywords.

#### Intelligent Filtering
The system can extract specific requirements (like time constraints) from natural language queries and filter recommendations accordingly. This is done using the Mistral model to analyze the query text.

#### Relevance Explanations
For each recommended assessment, the system generates a concise explanation of why it's relevant to the query, highlighting specific skills and competencies that match the job requirements.

## Deployment
The solution is deployed with both a web interface (Streamlit) and an API endpoint (FastAPI), allowing for flexible integration into existing workflows. The web interface provides a user-friendly way to explore recommendations, while the API allows for programmatic access from other systems.