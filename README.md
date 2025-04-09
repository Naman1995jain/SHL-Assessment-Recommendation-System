# SHL Assessment Recommender 🚀

A tool for recommending SHL assessments based on job descriptions using semantic search and natural language processing.

## Overview 📋

The SHL Assessment Recommender helps HR professionals and recruiters find the most appropriate SHL assessments for their hiring needs. By analyzing job descriptions or specific queries, the system recommends relevant assessments from SHL's catalog, providing explanations for why each assessment is suitable.

## 🧠 System Architecture

The following diagram illustrates the architecture of the SHL Assessment Recommender system:

![System Architecture](image/diagram.png)

## Features 🌟

- **Semantic Search**: Uses AI embeddings to understand the meaning behind job descriptions
- **Intelligent Filtering**: Automatically extracts requirements like time constraints from natural language queries
- **Relevance Explanations**: Provides concise explanations for why each assessment is recommended
- **Dual Interface**: Access via web UI or API endpoint
- **Data Visualization**: Visual breakdown of recommended assessment types

## Demo 🎥

A live demo is available at: [http://localhost:8503](https://shl-assessment-recommendation-system.streamlit.app/)

link for endpointapi : [Endpint](https://nj1995-shl-epo.hf.space)

A live demo for Api is available at:  [URLi is available at: [URL_ADDRESS:8000/docs](URL_ADDRESS000/docs](https://nj1995-shl-epo.hf.space/docs)

## API Usage 📡

### Endpoint

```
GET /api/recommend?query=your+query+here
```

### Response Format

```json
{
    "recommendations": [
        {
            "name": "Assessment Name",
            "url": "https://www.shl.com/path/to/assessment",
            "test_type": "Test Type",
            "remote_testing": "Yes/No",
            "adaptive_irt": "Yes/No",
            "duration": "Duration info",
            "relevance": "Explanation of relevance"
        },
    ]
}
```

## Setup 🛠️

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Run the Streamlit app: `streamlit run app.py`
5. Run the API server: `uvicorn api:app --reload`

## Technologies Used

- **FastAPI**: API endpoint
- **Streamlit**: Web interface
- **OpenRouter API** (Mistral model): Text embeddings and content generation
- **BeautifulSoup**: Web scraping
- **Pandas & NumPy**: Data processing
- **Plotly**: Data visualization

## Project Structure

- `app.py`: Streamlit web interface
- `api.py`: FastAPI endpoint
- `scrape_shl2.py`: SHL catalog scraper
- `openrouter_api.py`: OpenRouter API integration
- `shl_assessments.csv`: Cached assessment data
- `approach_document.md`: Technical approach documentation

---

## 👤 Author

Email: [namanjain34710@gmail.com](mailto\:namanjain34710@gmail.com)

LinkedIn: [Naman Jain](https://www.linkedin.com/in/naman-jain-226135201/)
