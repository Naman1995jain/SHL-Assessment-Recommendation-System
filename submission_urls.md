# SHL Assessment Recommender: Submission URLs

This document contains the three URLs required for submission of the SHL Assessment Recommender project.

## 1. Working Demo URL

The working demo is hosted at:
```
http://localhost:8503
```

This Streamlit application allows users to enter job descriptions or queries and receive personalized SHL assessment recommendations with relevance explanations.

## 2. API Endpoint URL

The API endpoint is available at:
```
http://localhost:8000/api/recommend?query=your_query_here
```

This endpoint accepts GET requests with a query parameter and returns assessment recommendations in JSON format.

## 3. GitHub Repository URL

The code is hosted on GitHub at:
```
[Insert your GitHub repository URL here after uploading the code]
```

## How to Start the Services

### Running the Streamlit Demo
The Streamlit demo is already running on port 8503. If you need to restart it, use:
```
streamlit run app.py
```

### Running the API Server
To start the API server, use:
```
uvicorn api:app --reload --port 8000
```

### API Documentation
Once the API server is running, you can access the auto-generated documentation at:
```
http://localhost:8000/docs
```