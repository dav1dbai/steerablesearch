# Steerable Search

A search interface for arXiv papers using Sparse Autoencoders to steer embeddings.

## Features

- Normal search with embedding-based vector retrieval
- Manual steering using Sparse Autoencoder features
- Auto-steering with Claude AI analyzing query intent
- Query rewriting for enhanced search effectiveness
- Citation and noise filtering
- arXiv paper linking with metadata

## Setup and Running

### Backend

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   ```bash
   # Create a .env file with your Claude API key
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Run the FastAPI server:
   ```bash
   cd steering_backend
   uvicorn server:app --reload --port 8000
   ```

### Frontend

1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Run the Next.js development server:
   ```bash
   npm run dev
   ```

3. Access the application at http://localhost:3000

## Using Steerable Search

1. **Normal Search**: Enter a query to search for relevant arXiv papers.

2. **Manual Steering**: Select one or more features and adjust their strength to steer the search in specific directions.

3. **Auto-Steering**: Let Claude analyze your query and automatically select features and strengths based on query intent.

4. **Query Rewriting**: Enable query rewriting to have Claude enhance your query with technical terms and synonyms.

5. **Filter Citations & Noise**: Filter out low-value results containing primarily citations or reference material.