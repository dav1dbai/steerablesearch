'''
FastAPI server for frontend
'''
import os
import logging
from fastapi import FastAPI, HTTPException
import numpy as np
from PyPDF2 import PdfReader # Use PyPDF2 directly
from pydantic import BaseModel # Import BaseModel
from sklearn.metrics.pairwise import cosine_similarity # For similarity search

# Assuming embed.py is in the same directory
from embed import load_embedding_model, _MLX_AVAILABLE, _TORCH_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Configuration ---
ARXIV_DIR = os.path.join(os.path.dirname(__file__), "arxiv")
TARGET_EMBEDDING_LAYER = 10 # Example layer, adjust as needed
CHUNK_SIZE = 500 # Characters per chunk (adjust as needed)
CHUNK_OVERLAP = 50 # Overlap between chunks (adjust as needed)

# --- Global Variables (Load model on startup) ---
MODEL = None
TOKENIZER = None
EMBED_FUNCTION = None
EMBEDDING_DIM = None # Will be determined after loading model
# In-memory storage for embeddings and metadata
STORED_EMBEDDINGS = [] # List to store numpy embeddings
STORED_METADATA = []  # List to store corresponding metadata dicts

# --- Pydantic model for search request body ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER, EMBED_FUNCTION, EMBEDDING_DIM
    try:
        logger.info("Loading embedding model...")
        MODEL, TOKENIZER, EMBED_FUNCTION, model_name = load_embedding_model()
        logger.info(f"Model {model_name} loaded successfully.")

        # Determine embedding dimension by embedding a dummy text
        logger.info("Determining embedding dimension...")
        dummy_text = "hello"
        try:
            dummy_embedding = EMBED_FUNCTION(MODEL, TOKENIZER, dummy_text, TARGET_EMBEDDING_LAYER)
            # Convert to numpy and get shape
            if _MLX_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_mlx':
                emb_np = np.array(dummy_embedding)
            elif _TORCH_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_torch':
                emb_np = dummy_embedding.cpu().numpy()
            else:
                 # Fallback/error case, though load_embedding_model should prevent this
                 raise RuntimeError("Unknown embedding function type during dimension check.")
            # Shape is likely (1, sequence_length, embedding_dim), take last dim
            EMBEDDING_DIM = emb_np.shape[-1]
            logger.info(f"Determined embedding dimension: {EMBEDDING_DIM}")
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {e}", exc_info=True)
            raise RuntimeError("Could not determine embedding dimension.") from e

    except RuntimeError as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        # Depending on deployment, you might want to exit or handle this differently
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during startup: {e}", exc_info=True)
        raise


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple text chunking function."""
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

@app.post("/process/{filename}")
def process_document(filename: str):
    # We modify global lists, so declare intent (though not strictly needed for append)
    global STORED_EMBEDDINGS, STORED_METADATA

    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(ARXIV_DIR, filename)
    logger.info(f"Processing document: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if MODEL is None or TOKENIZER is None or EMBED_FUNCTION is None:
        logger.error("Model not initialized. Check startup logs.")
        raise HTTPException(status_code=500, detail="Server not ready. Model not initialized.")

    try:
        # 1. Read PDF
        logger.info(f"Reading PDF: {filename}")
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        logger.info(f"Extracted {len(full_text)} characters from {filename}")

        if not full_text.strip():
            logger.warning(f"No text extracted from {filename}. Skipping.")
            return {"message": f"No text found in {filename}."}

        # 2. Chunk Text
        logger.info(f"Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"Created {len(text_chunks)} chunks.")

        # 3. Embed Chunks
        logger.info(f"Embedding {len(text_chunks)} chunks using layer {TARGET_EMBEDDING_LAYER}...")
        embeddings_list = []
        chunk_metadata = []

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue # Skip empty chunks
            try:
                # Embed and handle potential backend differences
                hidden_states = EMBED_FUNCTION(MODEL, TOKENIZER, chunk, TARGET_EMBEDDING_LAYER)

                # Convert to numpy float32, average sequence dimension if necessary
                if _MLX_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_mlx':
                    emb_np = np.array(hidden_states, dtype=np.float32)
                elif _TORCH_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_torch':
                    emb_np = hidden_states.cpu().numpy().astype(np.float32)
                else:
                    # Should not happen based on startup checks
                    raise RuntimeError("Unknown embedding function type during processing.")

                # Embeddings might have shape (1, seq_len, dim)
                # We need (dim,) or (num_embeddings, dim). Let's average the sequence length.
                if emb_np.ndim == 3 and emb_np.shape[0] == 1:
                    # Average across the sequence length dimension (axis 1)
                    avg_embedding = np.mean(emb_np[0, :, :], axis=0)
                elif emb_np.ndim == 2: # Maybe it's already (seq_len, dim)?
                    avg_embedding = np.mean(emb_np, axis=0)
                elif emb_np.ndim == 1: # Already (dim,)
                     avg_embedding = emb_np
                else:
                    logger.warning(f"Unexpected embedding shape {emb_np.shape} for chunk {i}. Skipping.")
                    continue

                if avg_embedding.shape[0] != EMBEDDING_DIM:
                     logger.warning(f"Embedding dimension mismatch ({avg_embedding.shape[0]} vs {EMBEDDING_DIM}) for chunk {i}. Skipping.")
                     continue

                embeddings_list.append(avg_embedding)
                chunk_metadata.append({"source": filename, "text": chunk, "chunk_index": i})
                if (i + 1) % 50 == 0:
                    logger.info(f"Embedded {i+1}/{len(text_chunks)} chunks...")

            except Exception as e:
                logger.error(f"Error embedding chunk {i} from {filename}: {e}", exc_info=True)
                # Decide whether to skip the chunk or stop processing
                continue # Skip problematic chunk

        if not embeddings_list:
            logger.warning(f"No valid embeddings generated for {filename}. Nothing added to storage.")
            return {"message": f"Could not generate valid embeddings for {filename}."}

        embeddings_np = np.array(embeddings_list).astype('float32')
        logger.info(f"Generated embeddings of shape: {embeddings_np.shape}")

        # 4. Add to in-memory storage
        initial_count = len(STORED_EMBEDDINGS)
        STORED_EMBEDDINGS.extend(embeddings_list) # Store as list of numpy arrays
        STORED_METADATA.extend(chunk_metadata)
        logger.info(f"Added {len(embeddings_list)} vectors to in-memory storage. Total vectors: {len(STORED_EMBEDDINGS)}")

        return {
            "message": f"Successfully processed {filename}",
            "chunks_added": len(embeddings_list),
            "total_vectors_in_storage": len(STORED_EMBEDDINGS)
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except ImportError as e:
         logger.error(f"Import error: {e}. Make sure required libraries (PyPDF2, scikit-learn) are installed.", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Server configuration error: {e}")
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document {filename}: {str(e)}")

# Add a simple root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Steering Search Backend is running."}

# New endpoint for searching
@app.post("/search")
def search_documents(request: SearchRequest):
    global STORED_EMBEDDINGS, STORED_METADATA

    if MODEL is None or TOKENIZER is None or EMBED_FUNCTION is None:
        logger.error("Model not initialized. Cannot perform search.")
        raise HTTPException(status_code=500, detail="Server not ready. Model not initialized.")
    if not STORED_EMBEDDINGS:
        logger.info("No documents processed yet. Cannot perform search.")
        return {"message": "No documents have been processed yet.", "results": []}

    try:
        logger.info(f"Embedding search query: '{request.query}'")
        # Embed the query
        query_hidden_states = EMBED_FUNCTION(MODEL, TOKENIZER, request.query, TARGET_EMBEDDING_LAYER)

        # Convert query embedding to numpy float32 and average sequence dimension
        if _MLX_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_mlx':
            query_emb_np = np.array(query_hidden_states, dtype=np.float32)
        elif _TORCH_AVAILABLE and EMBED_FUNCTION.__name__ == 'embed_torch':
            query_emb_np = query_hidden_states.cpu().numpy().astype(np.float32)
        else:
            raise RuntimeError("Unknown embedding function type during search.")

        # Average sequence length dimension if necessary
        if query_emb_np.ndim == 3 and query_emb_np.shape[0] == 1:
            query_vector = np.mean(query_emb_np[0, :, :], axis=0)
        elif query_emb_np.ndim == 2:
            query_vector = np.mean(query_emb_np, axis=0)
        elif query_emb_np.ndim == 1:
            query_vector = query_emb_np
        else:
             raise ValueError(f"Unexpected query embedding shape: {query_emb_np.shape}")

        # Ensure query vector is 2D for cosine_similarity
        query_vector = query_vector.reshape(1, -1)

        # Prepare stored embeddings (already list of 1D numpy arrays)
        stored_vectors_np = np.array(STORED_EMBEDDINGS).astype('float32') # Shape (num_docs, dim)

        # Calculate Cosine Similarity
        similarities = cosine_similarity(query_vector, stored_vectors_np)[0] # Get the first row

        # Get top_k results
        top_k_indices = np.argsort(similarities)[-request.top_k:][::-1] # Get indices of top k scores

        results = []
        for idx in top_k_indices:
            results.append({"score": float(similarities[idx]), "metadata": STORED_METADATA[idx]})

        logger.info(f"Search completed. Found {len(results)} results for query: '{request.query}'")
        return {"query": request.query, "results": results}

    except Exception as e:
        logger.error(f"Error during search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")

# If you want to run this directly using uvicorn for testing:
# You would typically run this from the command line:
# uvicorn steering_backend.server:app --reload --port 8000
# Make sure you run it from the root directory (steerablesearch/)