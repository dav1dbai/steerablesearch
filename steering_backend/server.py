'''
FastAPI server for frontend PDF embedding and search using FAISS.
'''

# Standard Library Imports
import glob
import json
import logging
import os
import random

# Third-party Imports
import faiss
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends
from PyPDF2 import PdfReader
from pydantic import BaseModel

# Local Application Imports
from sae import SAEModel  # Import the new SAEModel class
from embed import (load_embedding_model,
                   embed_mlx, embed_torch, # Keep specific functions if needed for type check
                   _MLX_AVAILABLE, _TORCH_AVAILABLE) # Import availability flags
import torch # Need torch for tensor operations

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Configuration ---
# Using __file__ ensures paths are relative to this script's location
_SCRIPT_DIR = os.path.dirname(__file__)
ARXIV_DIR = os.path.join(_SCRIPT_DIR, "arxiv")
FAISS_DATA_DIR = os.path.join(_SCRIPT_DIR, "faiss_data") # New directory for index/metadata
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "arxiv_index.faiss") # Updated path
METADATA_PATH = FAISS_INDEX_PATH + ".meta.json" # Updated path
# TARGET_EMBEDDING_LAYER = 20  # Example layer, adjust as needed
# TARGET_EMBEDDING_LAYER = None # This will be set based on SAE layer
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
DEFAULT_STEERING_STRENGTH = 2.0 # Default strength for random steering

# --- Global Application State (Initialized on Startup) ---
# These are managed by FastAPI startup/shutdown events
SAE_MODEL: SAEModel | None = None # Single instance of our SAEModel
EMBEDDING_DIM = None # Dynamically determined from the model
TARGET_EMBEDDING_LAYER = None # Dynamically determined from SAE

# Baseline Model (for indexing and standard search)
BASELINE_MODEL = None
BASELINE_TOKENIZER = None
BASELINE_EMBED_FUNCTION = None

FAISS_INDEX = None     # faiss.Index
STORED_METADATA = []   # List[dict] - [{"source": str, "text": str, "chunk_index": int, "faiss_index": int}]


# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# --- FastAPI Application Instance ---
app = FastAPI(title="Steering Search Backend", version="0.1.0")

# --- Startup and Shutdown Logic ---

def _load_sae_model():
    """Loads the SAE model and determines embedding dimension."""
    global SAE_MODEL, EMBEDDING_DIM, TARGET_EMBEDDING_LAYER
    logger.info("Loading SAE model...")
    try:
        SAE_MODEL = SAEModel() # Instantiate the class
        if SAE_MODEL.model is None or SAE_MODEL.sae is None:
             raise RuntimeError("SAEModel initialization failed (model or SAE is None).")
        # Determine embedding dimension from the SAE's input dimension
        EMBEDDING_DIM = SAE_MODEL.sae.cfg.d_in
        # Set the target layer based on the loaded SAE
        TARGET_EMBEDDING_LAYER = SAE_MODEL.sae.cfg.hook_layer
        logger.info(f"SAEModel loaded. Target Layer: {TARGET_EMBEDDING_LAYER}, Embedding Dim: {EMBEDDING_DIM}")
        return True
    except Exception as e:
        logger.error(f"Fatal error loading SAE model: {e}", exc_info=True)
        return False

def _load_baseline_model():
    """Loads the baseline embedding model using embed.py."""
    global BASELINE_MODEL, BASELINE_TOKENIZER, BASELINE_EMBED_FUNCTION
    logger.info("Loading baseline embedding model (via embed.py)...")
    try:
        BASELINE_MODEL, BASELINE_TOKENIZER, BASELINE_EMBED_FUNCTION, model_name = load_embedding_model()
        if not all([BASELINE_MODEL, BASELINE_TOKENIZER, BASELINE_EMBED_FUNCTION]):
            raise RuntimeError("Failed to load one or more baseline model components.")
        logger.info(f"Baseline model ({model_name}) loaded successfully.")
        # Optionally, add validation for TARGET_EMBEDDING_LAYER against this model here
        # e.g., check max layers if possible
        return True
    except Exception as e:
        logger.error(f"Fatal error loading baseline embedding model: {e}", exc_info=True)
        return False

def _load_or_initialize_index_and_metadata():
    """Loads existing FAISS index and metadata or initializes new ones."""
    global FAISS_INDEX, STORED_METADATA

    if EMBEDDING_DIM is None:
        logger.error("Cannot load/initialize index: Embedding dimension not set.")
        return False

    # --- Ensure FAISS data directory exists --- #
    os.makedirs(FAISS_DATA_DIR, exist_ok=True)
    # --- End directory check ---

    index_loaded = False
    metadata_loaded = False

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        logger.info(f"Found existing FAISS index ({FAISS_INDEX_PATH}) and metadata ({METADATA_PATH}). Attempting to load.")
        try:
            loaded_index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"Loaded FAISS index with {loaded_index.ntotal} vectors.")

            # Check index dimension against model dimension
            if loaded_index.d != EMBEDDING_DIM:
                logger.error(f"Existing index dimension ({loaded_index.d}) does not match model dimension ({EMBEDDING_DIM}). Discarding loaded index.")
            else:
                # Index dimension matches, now try loading metadata
                with open(METADATA_PATH, 'r') as f:
                    loaded_metadata = json.load(f)
                logger.info(f"Loaded metadata with {len(loaded_metadata)} entries.")

                # --- Crucial Validation --- #
                if loaded_index.ntotal != len(loaded_metadata):
                    logger.warning(f"Mismatch between FAISS index size ({loaded_index.ntotal}) and metadata entries ({len(loaded_metadata)}). Resetting both.")
                else:
                    # Success! Assign to globals
                    FAISS_INDEX = loaded_index
                    STORED_METADATA = loaded_metadata
                    index_loaded = True
                    metadata_loaded = True
                    logger.info("Successfully loaded and validated existing index and metadata.")

        except Exception as e:
            logger.error(f"Error loading FAISS index or metadata: {e}. Will create new ones.", exc_info=True)
            # Clean up potentially corrupted files if loading failed
            if os.path.exists(FAISS_INDEX_PATH):
                try: os.remove(FAISS_INDEX_PATH)
                except OSError: logger.warning(f"Could not remove potentially corrupt index file: {FAISS_INDEX_PATH}")
            if os.path.exists(METADATA_PATH):
                try: os.remove(METADATA_PATH)
                except OSError: logger.warning(f"Could not remove potentially corrupt metadata file: {METADATA_PATH}")

    # If loading failed or files didn't exist, create new index and metadata store
    if not index_loaded or not metadata_loaded:
        logger.info("Initializing new FAISS index and metadata store.")
        FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
        STORED_METADATA = []

    return True # Indicate successful initialization (even if new)

@app.on_event("startup")
def startup_event():
    """FastAPI startup event: Load model, determine dimension, load/init index."""
    logger.info("--- Server Startup Sequence Initiated ---")
    # Load SAE first to determine layer and dimension
    if not _load_sae_model():
         raise RuntimeError("Failed to determine embedding dimension. Server cannot start.")

    # Load baseline model (for indexing/standard search)
    if not _load_baseline_model():
         raise RuntimeError("Failed to load baseline embedding model. Server cannot start.")

    if not _load_or_initialize_index_and_metadata():
        # This function currently always returns True, but for future-proofing:
        raise RuntimeError("Failed to load or initialize FAISS index/metadata. Server cannot start.")

    logger.info("--- Server Startup Sequence Complete ---")


@app.on_event("shutdown")
def shutdown_event():
    """FastAPI shutdown event: Save FAISS index and metadata."""
    global FAISS_INDEX, STORED_METADATA
    logger.info("--- Server Shutdown Sequence Initiated ---")

    if FAISS_INDEX is None or not isinstance(FAISS_INDEX, faiss.Index):
        logger.info("FAISS index not initialized. Nothing to save.")
        return

    # Check for consistency before saving
    if FAISS_INDEX.ntotal > 0 and len(STORED_METADATA) != FAISS_INDEX.ntotal:
        logger.error(f"CRITICAL: Mismatch between index size ({FAISS_INDEX.ntotal}) and metadata size ({len(STORED_METADATA)}) on shutdown. Aborting save to prevent corruption.")
        return

    if FAISS_INDEX.ntotal == 0:
        logger.info("FAISS index is empty. Cleaning up any existing files.")
        if os.path.exists(FAISS_INDEX_PATH):
            try: os.remove(FAISS_INDEX_PATH); logger.info(f"Removed file: {FAISS_INDEX_PATH}")
            except OSError as e: logger.warning(f"Could not remove file {FAISS_INDEX_PATH}: {e}")
        if os.path.exists(METADATA_PATH):
            try: os.remove(METADATA_PATH); logger.info(f"Removed file: {METADATA_PATH}")
            except OSError as e: logger.warning(f"Could not remove file {METADATA_PATH}: {e}")
        return

    # Save FAISS index
    try:
        logger.info(f"Saving FAISS index ({FAISS_INDEX.ntotal} vectors) to {FAISS_INDEX_PATH}...")
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
        logger.info("FAISS index saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
        # Avoid saving metadata if index saving failed to maintain consistency
        logger.info("--- Server Shutdown Sequence Failed (Index Save Error) ---")
        return

    # Save Metadata
    try:
        logger.info(f"Saving metadata ({len(STORED_METADATA)} entries) to {METADATA_PATH}...")
        with open(METADATA_PATH, 'w') as f:
            json.dump(STORED_METADATA, f) # Save compact JSON
        logger.info("Metadata saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}", exc_info=True)
        # Consider rolling back index save? For now, log error.
        logger.info("--- Server Shutdown Sequence Failed (Metadata Save Error) ---")
        return

    logger.info("--- Server Shutdown Sequence Complete ---")


# --- Text Processing Utilities ---

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

# --- Helper function for processing a single PDF ---

# --- Embedding and Indexing Logic ---

def _read_pdf_text(file_path: str) -> str:
    """Extracts text content from a PDF file."""
    logger.info(f"Reading PDF: {os.path.basename(file_path)}")
    try:
        reader = PdfReader(file_path)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        logger.info(f"Extracted {len(full_text)} characters.")
        return full_text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}", exc_info=True)
        raise IOError(f"Failed to read PDF: {os.path.basename(file_path)}") from e

def _embed_chunks(text_chunks: list[str], filename: str) -> tuple[list[np.ndarray], list[dict]]:
    """
    Embeds text chunks using the global SAE_MODEL's activation layer.
    Averages the activations across the sequence length for each chunk.
    """
    embeddings_list = []
    chunk_metadata_list = []
    if not SAE_MODEL or not SAE_MODEL.model or not SAE_MODEL.sae:
         logger.error("SAE_MODEL not initialized during embedding.")
         return [], [] # Return empty if model isn't ready

    # Use the baseline embedding function
    if not BASELINE_MODEL or not BASELINE_TOKENIZER or not BASELINE_EMBED_FUNCTION:
         logger.error("Baseline model not initialized during embedding.")
          return [], [] # Return empty if model isn't ready

    logger.info(f"Embedding {len(text_chunks)} chunks for {filename} using baseline model at layer {TARGET_EMBEDDING_LAYER}...")

    for i, chunk in enumerate(text_chunks):
        if not chunk.strip():
            continue # Skip empty chunks

        try:
            # Use the baseline embedding function
            hidden_states = BASELINE_EMBED_FUNCTION(BASELINE_MODEL, BASELINE_TOKENIZER, chunk, TARGET_EMBEDDING_LAYER)

            # Activations shape: [batch=1, sequence_length, d_model]
            chunk_activations = hidden_states
            emb_np = chunk_activations.cpu().numpy().astype(np.float32)

            # Average across the sequence length dimension -> (embedding_dim,)
            # Handle potential empty sequences if needed (though unlikely with text chunks)
            if emb_np.shape[1] > 0: # Check sequence length > 0
                avg_embedding = np.mean(emb_np[0, :, :], axis=0)
            elif emb_np.ndim == 1:
                avg_embedding = emb_np
            else:
                logger.warning(f"Unexpected embedding shape {emb_np.shape} for chunk {i} of {filename}. Skipping.")
                continue

            # Validate embedding dimension
            if avg_embedding.shape != (EMBEDDING_DIM,):
                 logger.warning(f"Embedding dimension mismatch ({avg_embedding.shape[0]} vs {EMBEDDING_DIM}) for chunk {i} of {filename}. Skipping.")
                 continue

            embeddings_list.append(avg_embedding)
            # Metadata is simpler here, faiss index added later
            chunk_metadata_list.append({
                "source": filename,
                "text": chunk,
                "chunk_index": i
             })

        except Exception as e:
            logger.error(f"Error embedding chunk {i} from {filename}: {e}", exc_info=True)
            # Skip problematic chunk, maybe add to failed list later?
            continue

    return embeddings_list, chunk_metadata_list

def _add_embeddings_to_index(embeddings: list[np.ndarray], metadata: list[dict], filename: str) -> int:
    """Adds embeddings and corresponding metadata to the global FAISS index and store."""
    global FAISS_INDEX, STORED_METADATA

    if not embeddings:
        logger.warning(f"No valid embeddings generated for {filename}. Nothing to add.")
        return 0

    embeddings_np = np.array(embeddings).astype('float32')
    if embeddings_np.shape[1] != EMBEDDING_DIM:
         logger.error(f"FATAL: Dimension mismatch just before adding to index for {filename}. Expected {EMBEDDING_DIM}, got {embeddings_np.shape[1]}. Aborting add.")
         # This case should ideally be caught earlier, but as a safeguard.
         return 0

    try:
        start_index = FAISS_INDEX.ntotal
        FAISS_INDEX.add(embeddings_np)
        added_count = embeddings_np.shape[0]

        # Add FAISS index to metadata *after* successful add to index
        for i, meta in enumerate(metadata):
            meta['faiss_index'] = start_index + i
        STORED_METADATA.extend(metadata)

        logger.info(f"Added {added_count} vectors to FAISS index for {filename}. Total vectors: {FAISS_INDEX.ntotal}")
        return added_count
    except Exception as e:
        logger.error(f"Error adding vectors to FAISS index for {filename}: {e}", exc_info=True)
        # Attempt to rollback metadata addition is complex, log clearly.
        logger.warning(f"Index add failed for {filename}. Associated metadata was not stored.")
        return 0


def _process_pdf_file(file_path: str) -> dict:
    """Reads, chunks, embeds, and indexes a single PDF file."""
    filename = os.path.basename(file_path)
    logger.info(f"--- Starting processing for: {filename} ---")

    # Pre-check for server readiness (should always be true if called after startup)
    if BASELINE_MODEL is None or FAISS_INDEX is None:
        logger.error("Model or FAISS index not initialized during processing call.")
        # Indicate a server state issue, not just a file issue
        raise RuntimeError("Server components not ready. Check startup logs.")

    try:
        # 1. Read PDF
        full_text = _read_pdf_text(file_path)
        if not full_text.strip():
            logger.warning(f"No text extracted from {filename}. Skipping.")
            return {"success": False, "message": f"No text content found in {filename}.", "chunks_added": 0}

        # 2. Chunk Text
        logger.info(f"Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not text_chunks:
             logger.warning(f"Text chunking resulted in zero chunks for {filename}. Skipping.")
             return {"success": False, "message": f"Text chunking yielded no results for {filename}.", "chunks_added": 0}
        logger.info(f"Created {len(text_chunks)} chunks.")

        # 3. Embed Chunks
        embeddings_list, chunk_metadata = _embed_chunks(text_chunks, filename)
        if not embeddings_list:
             logger.warning(f"Embedding yielded no results for {filename}. Skipping add to index.")
             return {"success": False, "message": f"Embedding process failed to produce vectors for {filename}.", "chunks_added": 0}

        # 4. Add to FAISS Index and metadata store
        chunks_added_count = _add_embeddings_to_index(embeddings_list, chunk_metadata, filename)

        if chunks_added_count > 0:
             logger.info(f"--- Successfully finished processing: {filename} ({chunks_added_count} chunks added) --- ")
             return {
                "success": True,
                "message": f"Successfully processed {filename}",
                "chunks_added": chunks_added_count,
                "total_vectors_in_index": FAISS_INDEX.ntotal
            }
        else:
             logger.warning(f"--- Finished processing {filename}, but no chunks were added to the index. --- ")
             # Distinguish between file read errors (caught below) and embedding/adding failures
             return {
                "success": False,
                "message": f"Processing completed for {filename}, but no vectors were added to the index (check logs for embedding/add errors).",
                "chunks_added": 0
            }

    except (IOError, PdfReader.errors.PdfReadError) as e: # Catch specific PDF read errors
        logger.error(f"Failed to read or process PDF {filename}: {e}", exc_info=False) # Keep log cleaner
        return {"success": False, "message": f"Failed to read/process PDF file {filename}: {e}", "chunks_added": 0}
    except ImportError as e:
         # This indicates a missing dependency, should be caught at startup but good to have.
         logger.error(f"Import error during processing {filename}: {e}. Is PyPDF2 installed?", exc_info=True)
         raise RuntimeError(f"Server configuration error (missing import): {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error processing document {filename}: {e}", exc_info=True)
        # Return failure for this specific file
        return {"success": False, "message": f"Unexpected error processing {filename}: {str(e)}", "chunks_added": 0}

def index_all_arxiv_pdfs():
    """Finds all PDFs in ARXIV_DIR and processes them, skipping existing ones."""
    global STORED_METADATA # Need to read global metadata

    logger.info(f"Starting indexing of all PDFs in {ARXIV_DIR}...")

    # Build a set of filenames already in the metadata
    processed_filenames = set()
    if STORED_METADATA:
        processed_filenames = set(meta['source'] for meta in STORED_METADATA if 'source' in meta)
        logger.info(f"Found {len(processed_filenames)} previously processed files in metadata.")
    else:
        logger.info("No existing metadata found.")


    pdf_files = glob.glob(os.path.join(ARXIV_DIR, "**", "*.pdf"), recursive=True)

    if not pdf_files:
        logger.warning(f"No PDF files found in {ARXIV_DIR}. Nothing to index.")
        return {"message": f"No PDF files found in {ARXIV_DIR}.", "processed_count": 0, "failed_count": 0}

    processed_count = 0
    failed_count = 0
    total_chunks_added = 0
    skipped_count = 0 # Track skipped files

    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    for i, file_path in enumerate(pdf_files):
        filename = os.path.basename(file_path)

        # --- Skip Check --- #
        if filename in processed_filenames:
            logger.info(f"Skipping file {i+1}/{len(pdf_files)}: {filename} (already indexed)." )
            skipped_count += 1
            continue
        # --- End Skip Check --- #

        logger.info(f"Processing file {i+1}/{len(pdf_files)}: {filename}")
        try:
            result = _process_pdf_file(file_path)
            if result["success"]:
                processed_count += 1
                total_chunks_added += result.get("chunks_added", 0)
            else:
                failed_count += 1
                logger.error(f"Failed to process {filename}: {result['message']}")
        except RuntimeError as e:
            # Stop processing if there's a server state issue
            logger.error(f"Critical runtime error during bulk indexing: {e}. Aborting further processing.", exc_info=True)
            return {
                "message": f"Critical error during indexing: {e}. Aborted.",
                "processed_count": processed_count,
                "failed_count": failed_count + (len(pdf_files) - i),
                "total_chunks_added": total_chunks_added
            }
        except Exception as e:
            # Catch unexpected errors for a specific file
            failed_count += 1
            logger.error(f"Unexpected error processing {filename} during bulk index: {e}", exc_info=True)

    summary_message = (f"Finished indexing. Processed: {processed_count}, "
                       f"Skipped: {skipped_count}, Failed: {failed_count}. "
                       f"Total chunks added in this run: {total_chunks_added}."
                      )
    logger.info(summary_message)
    return {
        "message": summary_message,
        "processed_count": processed_count,
        "skipped_count": skipped_count, # Add skipped count
        "failed_count": failed_count,
        "total_chunks_added_this_run": total_chunks_added, # Clarify this is for the current run
        "total_vectors_in_index": FAISS_INDEX.ntotal if FAISS_INDEX else 0
    }

# --- FastAPI Endpoints ---

@app.post("/process/{filename}")
def process_document(filename: str):
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(ARXIV_DIR, filename)
    logger.info(f"Received request to process document: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found via API request: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        result = _process_pdf_file(file_path)
        if result["success"]:
            return result
        else:
            # Use 500 for processing failure on a specific file,
            # as the server itself is okay, but this file failed.
            raise HTTPException(status_code=500, detail=result["message"])
    except RuntimeError as e:
        # RuntimeErrors from _process_pdf_file indicate server state issues
         logger.error(f"Runtime error processing {filename}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch any unexpected errors from the helper
        logger.error(f"Unexpected error processing {filename} via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred processing {filename}: {str(e)}")


# New endpoint to trigger indexing all PDFs in the arxiv directory
@app.post("/index_all")
def trigger_index_all(background_tasks: BackgroundTasks): # Add BackgroundTasks dependency
    """Triggers the indexing of all PDFs in the background."""
    # Check if server is ready before scheduling the task
    if BASELINE_MODEL is None or FAISS_INDEX is None:
        logger.error("Cannot start indexing task. Server components not ready.")
        # Return error immediately, don't schedule task
        raise HTTPException(status_code=503, detail="Server not fully initialized. Cannot start indexing.") # 503 Service Unavailable

    logger.info("Received request to index all documents. Scheduling background task...")
    try:
        # Add the potentially long-running task to FastAPI's background tasks
        background_tasks.add_task(index_all_arxiv_pdfs)
        # Return immediately to the client
        return {"message": "Indexing process started in the background. Check server logs for progress and completion status."}
    except Exception as e:
        # This catch block is less likely to be hit now, as add_task itself is quick,
        # but kept as a safeguard against unforeseen issues during task scheduling.
        logger.error(f"Error scheduling the indexing task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to schedule indexing task: {str(e)}")

# Add a simple root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Steering Search Backend is running."}

# New endpoint for searching
@app.post("/search")
def search_documents(request: SearchRequest):
    global FAISS_INDEX, STORED_METADATA

    if BASELINE_MODEL is None or FAISS_INDEX is None:
        logger.error("Model or FAISS index not initialized. Cannot perform search.")
        raise HTTPException(status_code=500, detail="Server not ready. Model or index not initialized.")
    if FAISS_INDEX.ntotal == 0:
        logger.info("No documents processed yet. Cannot perform search.")
        return {"message": "No documents have been processed yet.", "results": []}
    if not request.query.strip():
        logger.info("Received empty search query.")
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        logger.info(f"Embedding search query: '{request.query}'")

        # Embed query using baseline model
        query_hidden_states = BASELINE_EMBED_FUNCTION(
            BASELINE_MODEL, BASELINE_TOKENIZER, request.query, TARGET_EMBEDDING_LAYER
        )

        # Convert query embedding to numpy float32 and average sequence dimension
        if _MLX_AVAILABLE and BASELINE_EMBED_FUNCTION.__name__ == 'embed_mlx':
            query_emb_np_full = np.array(query_hidden_states, dtype=np.float32)
        elif _TORCH_AVAILABLE and BASELINE_EMBED_FUNCTION.__name__ == 'embed_torch':
            query_emb_np_full = query_hidden_states.cpu().numpy().astype(np.float32)
        else:
            raise RuntimeError("Unknown embedding function type during search.")

        # Average sequence dimension -> (embedding_dim,)
        if query_emb_np_full.shape[1] > 0:
             query_vector_avg = np.mean(query_emb_np_full[0, :, :], axis=0)
        else:
             logger.warning(f"Query '{request.query}' resulted in empty sequence embedding. Using zeros.")
             query_vector_avg = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Ensure the final vector has the correct shape for FAISS
        if query_vector_avg.shape != (EMBEDDING_DIM,):
             raise ValueError(f"Unexpected averaged query embedding shape: {query_vector_avg.shape}. Expected: ({EMBEDDING_DIM},)")

        # Reshape for FAISS search (needs 2D array: [1, embedding_dim])
        query_emb_np_final = query_vector_avg.reshape(1, -1)

        # Perform FAISS search
        logger.info(f"Searching index with {FAISS_INDEX.ntotal} vectors for top {request.top_k} results.")
        distances, indices = FAISS_INDEX.search(query_emb_np_final, request.top_k)

        # Process results
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1: # FAISS returns -1 for padding if fewer results than k are found
                    results.append({
                        "score": float(distances[0][i]), # For L2 index, distance is squared L2. Lower is better.
                        "metadata": STORED_METADATA[idx]
                    })

        logger.info(f"Search completed. Found {len(results)} results for query: '{request.query}'")
        return {"query": request.query, "results": results}

    except Exception as e:
        logger.error(f"Error during search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")

# --- Steered Search Endpoint ---

@app.post("/steered_search")
def steered_search_documents(request: SearchRequest): # Using SearchRequest for now
    global FAISS_INDEX, STORED_METADATA, SAE_MODEL

    if SAE_MODEL is None or FAISS_INDEX is None:
        logger.error("Model or FAISS index not initialized. Cannot perform steered search.")
        raise HTTPException(status_code=500, detail="Server not ready. Model or index not initialized.")
    if FAISS_INDEX.ntotal == 0:
        logger.info("No documents processed yet. Cannot perform steered search.")
        return {"message": "No documents have been processed yet.", "results": []}
    if not request.query.strip():
        logger.info("Received empty search query for steered search.")
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    # --- Random Feature Selection ---
    feature_id = None
    explanation = "No features available for steering."
    try:
        if SAE_MODEL.explanations_df is not None and not SAE_MODEL.explanations_df.empty:
            available_features = SAE_MODEL.explanations_df['feature'].unique()
            if len(available_features) > 0:
                feature_id = random.choice(available_features)
                # Get the first explanation for this feature
                explanation_rows = SAE_MODEL.explanations_df[SAE_MODEL.explanations_df['feature'] == feature_id]
                if not explanation_rows.empty:
                    explanation = explanation_rows['description'].iloc[0]
                else:
                    explanation = "Explanation not found for selected feature."
                logger.info(f"Randomly selected feature {feature_id} for steering. Explanation: {explanation}")
            else:
                 logger.warning("Explanations DataFrame is loaded but contains no unique features.")
        else:
            logger.warning("Cannot select random feature: Explanations DataFrame not loaded or empty.")

    except Exception as e:
         logger.error(f"Error selecting random feature: {e}", exc_info=True)
         # Continue without steering if feature selection fails
         feature_id = None
         explanation = "Error selecting steering feature."

    # --- Perform Search (Steered if feature available, otherwise fallback to normal) ---
    try:
        if feature_id is not None:
            logger.info(f"Embedding steered query: '{request.query}', Feature: {feature_id}, Strength: {DEFAULT_STEERING_STRENGTH}")
            query_activation_tensor = SAE_MODEL.get_steered_activation(
                request.query, feature_id, DEFAULT_STEERING_STRENGTH
            )
        else:
            logger.warning("Performing non-steered search as no feature was selected.")
            query_activation_tensor = SAE_MODEL.get_activation(request.query)

        # Convert tensor to numpy for FAISS
        query_emb_np = query_activation_tensor.cpu().numpy().astype('float32')
        if query_emb_np.shape != (1, EMBEDDING_DIM):
            query_emb_np = query_activation_tensor.cpu().numpy().astype('float32')
            if query_emb_np.shape != (1, EMBEDDING_DIM):
                raise ValueError(f"Unexpected query activation shape: {query_emb_np.shape}. Expected: (1, {EMBEDDING_DIM})")

        # FAISS search
        logger.info(f"Searching index with {FAISS_INDEX.ntotal} vectors for top {request.top_k} results.")
        distances, indices = FAISS_INDEX.search(query_emb_np, request.top_k)

        # Process results
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append({
                        "score": float(distances[0][i]),
                        "metadata": STORED_METADATA[idx]
                    })

        logger.info(f"Steered search completed. Found {len(results)} results.")
        return {
            "query": request.query,
            "steering_info": {
                "feature_id": feature_id if feature_id is not None else "None",
                "strength": DEFAULT_STEERING_STRENGTH if feature_id is not None else 0,
                "explanation": explanation
            },
            "results": results
        }

    except Exception as e:
        logger.error(f"Error during steered search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform steered search: {str(e)}")

# If you want to run this directly using uvicorn for testing:
# You would typically run this from the command line:
# uvicorn steering_backend.server:app --reload --port 8000
# Make sure you run it from the root directory (steerablesearch/)
