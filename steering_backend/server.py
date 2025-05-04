'''
FastAPI server for frontend PDF embedding and search using FAISS.
'''

# Standard Library Imports
import glob
import json
import logging
import os
import random
import math # Add math import for ceiling division
from typing import List, Dict, Union, Any # Add List, Dict, Union, Any

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
import mlx.core # Add missing import

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
CHUNK_SIZE = 500  # Preferred characters per chunk
CHUNK_OVERLAP = 50  # Preferred overlap between chunks
MAX_CHUNKS = 100 # Maximum desired number of chunks per document
CHUNK_COUNT_TOLERANCE = 10 # Allow up to MAX_CHUNKS + TOLERANCE from initial chunking
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

# Define a model for a single steering parameter
class SteeringParam(BaseModel):
    feature_id: int
    strength: float

class SteeredSearchRequest(SearchRequest): # Inherit from SearchRequest
    # Replace single feature_id/strength with a list of parameters
    steering_params: List[SteeringParam] | None = None


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
    """Helper function for basic chunking with size and overlap."""
    if overlap >= size:
        # Prevent issues with overlap >= size
        logger.warning(f"Overlap ({overlap}) >= size ({size}). Chunking without overlap.")
        overlap = 0

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
             break # Exit if we've reached the end
        start += size - overlap
        # Ensure start doesn't go backwards if overlap is large relative to size (shouldn't happen with check above)
        if start < 0: start = 0
    return chunks

def process_and_chunk_text(text: str) -> list[str]:
    """
    Chunks text using CHUNK_SIZE and CHUNK_OVERLAP.
    If the number of chunks exceeds MAX_CHUNKS + CHUNK_COUNT_TOLERANCE,
    it re-chunks based on MAX_CHUNKS, trying to preserve CHUNK_OVERLAP.
    """
    total_length = len(text)
    if total_length == 0:
        return []

    # Attempt 1: Chunk with preferred size and overlap
    logger.info(f"Attempting chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    initial_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    initial_chunk_count = len(initial_chunks)

    # Check if the initial chunk count is within acceptable limits
    allowed_max_chunks = MAX_CHUNKS + CHUNK_COUNT_TOLERANCE
    if initial_chunk_count <= allowed_max_chunks:
        logger.info(
            f"Initial chunking yielded {initial_chunk_count} chunks "
            f"(<= allowed max {allowed_max_chunks}). Using this result."
        )
        return initial_chunks
    else:
        # Only re-chunk if significantly over the limit
        logger.warning(
            f"Initial chunking yielded {initial_chunk_count} chunks (> allowed max {allowed_max_chunks}). "
            f"Re-chunking based on total length ({total_length}) / {MAX_CHUNKS}, preserving overlap if possible."
        )

        # Attempt 2: Calculate new size based on MAX_CHUNKS, keep overlap
        new_size = math.ceil(total_length / MAX_CHUNKS)
        effective_overlap = CHUNK_OVERLAP # Try to use preferred overlap

        # Sanity check: If new_size is too small for the overlap, chunk without overlap
        if new_size <= effective_overlap:
             logger.warning(
                 f"Calculated new chunk size ({new_size}) is <= overlap ({effective_overlap}). "
                 f"Re-chunking with size {new_size} and NO overlap."
             )
             effective_overlap = 0 # Force overlap to 0
        else:
             logger.info(f"Re-chunking with calculated size={new_size}, overlap={effective_overlap}")

        # Perform the second chunking attempt using the basic helper
        final_chunks = chunk_text(text, new_size, effective_overlap)
        logger.info(f"Re-chunking yielded {len(final_chunks)} chunks.")
        return final_chunks

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
            # Assuming BASELINE_EMBED_FUNCTION returns a type directly convertible to NumPy (e.g., MLX array or list/NumPy already)
            chunk_activations = hidden_states
            emb_np = np.array(chunk_activations).astype(np.float32)

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

    # --- Check if already indexed --- #
    global STORED_METADATA
    # This is a linear scan; might be slow for very large metadata. Optimize if needed.
    if any(meta.get('source') == filename for meta in STORED_METADATA):
        logger.info(f"Skipping {filename}: Already found in stored metadata.")
        return {
            "success": True, # Indicate success as the file exists in the index
            "message": f"Skipped: {filename} already indexed.",
            "chunks_added": 0 # No chunks added *in this run*
        }
    # --- End Check --- #

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
        logger.info(f"Processing and chunking text (target size={CHUNK_SIZE}, max chunks={MAX_CHUNKS})...")
        text_chunks = process_and_chunk_text(full_text) # Use the new processing function
        if not text_chunks:
             logger.warning(f"Text chunking resulted in zero chunks for {filename}. Skipping.")
             return {"success": False, "message": f"Text chunking yielded no results for {filename}.", "chunks_added": 0}
        logger.info(f"Created {len(text_chunks)} actual chunks for {filename}.") # Log actual final number

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
def steered_search_documents(request: SteeredSearchRequest):
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

    # --- Get Base Activation ---
    try:
        logger.info(f"Getting base activation for query: '{request.query}'")
        # Use the SAE model's method to get activation at the correct layer
        base_activation_tensor = SAE_MODEL.get_activation(request.query)
        # Ensure it's on the correct device (matching SAE_MODEL.device)
        base_activation_tensor = base_activation_tensor.to(SAE_MODEL.device)
        logger.info(f"Base activation shape: {base_activation_tensor.shape}")

    except Exception as e:
        logger.error(f"Error getting base activation for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get base activation: {str(e)}")


    # --- Calculate Combined Steering Offset ---
    total_steering_offset = torch.zeros_like(base_activation_tensor) # Initialize offset tensor on the same device
    applied_steering_info = [] # Store details of features successfully applied

    if request.steering_params:
        logger.info(f"Processing {len(request.steering_params)} steering parameters...")
        valid_feature_ids = set(SAE_MODEL.explanations_df['feature'].unique()) if SAE_MODEL.explanations_df is not None else set()

        for param in request.steering_params:
            feature_id = param.feature_id
            strength = param.strength

            if feature_id not in valid_feature_ids:
                logger.warning(f"Requested steering feature {feature_id} not found or explanations not loaded. Skipping.")
                continue

            try:
                # Get max activation for scaling
                max_act = SAE_MODEL.get_feature_max_activation(feature_id)
                if max_act is None or max_act == 0: # Handle cases where max_act is invalid
                     logger.warning(f"Max activation for feature {feature_id} is invalid ({max_act}). Skipping feature.")
                     continue

                # Get the steering vector (decoder weight)
                steering_vector = SAE_MODEL.sae.W_dec[feature_id]
                steering_vector = steering_vector.to(total_steering_offset.device, dtype=total_steering_offset.dtype)

                # Calculate offset for this feature
                offset = strength * max_act * steering_vector
                total_steering_offset += offset

                # Log applied feature details
                explanation_rows = SAE_MODEL.explanations_df[SAE_MODEL.explanations_df['feature'] == feature_id]
                explanation = explanation_rows['description'].iloc[0] if not explanation_rows.empty else "Explanation not found."
                applied_steering_info.append({
                    "feature_id": feature_id,
                    "strength": strength,
                    "max_activation_used": max_act, # Include max_act used for scaling
                    "explanation": explanation
                })
                logger.info(f"Applied steering for feature {feature_id} with strength {strength:.2f} (max_act={max_act:.4f})")

            except Exception as e:
                logger.error(f"Error processing steering for feature {feature_id}: {e}", exc_info=True)
                # Decide whether to continue or raise an error for the whole request

    # --- Apply Offset and Perform Search ---
    try:
        # Add the combined offset to the base activation
        final_query_emb_tensor = base_activation_tensor + total_steering_offset
        logger.info(f"Final query embedding shape after steering: {final_query_emb_tensor.shape}")

        # Convert final tensor to numpy for FAISS
        # Ensure shape is (1, EMBEDDING_DIM)
        query_emb_np = final_query_emb_tensor.detach().numpy().astype('float32')
        if query_emb_np.shape != (1, EMBEDDING_DIM):
             # Add potential reshaping logic if needed (e.g., if pooling was missed in get_activation)
             if len(query_emb_np.shape) == 3 and query_emb_np.shape[0] == 1:
                 query_emb_np = np.mean(query_emb_np[0, :, :], axis=0).reshape(1, -1)
             if query_emb_np.shape != (1, EMBEDDING_DIM):
                 raise ValueError(f"Unexpected final query embedding shape after potential pooling: {query_emb_np.shape}. Expected: (1, {EMBEDDING_DIM})")

        # FAISS search
        logger.info(f"Searching index with {FAISS_INDEX.ntotal} vectors for top {request.top_k} results.")
        distances, indices = FAISS_INDEX.search(query_emb_np, request.top_k)

        # Process results
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    # Ensure metadata exists for the index
                    if 0 <= idx < len(STORED_METADATA):
                        results.append({
                            "score": float(distances[0][i]),
                            "metadata": STORED_METADATA[idx]
                        })
                    else:
                         logger.warning(f"FAISS index {idx} is out of bounds for STORED_METADATA (size {len(STORED_METADATA)}). Skipping result.")


        search_type = f"Steered ({len(applied_steering_info)} features)" if applied_steering_info else "Non-Steered"
        logger.info(f"{search_type} search completed. Found {len(results)} results.")
        # Update response structure
        return {
            "query": request.query,
            "steering_info": applied_steering_info, # Return list of applied features/strengths
            "results": results
        }

    except Exception as e:
        logger.error(f"Error during final embedding processing or FAISS search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")

# --- New Endpoint to Get Available Features ---
@app.get("/features")
def get_available_features():
    """Returns a list of available steering features and their descriptions."""
    global SAE_MODEL
    if SAE_MODEL is None or SAE_MODEL.explanations_df is None or SAE_MODEL.explanations_df.empty:
        logger.warning("Feature explanations not loaded or empty. Returning empty list.")
        return [] # Return empty list if not available

    try:
        # Select relevant columns and drop duplicates based on feature ID
        features_df = SAE_MODEL.explanations_df[['feature', 'description']].drop_duplicates(subset=['feature'])
        # Convert to the desired list of dictionaries format
        features_list = features_df.rename(columns={'feature': 'feature_id'}).to_dict('records')
        # Sort by feature_id for consistency
        features_list.sort(key=lambda x: x['feature_id'])
        logger.info(f"Returning {len(features_list)} available features.")
        return features_list
    except Exception as e:
        logger.error(f"Error retrieving features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving feature list.")

# If you want to run this directly using uvicorn for testing:
# You would typically run this from the command line:
# uvicorn steering_backend.server:app --reload --port 8000
# Make sure you run it from the root directory (steerablesearch/)
