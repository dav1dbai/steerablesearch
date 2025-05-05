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
import re # For citation and noise filtering
from typing import List, Dict, Union, Any, Optional, Tuple

# Third-party Imports
import faiss
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from pydantic import BaseModel

# Local Application Imports
from sae import SAEModel  # Import the new SAEModel class
from llm import ClaudeAPIClient  # Import Claude API client
from embed import (load_embedding_model,
                   embed_mlx, embed_torch, # Keep specific functions if needed for type check
                   _MLX_AVAILABLE, _TORCH_AVAILABLE) # Import availability flags
import torch # Need torch for tensor operations
import mlx.core # Add missing import
import dotenv

dotenv.load_dotenv()

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
ARXIV_METADATA_PATH = os.path.join(ARXIV_DIR, "arxiv_metadata.json") # Paper metadata with titles
# TARGET_EMBEDDING_LAYER = 20  # Example layer, adjust as needed
# TARGET_EMBEDDING_LAYER = None # This will be set based on SAE layer
CHUNK_SIZE = 500  # Preferred characters per chunk
CHUNK_OVERLAP = 50  # Preferred overlap between chunks
MAX_CHUNKS = 100 # Maximum desired number of chunks per document
CHUNK_COUNT_TOLERANCE = 10 # Allow up to MAX_CHUNKS + TOLERANCE from initial chunking
DEFAULT_STEERING_STRENGTH = 2.0 # Default strength for random steering
MAX_FEATURES_TO_MATCH = 20  # Maximum number of features to pass to Claude for matching
MAX_AUTO_STEERING_FEATURES = 5  # Maximum number of features to use for auto-steering
MAX_STEERING_STRENGTH = 8.0  # Maximum steering strength for auto-steering
ARXIV_BASE_URL = "https://arxiv.org/abs/"  # Base URL for arXiv papers

# --- Global Application State (Initialized on Startup) ---
# These are managed by FastAPI startup/shutdown events
SAE_MODEL: SAEModel | None = None # Single instance of our SAEModel
CLAUDE_CLIENT: ClaudeAPIClient | None = None # Single instance of Claude API client
EMBEDDING_DIM = None # Dynamically determined from the model
TARGET_EMBEDDING_LAYER = None # Dynamically determined from SAE

# Baseline Model (for indexing and standard search)
BASELINE_MODEL = None
BASELINE_TOKENIZER = None
BASELINE_EMBED_FUNCTION = None

FAISS_INDEX = None     # faiss.Index
STORED_METADATA = []   # List[dict] - [{"source": str, "text": str, "chunk_index": int, "faiss_index": int}]
PAPER_METADATA = {}    # Dict[str, Dict] - {"arxiv_id": {"title": str, "authors": str, ...}}


# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_noise: bool = True  # Option to filter out citation noise
    rewrite_query: bool = False  # Option to use Claude to rewrite the query

# Define a model for a single steering parameter
class SteeringParam(BaseModel):
    feature_id: int
    strength: float

class SteeredSearchRequest(SearchRequest): # Inherit from SearchRequest
    # Replace single feature_id/strength with a list of parameters
    steering_params: List[SteeringParam] | None = None

class AutoSteeredSearchRequest(SearchRequest):
    # For auto-steered search, we don't need steering parameters
    # But we might want additional control parameters
    max_features: int = MAX_AUTO_STEERING_FEATURES
    max_strength: float = MAX_STEERING_STRENGTH


# --- FastAPI Application Instance ---
app = FastAPI(title="Steering Search Backend", version="0.1.0")

# Import CORS middleware
try:
    from middleware import add_cors_middleware
    app = add_cors_middleware(app)
except ImportError:
    # Handle case when middleware module is not found
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for easier development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("Using default CORS settings")

# --- Startup and Shutdown Logic ---

def _load_paper_metadata():
    """Load paper metadata from JSON file if available."""
    global PAPER_METADATA
    if os.path.exists(ARXIV_METADATA_PATH):
        try:
            with open(ARXIV_METADATA_PATH, 'r') as f:
                raw_metadata = json.load(f)
            
            # Copy raw metadata to global variable
            PAPER_METADATA = {}
            
            # Process and normalize IDs for better matching
            for paper_id, metadata in raw_metadata.items():
                # Keep the original entry
                PAPER_METADATA[paper_id] = metadata
                
                # Add entries with and without version suffixes for better matching
                if 'v' in paper_id and paper_id.split('v')[0] != paper_id:
                    base_id = paper_id.split('v')[0]
                    if base_id not in PAPER_METADATA:
                        PAPER_METADATA[base_id] = metadata.copy()
                        logger.info(f"Added base ID entry: {base_id} from {paper_id}")
                
                # Add entries with and without dots for better matching
                if '.' in paper_id:
                    no_dot_id = paper_id.replace('.', '')
                    if no_dot_id not in PAPER_METADATA:
                        PAPER_METADATA[no_dot_id] = metadata.copy()
                        logger.info(f"Added no-dot ID entry: {no_dot_id} from {paper_id}")
            
            # Verify the loaded data
            title_count = sum(1 for paper_id, metadata in PAPER_METADATA.items() 
                             if metadata.get('title') and len(metadata.get('title', '')) > 3)
            
            logger.info(f"Loaded metadata for {len(raw_metadata)} papers, expanded to {len(PAPER_METADATA)} entries ({title_count} with valid titles).")
            
            # Sample some entries to verify
            for i, (paper_id, metadata) in enumerate(list(PAPER_METADATA.items())[:10]):
                title = metadata.get('title', 'No title')
                if len(title) > 50:
                    title = title[:50] + "..."
                logger.info(f"Sample paper {i+1}: ID={paper_id}, Title={title}")
            
            # Simulates a search to check what would happen
            test_ids = ["2504.14879", "2504.14879v1", "250414879", "250414879v1"]
            logger.info("Testing ID lookup for common formats:")
            for test_id in test_ids:
                if test_id in PAPER_METADATA:
                    logger.info(f"✓ Found metadata for {test_id}: {PAPER_METADATA[test_id].get('title', '')[:30]}...")
                else:
                    logger.warning(f"✗ No metadata found for {test_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading paper metadata: {e}", exc_info=True)
            PAPER_METADATA = {}
            return False
    else:
        logger.warning(f"Paper metadata file not found at {ARXIV_METADATA_PATH}. No paper titles will be available.")
        PAPER_METADATA = {}
        return False

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

def _load_claude_client():
    """Loads the Claude API client."""
    global CLAUDE_CLIENT
    logger.info("Initializing Claude API client...")
    try:
        CLAUDE_CLIENT = ClaudeAPIClient()
        # Check if API key is available
        if not CLAUDE_CLIENT.api_key:
            logger.warning("Claude API key not found. Auto-steering will not be available.")
            return False
        logger.info("Claude API client initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error initializing Claude API client: {e}", exc_info=True)
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

    # Initialize Claude API client
    _load_claude_client()  # Non-critical, don't raise exception if fails

    if not _load_or_initialize_index_and_metadata():
        # This function currently always returns True, but for future-proofing:
        raise RuntimeError("Failed to load or initialize FAISS index/metadata. Server cannot start.")
        
    # Load paper metadata (titles, authors, etc.)
    _load_paper_metadata()  # Non-critical, don't raise exception if fails

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

    # Extract arXiv ID from filename
    filename_base = os.path.splitext(filename)[0]  # Remove extension
    
    # Try to preserve version info but still get the base ID
    match = re.match(r'^(.*?)(v\d+)?$', filename_base)
    if match:
        base_arxiv_id = match.group(1)
        version = match.group(2) or ''
    else:
        base_arxiv_id = filename_base
        version = ''
        
    # Full ID with version (if any)
    arxiv_id = base_arxiv_id + version
    
    # Try to get paper title from metadata - be very explicit in logging
    logger.info(f"Looking for title for paper: filename={filename}, base_id={base_arxiv_id}, full_id={arxiv_id}")
    
    # Initialize paper title
    paper_title = ""
    
    # Try different ID variations
    variations = [
        arxiv_id,  # Full ID with version
        base_arxiv_id,  # Base ID without version
    ]
    
    # Add variations with/without dots
    if '.' in base_arxiv_id:
        variations.append(base_arxiv_id.replace('.', ''))  # No dots
        variations.append(arxiv_id.replace('.', ''))  # No dots with version
    elif len(base_arxiv_id) >= 8:  # Looks like it might be YYMMNNNNN format
        # Try adding a dot after first 4 chars
        dotted_base = f"{base_arxiv_id[:4]}.{base_arxiv_id[4:]}"
        dotted_full = dotted_base + version
        variations.append(dotted_base)
        variations.append(dotted_full)
    
    # Try each variation
    for var_id in variations:
        if var_id in PAPER_METADATA:
            title = PAPER_METADATA[var_id].get('title', '')
            if title and len(title) > 5:
                paper_title = title
                logger.info(f"✓ SUCCESS! Found title using ID {var_id}: '{paper_title}'")
                break
            else:
                logger.warning(f"Found metadata entry for {var_id} but title is invalid: '{title}'")
    
    # Log if no title found
    if not paper_title:
        logger.warning(f"❌ No title found for any variation of {arxiv_id}. Tried: {variations}")
        # Dump a few metadata keys to help debug
        sample_keys = list(PAPER_METADATA.keys())[:5]
        logger.info(f"First few metadata keys: {sample_keys}")

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
            # Enhanced metadata with arXiv ID, paper URL, and title
            chunk_metadata_list.append({
                "source": filename,
                "text": chunk,
                "chunk_index": i,
                "arxiv_id": arxiv_id,
                "paper_url": f"{ARXIV_BASE_URL}{arxiv_id}",
                "paper_title": paper_title
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

# --- Helper functions for paper result formatting and filtering ---

def is_citation_or_noise(text: str) -> bool:
    """
    Legacy function for determining if a text chunk is primarily citations or noise.
    Kept for backward compatibility but not used in the main search flow.
    
    Returns:
        bool: True if the chunk contains primarily citations or noise, False otherwise
    """
    # Common patterns for citations
    citation_patterns = [
        r"\[\d+\]",       # [1], [2], etc.
        r"\(\d{4}\)",     # (2020), (2021), etc.
        r"et al\.",       # et al.
        r"References",    # References section header
        r"Bibliography",  # Bibliography section header
    ]
    
    # Check for high density of citation patterns
    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, text))
    
    # If text has a high density of citations (more than 5 per 500 chars)
    if citation_count > 5 and len(text) < 500:
        return True
        
    # Check if the chunk is mostly references (starts with a number and dash/dot)
    lines = text.strip().split('\n')
    reference_line_count = 0
    for line in lines:
        if re.match(r"^\s*\d+[\.\)]", line.strip()):
            reference_line_count += 1
    
    # If more than 50% of lines look like references
    if reference_line_count > len(lines) * 0.5 and len(lines) > 2:
        return True
        
    # Check for chunks that are mostly whitespace or very short
    content_ratio = len(text.strip()) / max(len(text), 1)
    if content_ratio < 0.3 or len(text.strip()) < 50:
        return True
        
    return False

async def filter_content_async(text: str, client=None) -> dict:
    """
    Asynchronous wrapper around CLAUDE_CLIENT.filter_content.
    This allows for parallel content filtering.
    
    Args:
        text: The text to analyze for filtering
        client: Optional Claude client (uses global client if None)
        
    Returns:
        Dict with 'should_filter' and 'reason' keys
    """
    global CLAUDE_CLIENT
    if client is None:
        client = CLAUDE_CLIENT
        
    # Quick rule-based checks to avoid unnecessary API calls
    # Very short content is automatically filtered
    if len(text.strip()) < 50:
        return {"should_filter": True, "reason": "Content too short"}
        
    try:
        if client and client.client is not None:
            return client.filter_content(text)
        else:
            # Fallback to the simple rule-based approach if Claude is unavailable
            should_filter = is_citation_or_noise(text)
            reason = "Rule-based filtering" if should_filter else "Passed rule-based check"
            return {"should_filter": should_filter, "reason": reason}
    except Exception as e:
        logger.error(f"Error in parallel filtering: {e}")
        # Return a safe default if something goes wrong
        return {"should_filter": False, "reason": f"Error during filtering: {str(e)}"}

def format_paper_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhances the paper metadata with better formatting.
    
    Args:
        metadata: The original metadata dictionary
        
    Returns:
        Enhanced metadata with better paper presentation
    """
    source = metadata.get('source', 'Unknown source')
    text = metadata.get('text', 'No text available')
    arxiv_id = metadata.get('arxiv_id', '')
    paper_url = metadata.get('paper_url', '')
    paper_title = metadata.get('paper_title', '')
    
    # If no arxiv_id was included in metadata, try to extract it from source
    if not arxiv_id and source.endswith('.pdf'):
        arxiv_id = os.path.splitext(source)[0].split('v')[0]  # Remove extension and version
        paper_url = f"{ARXIV_BASE_URL}{arxiv_id}" if arxiv_id else ''
    
    # Log the original metadata title if it exists
    if paper_title:
        logger.info(f"Paper {arxiv_id} already has title in chunk metadata: {paper_title}")
    
    # Try all different variations of ArXiv IDs to maximize chances of finding metadata
    variations = []
    if arxiv_id:
        # Add base ID (without version)
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        variations.append(base_id)
        
        # Try with/without dots
        if '.' in arxiv_id:
            variations.append(arxiv_id.replace('.', ''))
        elif len(arxiv_id) >= 9:  # Looks like YYMMNNNNN format
            # Try to insert a dot after the first 4 characters
            dotted_id = f"{arxiv_id[:4]}.{arxiv_id[4:]}"
            variations.append(dotted_id)
        
        # Try with explicit version suffixes
        for v in range(1, 3):  # Try v1, v2
            variations.append(f"{base_id}v{v}")
    
    # Log what we're trying
    logger.info(f"Looking for metadata with ID variations: {arxiv_id}, {variations}")
    
    # First try the exact ID
    if not paper_title and arxiv_id in PAPER_METADATA:
        paper_title = PAPER_METADATA[arxiv_id].get('title', '')
        if paper_title:
            logger.info(f"✓ Found title for exact ID {arxiv_id}: {paper_title}")
    
    # If still no title, try variations
    if not paper_title:
        for var_id in variations:
            if var_id in PAPER_METADATA:
                paper_title = PAPER_METADATA[var_id].get('title', '')
                if paper_title:
                    logger.info(f"✓ Found title using variation {var_id}: {paper_title}")
                    break
    
    # If still no title, log warning
    if not paper_title:
        logger.warning(f"❌ No title found for any variant of {arxiv_id}")
        
        # Dump the first few characters of metadata keys to help debug
        sample_keys = str(list(PAPER_METADATA.keys())[:5])
        logger.info(f"Sample metadata keys: {sample_keys}")
    
    # Create enhanced metadata
    enhanced = {
        **metadata,
        "arxiv_id": arxiv_id,
        "paper_url": paper_url,
        "paper_title": paper_title,
        "display_text": text
    }
    
    return enhanced

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
async def search_documents(request: SearchRequest):
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
        original_query = request.query
        query_to_use = original_query
        
        # Apply query rewriting if requested and Claude client is available
        if request.rewrite_query and CLAUDE_CLIENT and CLAUDE_CLIENT.client is not None:
            try:
                logger.info(f"Attempting to rewrite query: '{original_query}'")
                rewritten_query = CLAUDE_CLIENT.rewrite_query(original_query)
                if rewritten_query and rewritten_query != original_query:
                    query_to_use = rewritten_query
                    logger.info(f"Using rewritten query: '{query_to_use}'")
            except Exception as e:
                logger.error(f"Error during query rewriting: {e}. Using original query.")
                query_to_use = original_query
        
        logger.info(f"Embedding search query: '{query_to_use}'")

        # Embed query using baseline model
        query_hidden_states = BASELINE_EMBED_FUNCTION(
            BASELINE_MODEL, BASELINE_TOKENIZER, query_to_use, TARGET_EMBEDDING_LAYER
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

        # Perform FAISS search with more results than needed to account for filtered results
        extra_k = request.top_k * 2  # Get extra results to filter
        logger.info(f"Searching index with {FAISS_INDEX.ntotal} vectors for top {extra_k} results (will filter down to {request.top_k}).")
        distances, indices = FAISS_INDEX.search(query_emb_np_final, extra_k)

        # Process results with parallel filtering for citations and noise
        results = []
        filtered_count = 0
        filtering_tasks = []
        candidate_results = []
        
        if indices.size > 0:
            # First collect all candidate results and create filtering tasks
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS returns -1 for padding if fewer results than k are found
                    if 0 <= idx < len(STORED_METADATA):
                        metadata = STORED_METADATA[idx]
                        text = metadata.get("text", "")
                        
                        # Store the candidate result info
                        candidate_result = {
                            "idx": idx,
                            "rank": i,
                            "score": float(distances[0][i]),
                            "metadata": metadata,
                            "text": text
                        }
                        candidate_results.append(candidate_result)
                        
                        # Create filtering task if filtering is requested
                        if request.filter_noise:
                            task = filter_content_async(text)
                            filtering_tasks.append((candidate_result, task))
            
            # If filtering is enabled, process filtering results
            if request.filter_noise and filtering_tasks:
                # Execute all filtering tasks in parallel
                for candidate, task in filtering_tasks:
                    filter_result = await task
                    
                    # Add to results if it shouldn't be filtered
                    if not filter_result.get("should_filter", False):
                        if len(results) < request.top_k:
                            enhanced_metadata = format_paper_info(candidate["metadata"])
                            results.append({
                                "score": candidate["score"],
                                "metadata": enhanced_metadata
                            })
                    else:
                        filtered_count += 1
                        logger.debug(f"Filtered content: {filter_result.get('reason', 'Unknown reason')}")
            else:
                # If no filtering, just add results up to top_k
                for candidate in candidate_results[:request.top_k]:
                    enhanced_metadata = format_paper_info(candidate["metadata"])
                    results.append({
                        "score": candidate["score"],
                        "metadata": enhanced_metadata
                    })

        logger.info(f"Search completed. Found {len(results)} results for query: '{query_to_use}' (filtered {filtered_count} noisy results)")
        response = {
            "query": request.query, 
            "results": results,
            "filtered_count": filtered_count
        }
        
        # Include rewritten query in response if applicable
        if request.rewrite_query and query_to_use != original_query:
            response["rewritten_query"] = query_to_use
            
        return response

    except Exception as e:
        logger.error(f"Error during search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")

# --- Steered Search Endpoint ---

@app.post("/steered_search")
async def steered_search_documents(request: SteeredSearchRequest):
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
        original_query = request.query
        query_to_use = original_query
        
        # Apply query rewriting if requested and Claude client is available
        if request.rewrite_query and CLAUDE_CLIENT and CLAUDE_CLIENT.client is not None:
            try:
                logger.info(f"Attempting to rewrite query: '{original_query}'")
                rewritten_query = CLAUDE_CLIENT.rewrite_query(original_query)
                if rewritten_query and rewritten_query != original_query:
                    query_to_use = rewritten_query
                    logger.info(f"Using rewritten query: '{query_to_use}'")
            except Exception as e:
                logger.error(f"Error during query rewriting: {e}. Using original query.")
                query_to_use = original_query
        
        logger.info(f"Getting base activation for query: '{query_to_use}'")
        # Use the SAE model's method to get activation at the correct layer
        base_activation_tensor = SAE_MODEL.get_activation(query_to_use)
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

        # Perform FAISS search with more results than needed to account for filtered results
        extra_k = request.top_k * 2  # Get extra results to filter
        logger.info(f"Searching index with {FAISS_INDEX.ntotal} vectors for top {extra_k} results (will filter down to {request.top_k}).")
        distances, indices = FAISS_INDEX.search(query_emb_np, extra_k)

        # Process results with parallel filtering
        results = []
        filtered_count = 0
        filtering_tasks = []
        candidate_results = []
        
        if indices.size > 0:
            # First collect all candidate results and create filtering tasks
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS returns -1 for padding if fewer results than k are found
                    if 0 <= idx < len(STORED_METADATA):
                        metadata = STORED_METADATA[idx]
                        text = metadata.get("text", "")
                        
                        # Store the candidate result info
                        candidate_result = {
                            "idx": idx,
                            "rank": i,
                            "score": float(distances[0][i]),
                            "metadata": metadata,
                            "text": text
                        }
                        candidate_results.append(candidate_result)
                        
                        # Create filtering task if filtering is requested
                        if request.filter_noise:
                            task = filter_content_async(text)
                            filtering_tasks.append((candidate_result, task))
            
            # If filtering is enabled, process filtering results
            if request.filter_noise and filtering_tasks:
                # Execute all filtering tasks in parallel
                for candidate, task in filtering_tasks:
                    filter_result = await task
                    
                    # Add to results if it shouldn't be filtered
                    if not filter_result.get("should_filter", False):
                        if len(results) < request.top_k:
                            enhanced_metadata = format_paper_info(candidate["metadata"])
                            results.append({
                                "score": candidate["score"],
                                "metadata": enhanced_metadata
                            })
                    else:
                        filtered_count += 1
                        logger.debug(f"Filtered content: {filter_result.get('reason', 'Unknown reason')}")
            else:
                # If no filtering, just add results up to top_k
                for candidate in candidate_results[:request.top_k]:
                    enhanced_metadata = format_paper_info(candidate["metadata"])
                    results.append({
                        "score": candidate["score"],
                        "metadata": enhanced_metadata
                    })
        else:
            logger.warning(f"FAISS index {idx} is out of bounds for STORED_METADATA (size {len(STORED_METADATA)}). Skipping result.")

        search_type = f"Steered ({len(applied_steering_info)} features)" if applied_steering_info else "Non-Steered"
        logger.info(f"{search_type} search completed. Found {len(results)} results (filtered {filtered_count} noisy results).")
        
        # Update response structure
        response = {
            "query": request.query,
            "steering_info": applied_steering_info, # Return list of applied features/strengths
            "results": results,
            "filtered_count": filtered_count
        }
        
        # Include rewritten query in response if applicable
        if request.rewrite_query and query_to_use != original_query:
            response["rewritten_query"] = query_to_use
            
        return response

    except Exception as e:
        logger.error(f"Error during final embedding processing or FAISS search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")

# --- Auto-Steered Search Endpoint ---

@app.post("/auto_steered_search")
async def auto_steered_search(request: AutoSteeredSearchRequest):
    logger.info(f"Received auto_steered_search request: {request.query}")

    # Check if Claude client is available
    if CLAUDE_CLIENT is None or CLAUDE_CLIENT.client is None:
        logger.error("Claude API client not available or not initialized with API key.")
        return JSONResponse(
            status_code=503,
            content={"message": "Auto-steering not available. Claude API client not initialized with API key."}
        )
    
    if FAISS_INDEX.ntotal == 0:
        logger.info("No documents processed yet. Cannot perform auto-steered search.")
        return {"message": "No documents have been processed yet.", "results": []}
    
    if not request.query.strip():
        logger.info("Received empty search query for auto-steered search.")
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        original_query = request.query
        query_to_use = original_query
        
        # Apply query rewriting if requested
        if request.rewrite_query:
            try:
                logger.info(f"Attempting to rewrite query: '{original_query}'")
                rewritten_query = CLAUDE_CLIENT.rewrite_query(original_query)
                if rewritten_query and rewritten_query != original_query:
                    query_to_use = rewritten_query
                    logger.info(f"Using rewritten query: '{query_to_use}'")
            except Exception as e:
                logger.error(f"Error during query rewriting: {e}. Using original query.")
                query_to_use = original_query
        
        logger.info(f"Starting auto-steered search for query: '{query_to_use}'")
        
        # Step 1: Analyze query intent using Claude
        query_intent = CLAUDE_CLIENT.analyze_query_intent(query_to_use)
        if not query_intent:
            logger.warning(f"Failed to analyze query intent for '{query_to_use}'. Falling back to standard search.")
            # Fall back to standard search
            return search_documents(SearchRequest(query=query_to_use, top_k=request.top_k, filter_noise=request.filter_noise, rewrite_query=False))
        
        # Step 2: Get candidate features
        top_features = SAE_MODEL.get_top_features(query_to_use, k=MAX_FEATURES_TO_MATCH)
        if not top_features:
            logger.warning(f"No candidate features found for query '{query_to_use}'. Falling back to standard search.")
            # Fall back to standard search
            return search_documents(SearchRequest(query=query_to_use, top_k=request.top_k, filter_noise=request.filter_noise, rewrite_query=False))
        
        # Format features for Claude API
        features_for_matching = []
        for feature in top_features:
            feature_id = feature["feature_id"]
            activation = feature["activation"]
            explanations = feature["explanations"]
            description = explanations[0] if explanations else "No explanation available."
            
            features_for_matching.append({
                "feature_id": feature_id,
                "description": description,
                "activation": activation
            })
        
        # Step 3: Match features to query intent
        selected_features = CLAUDE_CLIENT.match_features_to_intent(
            features_for_matching, 
            query_intent
        )
        
        if not selected_features:
            logger.warning(f"No features selected for query '{query_to_use}'. Falling back to standard search.")
            # Fall back to standard search
            return search_documents(SearchRequest(query=query_to_use, top_k=request.top_k, filter_noise=request.filter_noise, rewrite_query=False))
        
        # Limit to requested number of features
        if len(selected_features) > request.max_features:
            selected_features = selected_features[:request.max_features]
        
        # Step 4: Convert to steering parameters
        steering_params = []
        for feature in selected_features:
            # Cap strength to max_strength
            raw_strength = float(feature.get("strength", 0))
            capped_strength = max(min(raw_strength, 10), -10)
            scaled_strength = (capped_strength / 10) * request.max_strength
            
            steering_params.append(SteeringParam(
                feature_id=int(feature["feature_id"]),
                strength=scaled_strength
            ))
        
        logger.info(f"Auto-selected {len(steering_params)} steering features for query '{request.query}'")
        
        # Step 5: Create steered search request with selected parameters
        steered_request = SteeredSearchRequest(
            query=query_to_use,
            top_k=request.top_k,
            steering_params=steering_params,
            filter_noise=request.filter_noise,
            rewrite_query=False  # Already rewritten if needed
        )
        
        # Step 6: Execute steered search with selected parameters
        search_results = await steered_search_documents(steered_request)
        
        # Add auto-steering info to the response
        search_results["auto_steering"] = {
            "query_intent": query_intent,
            "selected_features": selected_features
        }
        
        # Include the original query and rewritten query if different
        if request.rewrite_query and query_to_use != original_query:
            if "rewritten_query" not in search_results:  # avoid duplication if steered_search already added it
                search_results["rewritten_query"] = query_to_use
            search_results["original_query"] = original_query
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error during auto-steered search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform auto-steered search: {str(e)}")

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