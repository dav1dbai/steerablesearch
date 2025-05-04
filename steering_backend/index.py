# steering_backend/profile_index.py

import os
import faiss
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Assumes this script is in steering_backend, adjust if needed
SCRIPT_DIR = Path(__file__).parent
FAISS_DATA_DIR = SCRIPT_DIR / "faiss_data"
FAISS_INDEX_PATH = FAISS_DATA_DIR / "arxiv_index.faiss"
METADATA_PATH = FAISS_DATA_DIR / "arxiv_index.faiss.meta.json"

def profile_faiss_data():
    """Loads FAISS index and metadata, then prints statistics."""

    logger.info("--- Starting FAISS Data Profiling ---")
    logger.info(f"Looking for index in: {FAISS_INDEX_PATH}")
    logger.info(f"Looking for metadata in: {METADATA_PATH}")

    index_vector_count = 0
    metadata_entry_count = 0
    unique_pdf_count = 0
    pdf_filenames = set()
    index_dimension = None
    faiss_index_loaded = False
    metadata_loaded = False

    # 1. Load FAISS Index
    if FAISS_INDEX_PATH.exists():
        try:
            logger.info("Attempting to load FAISS index...")
            faiss_index = faiss.read_index(str(FAISS_INDEX_PATH)) # read_index needs string path
            index_vector_count = faiss_index.ntotal
            index_dimension = faiss_index.d
            faiss_index_loaded = True
            logger.info(f"Successfully loaded FAISS index. Found {index_vector_count} vectors (dimension: {index_dimension}).")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")
    else:
        logger.warning(f"FAISS index file not found at {FAISS_INDEX_PATH}")

    # 2. Load Metadata
    if METADATA_PATH.exists():
        try:
            logger.info("Attempting to load metadata...")
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata_entry_count = len(metadata)
            metadata_loaded = True
            logger.info(f"Successfully loaded metadata. Found {metadata_entry_count} entries.")

            # Extract unique PDF filenames
            for entry in metadata:
                if 'source' in entry:
                    pdf_filenames.add(entry['source'])
                else:
                    logger.warning("Found metadata entry without a 'source' key.")
            unique_pdf_count = len(pdf_filenames)
            logger.info(f"Found {unique_pdf_count} unique PDF filenames in metadata.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON metadata from {METADATA_PATH}: {e}")
        except Exception as e:
            logger.error(f"Failed to load metadata from {METADATA_PATH}: {e}")
    else:
        logger.warning(f"Metadata file not found at {METADATA_PATH}")

    # 3. Print Summary
    logger.info("--- Profiling Summary ---")
    if faiss_index_loaded:
        print(f"FAISS Index Vectors: {index_vector_count}")
        print(f"FAISS Index Dimension: {index_dimension}")
    else:
        print("FAISS Index: Not loaded (file not found or error)")

    if metadata_loaded:
        print(f"Metadata Entries: {metadata_entry_count}")
        print(f"Unique PDF files in Metadata: {unique_pdf_count}")
    else:
        print("Metadata: Not loaded (file not found or error)")

    # 4. Consistency Check
    if faiss_index_loaded and metadata_loaded:
        if index_vector_count == metadata_entry_count:
            logger.info("Consistency Check: Index vector count matches metadata entry count. OK.")
        else:
            logger.warning(f"Consistency Check: Mismatch! Index vectors ({index_vector_count}) != Metadata entries ({metadata_entry_count}).")
    elif faiss_index_loaded or metadata_loaded:
         logger.warning("Consistency Check: Cannot perform check as only one of index/metadata was loaded.")
    else:
         logger.info("Consistency Check: Neither index nor metadata loaded.")

    logger.info("--- Profiling Complete ---")


if __name__ == "__main__":
    profile_faiss_data()