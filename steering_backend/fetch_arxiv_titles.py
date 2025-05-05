#!/usr/bin/env python3
"""
Script to fetch titles for existing arXiv papers in the arxiv directory

Usage:
  python fetch_arxiv_titles.py  # Defaults to ./arxiv directory and saves to ./arxiv/arxiv_metadata.json
  python fetch_arxiv_titles.py --directory /path/to/arxiv/pdfs --output /path/to/output.json

Requirements:
  pip install requests xmltodict tqdm
"""

import os
import re
import json
import time
import logging
import requests
import xmltodict
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_METADATA_FILE = "arxiv_metadata.json"
RATE_LIMIT_SLEEP = 3  # Sleep between API calls to respect arXiv's rate limits

def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Normalize arXiv ID to a consistent format.
    This handles various formats like:
    - YYMM.NNNNN
    - YYMMNNNNNvN
    - YYMM.NNNNNvN
    """
    # Remove any version suffix
    id_without_version = re.sub(r'v\d+$', '', arxiv_id)    
    
    # Check if the ID has a dot (e.g., 2505.00025)
    if '.' in id_without_version:
        return id_without_version  # Already in normalized format
    
    # Assume format like 2504NNNNN or YYYYNNNNN
    if len(id_without_version) >= 9:
        # Try to insert dot after the first 4 characters if it's a numeric ID
        if id_without_version.isdigit():
            return f"{id_without_version[:4]}.{id_without_version[4:]}"
    
    # Return original ID without version as fallback
    return id_without_version

def get_arxiv_ids_from_directory(directory: str) -> List[str]:
    """
    Get arXiv IDs from PDF filenames in the directory.
    Filenames are expected to be in the format YYMM.NNNNN.pdf or YYMM.NNNNNvN.pdf
    """
    arxiv_ids = []
    id_mapping = {}  # Map from normalized ID to original filename
    
    # List all PDF files
    try:
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return []
    except PermissionError:
        logger.error(f"Permission denied when accessing directory: {directory}")
        return []
    
    # Extract arXiv IDs from filenames
    for filename in pdf_files:
        # Remove the extension and version suffix (vN)
        base_name = os.path.splitext(filename)[0]
        raw_arxiv_id = re.sub(r'v\d+$', '', base_name)  # Remove version suffix
        
        # Normalize the ID
        arxiv_id = normalize_arxiv_id(raw_arxiv_id)
        
        # Store the normalized ID and its original filename
        arxiv_ids.append(arxiv_id)
        id_mapping[arxiv_id] = filename
    
    # Log some sample IDs for debugging
    for i, arxiv_id in enumerate(arxiv_ids[:5]):
        logger.info(f"Sample ID {i+1}: {arxiv_id} (from file: {id_mapping[arxiv_id]})")
    
    return arxiv_ids

def batch_arxiv_ids(arxiv_ids: List[str], batch_size: int = 100) -> List[List[str]]:
    """
    Split the list of arXiv IDs into batches to avoid making too many API calls.
    """
    return [arxiv_ids[i:i + batch_size] for i in range(0, len(arxiv_ids), batch_size)]

def fetch_arxiv_metadata(arxiv_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch metadata for a list of arXiv IDs using the arXiv API.
    
    Args:
        arxiv_ids: List of arXiv IDs
        
    Returns:
        Dict mapping arXiv IDs to their metadata
    """
    if not arxiv_ids:
        return {}
    
    # Join IDs for API query
    id_list = ','.join(arxiv_ids)
    
    # Query parameters
    params = {
        'id_list': id_list,
        'max_results': len(arxiv_ids)
    }
    
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        
        # Parse XML response
        data = xmltodict.parse(response.content)
        
        # Extract entries
        entries = data['feed'].get('entry', [])
        if not isinstance(entries, list):
            entries = [entries]  # Ensure entries is a list
        
        # Process entries into a dictionary
        metadata = {}
        for entry in entries:
            # Extract the arXiv ID
            if 'id' in entry:
                arxiv_url = entry['id']
                original_id = arxiv_url.split('/abs/')[-1]
                
                # Normalize the ID for storage
                arxiv_id = normalize_arxiv_id(original_id)
                
                # Log the IDs
                if arxiv_id != original_id:
                    logger.info(f"Normalized ID from {original_id} to {arxiv_id}")
                
                # Get title and clean it
                title = entry.get('title', '').replace('\n', ' ').strip()
                
                # Handle authors
                author_data = entry.get('author', [])
                if not isinstance(author_data, list):
                    author_data = [author_data]
                
                authors = ", ".join([author.get('name', '') for author in author_data if author])
                
                # Build metadata
                metadata[arxiv_id] = {
                    'title': title,
                    'authors': authors,
                    'summary': entry.get('summary', '').replace('\n', ' ').strip(),
                    'published': entry.get('published', ''),
                    'updated': entry.get('updated', ''),
                    'arxiv_id': arxiv_id,
                    'paper_url': f"https://arxiv.org/abs/{arxiv_id}"
                }
                
                # Also store under the original ID if different
                if arxiv_id != original_id:
                    metadata[original_id] = metadata[arxiv_id].copy()
                
                # Extract categories
                if 'category' in entry:
                    categories = entry['category']
                    if isinstance(categories, list):
                        metadata[arxiv_id]['categories'] = [cat.get('@term', '') for cat in categories]
                    else:
                        metadata[arxiv_id]['categories'] = [categories.get('@term', '')]
                    
                    # Copy categories to the original ID version if different
                    if arxiv_id != original_id:
                        metadata[original_id]['categories'] = metadata[arxiv_id]['categories']
        
        return metadata
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching metadata from arXiv API: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error processing arXiv API response: {e}")
        # Try to provide more detailed error info
        if 'data' in locals() and data:
            logger.error(f"API response data: {data}")
        return {}

def load_existing_metadata(file_path: str) -> Dict[str, Dict]:
    """
    Load existing metadata from a JSON file if it exists.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Existing metadata file {file_path} is not valid JSON. Creating new file.")
            return {}
        except Exception as e:
            logger.error(f"Error loading existing metadata file {file_path}: {e}")
            return {}
    return {}

def save_metadata(metadata: Dict[str, Dict], file_path: str):
    """
    Save metadata to a JSON file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {len(metadata)} papers to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metadata to {file_path}: {e}")

def fetch_titles_for_directory(directory_path: str, output_file: Optional[str] = None):
    """
    Fetch titles for all arXiv papers in a directory and save to a JSON file.
    
    Args:
        directory_path: Path to directory containing arXiv PDFs
        output_file: Path to output JSON file (default: directory/arxiv_metadata.json)
    """
    # Resolve directory and output file paths
    directory = Path(directory_path).resolve()
    if output_file is None:
        output_file = directory / ARXIV_METADATA_FILE
    else:
        output_file = Path(output_file).resolve()
    
    logger.info(f"Fetching titles for arXiv papers in {directory}")
    logger.info(f"Will save metadata to {output_file}")
    
    # Get existing metadata
    existing_metadata = load_existing_metadata(output_file)
    logger.info(f"Loaded {len(existing_metadata)} existing paper metadata entries")
    
    # Get arXiv IDs from directory
    arxiv_ids = get_arxiv_ids_from_directory(directory)
    
    # Filter out IDs that we already have metadata for
    new_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in existing_metadata]
    logger.info(f"Need to fetch metadata for {len(new_ids)} new papers")
    
    # If no new papers, we're done
    if not new_ids:
        logger.info("No new papers to fetch metadata for")
        return
    
    # Batch IDs to respect API rate limits (max 100 IDs per request)
    batches = batch_arxiv_ids(new_ids, batch_size=100)
    logger.info(f"Split {len(new_ids)} IDs into {len(batches)} batches")
    
    # Fetch metadata for each batch
    all_metadata = existing_metadata.copy()
    
    with tqdm(total=len(batches), desc="Fetching batches") as pbar:
        for i, batch in enumerate(batches):
            logger.info(f"Fetching batch {i+1}/{len(batches)} with {len(batch)} IDs")
            batch_metadata = fetch_arxiv_metadata(batch)
            all_metadata.update(batch_metadata)
            
            # Save after each batch in case of errors
            save_metadata(all_metadata, output_file)
            
            # Respect rate limits
            if i < len(batches) - 1:  # Don't sleep after the last batch
                logger.info(f"Sleeping {RATE_LIMIT_SLEEP} seconds to respect API rate limits")
                time.sleep(RATE_LIMIT_SLEEP)
            
            pbar.update(1)
    
    # Final save
    save_metadata(all_metadata, output_file)
    logger.info(f"Completed fetching metadata for {len(new_ids)} papers")
    logger.info(f"Total papers in metadata: {len(all_metadata)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch titles for arXiv papers")
    parser.add_argument("--directory", "-d", type=str, default="./arxiv",
                        help="Directory containing arXiv PDFs (default: ./arxiv)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: <directory>/arxiv_metadata.json)")
    
    args = parser.parse_args()
    
    # Use script directory as base if paths are relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.directory):
        args.directory = os.path.join(script_dir, args.directory)
    
    if args.output and not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    fetch_titles_for_directory(args.directory, args.output)