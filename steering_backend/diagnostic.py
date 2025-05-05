"""
Diagnostic script for the steerable search system.
Analyzes index, embeddings, and search quality.
"""

import os
import sys
import json
import numpy as np
import faiss
import logging
from collections import Counter
import matplotlib.pyplot as plt
import re
from typing import List, Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import server components
try:
    from server import (
        FAISS_INDEX, STORED_METADATA, SAE_MODEL, PAPER_METADATA, 
        BASELINE_MODEL, BASELINE_TOKENIZER, BASELINE_EMBED_FUNCTION,
        _MLX_AVAILABLE, _TORCH_AVAILABLE, 
        EMBEDDING_DIM, TARGET_EMBEDDING_LAYER,
        load_embedding_model
    )
    server_modules_loaded = True
except ImportError as e:
    logger.error(f"Error importing server modules: {e}")
    server_modules_loaded = False
    # Define placeholder for minimal functionality
    FAISS_INDEX = None
    STORED_METADATA = []
    
# Paths
_SCRIPT_DIR = os.path.dirname(__file__)
FAISS_DATA_DIR = os.path.join(_SCRIPT_DIR, "faiss_data")
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "arxiv_index.faiss")
METADATA_PATH = FAISS_INDEX_PATH + ".meta.json"
DIAGNOSTIC_OUTPUT = os.path.join(_SCRIPT_DIR, "diagnostic_results.json")

def load_index_and_metadata() -> Tuple[Any, List[Dict[str, Any]]]:
    """Load FAISS index and metadata from disk."""
    if server_modules_loaded and FAISS_INDEX is not None:
        logger.info("Using already loaded FAISS index and metadata from server")
        return FAISS_INDEX, STORED_METADATA
    
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = None
    metadata = []
    
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        else:
            logger.error(f"FAISS index file not found at {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
    
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
        else:
            logger.error(f"Metadata file not found at {METADATA_PATH}")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
    
    return index, metadata

def analyze_index_statistics(index, metadata) -> Dict[str, Any]:
    """Analyze statistics about the FAISS index and metadata."""
    results = {}
    
    # Basic index info
    results["index_size"] = index.ntotal if index else 0
    results["metadata_size"] = len(metadata)
    
    if not index or not metadata:
        return results
    
    # Check for size mismatch
    results["size_match"] = results["index_size"] == results["metadata_size"]
    
    # Analyze metadata
    if metadata:
        # Count papers by source
        sources = Counter(item.get("source", "unknown") for item in metadata)
        results["unique_papers"] = len(sources)
        results["chunks_per_paper_avg"] = results["metadata_size"] / max(results["unique_papers"], 1)
        
        # Count papers with titles
        papers_with_titles = sum(1 for item in metadata if item.get("paper_title", ""))
        results["papers_with_titles_pct"] = papers_with_titles / max(results["metadata_size"], 1) * 100
        
        # Analyze chunk lengths
        chunk_lengths = [len(item.get("text", "")) for item in metadata]
        if chunk_lengths:
            results["chunk_length_avg"] = sum(chunk_lengths) / len(chunk_lengths)
            results["chunk_length_min"] = min(chunk_lengths)
            results["chunk_length_max"] = max(chunk_lengths)
            results["total_text_size_mb"] = sum(chunk_lengths) / (1024 * 1024)
    
    return results

def analyze_embedding_distribution(index, metadata, samples=100) -> Dict[str, Any]:
    """Analyze the distribution of embeddings in the index."""
    results = {}
    
    if not index or not metadata or index.ntotal == 0:
        logger.warning("Cannot analyze embedding distribution - empty index or metadata")
        return {"embedding_samples": 0}
    
    # Get a sample of vectors from the index
    sample_size = min(samples, index.ntotal)
    
    # Randomly sample indices
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(index.ntotal, sample_size, replace=False)
    
    # Extract vectors
    sample_vectors = np.zeros((sample_size, index.d), dtype=np.float32)
    for i, idx in enumerate(sample_indices):
        # Correctly assign the reconstructed vector (which is already a NumPy array)
        sample_vectors[i] = index.reconstruct(int(idx))
    
    # Calculate statistics on the sample
    vector_norms = np.linalg.norm(sample_vectors, axis=1)
    
    results["embedding_samples"] = sample_size
    results["embedding_norms_mean"] = float(np.mean(vector_norms))
    results["embedding_norms_std"] = float(np.std(vector_norms))
    results["embedding_norms_min"] = float(np.min(vector_norms))
    results["embedding_norms_max"] = float(np.max(vector_norms))
    
    # Calculate pairwise cosine similarities within the sample
    norm_vectors = sample_vectors / np.maximum(vector_norms[:, np.newaxis], 1e-8)
    similarities = np.matmul(norm_vectors, norm_vectors.T)
    
    # Get average similarity (excluding self-similarity)
    np.fill_diagonal(similarities, 0)
    avg_similarity = np.sum(similarities) / (sample_size * (sample_size - 1))
    results["avg_pairwise_similarity"] = float(avg_similarity)
    
    return results

def analyze_search_examples(index, metadata) -> Dict[str, Any]:
    """Run some example searches and analyze the results."""
    results = {}
    
    if not server_modules_loaded or not index or not metadata:
        logger.warning("Cannot perform search analysis - server modules not loaded or missing data")
        return {"searches_performed": 0}
    
    # Skip if we don't have a properly loaded embedding model
    if BASELINE_MODEL is None or BASELINE_TOKENIZER is None or BASELINE_EMBED_FUNCTION is None:
        logger.warning("Cannot perform search analysis - embedding models not loaded")
        return {"searches_performed": 0}
    
    # Example queries to test
    example_queries = [
        "sparse autoencoders for interpretability",
        "language model alignment techniques",
        "transformer attention mechanisms",
        "RLHF reinforcement learning from human feedback",
        "mechanistic interpretability"
    ]
    
    searches = []
    
    for query in example_queries:
        logger.info(f"Running test search for query: '{query}'")
        try:
            # Embed query (simplified from server.py search endpoint)
            query_hidden_states = BASELINE_EMBED_FUNCTION(
                BASELINE_MODEL, BASELINE_TOKENIZER, query, TARGET_EMBEDDING_LAYER
            )
            
            # Convert to numpy and handle different backends
            if _MLX_AVAILABLE and BASELINE_EMBED_FUNCTION.__name__ == 'embed_mlx':
                query_emb_np_full = np.array(query_hidden_states, dtype=np.float32)
            elif _TORCH_AVAILABLE and BASELINE_EMBED_FUNCTION.__name__ == 'embed_torch':
                query_emb_np_full = query_hidden_states.cpu().numpy().astype(np.float32)
            else:
                logger.warning(f"Unknown embedding function type for query '{query}'")
                continue
            
            # Average and reshape
            query_vector_avg = np.mean(query_emb_np_full[0, :, :], axis=0)
            query_emb_np = query_vector_avg.reshape(1, -1)
            
            # Perform search
            k = 5
            distances, indices = index.search(query_emb_np, k)
            
            # Collect results
            search_results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and 0 <= idx < len(metadata):
                    result = {
                        "rank": i + 1,
                        "score": float(distances[0][i]),
                        "paper_title": metadata[idx].get("paper_title", "Unknown"),
                        "text_snippet": metadata[idx].get("text", "")[:100] + "..." if len(metadata[idx].get("text", "")) > 100 else metadata[idx].get("text", ""),
                        "chunk_index": metadata[idx].get("chunk_index", None),
                        "text_length": len(metadata[idx].get("text", ""))
                    }
                    search_results.append(result)
            
            searches.append({
                "query": query,
                "results": search_results,
                "num_results": len(search_results)
            })
            
        except Exception as e:
            logger.error(f"Error during test search for query '{query}': {e}")
    
    results["searches_performed"] = len(searches)
    results["example_searches"] = searches
    
    return results

def analyze_chunk_quality(metadata, samples=10) -> Dict[str, Any]:
    """Analyze the quality of text chunks in the metadata."""
    results = {}
    
    if not metadata:
        logger.warning("Cannot analyze chunk quality - empty metadata")
        return {"chunk_samples": 0}
    
    # Get a random sample of chunks
    sample_size = min(samples, len(metadata))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(metadata), sample_size, replace=False)
    
    chunk_samples = []
    for idx in sample_indices:
        chunk = metadata[idx]
        
        # Calculate some metrics on the chunk
        text = chunk.get("text", "")
        text_length = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Look for citation patterns
        citation_count = len(re.findall(r'\[\d+\]', text))
        reference_count = len(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE))
        
        chunk_samples.append({
            "source": chunk.get("source", "unknown"),
            "paper_title": chunk.get("paper_title", "Unknown"),
            "text_length": text_length,
            "sentences": sentence_count,
            "citations": citation_count,
            "references": reference_count,
            "chunk_index": chunk.get("chunk_index", None),
            "text_snippet": text[:100] + "..." if text_length > 100 else text,
            "has_title": bool(chunk.get("paper_title", ""))
        })
    
    results["chunk_samples"] = chunk_samples
    
    # Calculate overall statistics on samples
    results["avg_text_length"] = sum(s["text_length"] for s in chunk_samples) / sample_size
    results["avg_sentences"] = sum(s["sentences"] for s in chunk_samples) / sample_size
    results["avg_citations"] = sum(s["citations"] for s in chunk_samples) / sample_size
    results["avg_references"] = sum(s["references"] for s in chunk_samples) / sample_size
    results["pct_has_title"] = sum(1 for s in chunk_samples if s["has_title"]) / sample_size * 100
    
    return results

def generate_visualization(statistics, output_dir=_SCRIPT_DIR):
    """Generate some visualizations based on the statistics."""
    if not statistics:
        return
    
    # Only attempt to generate if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualizations")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Example 1: Chunk length distribution from the chunk_quality analysis
        if "chunk_quality" in statistics and "chunk_samples" in statistics["chunk_quality"]:
            chunk_lengths = [chunk["text_length"] for chunk in statistics["chunk_quality"]["chunk_samples"]]
            if chunk_lengths:
                plt.figure(figsize=(10, 6))
                plt.hist(chunk_lengths, bins=20, alpha=0.7)
                plt.title('Distribution of Chunk Lengths')
                plt.xlabel('Chunk Length (characters)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'chunk_length_distribution.png'))
                plt.close()
                logger.info(f"Generated chunk length distribution visualization")
        
        # Example 2: Embedding norms distribution
        if "embedding_distribution" in statistics and "embedding_samples" in statistics["embedding_distribution"]:
            if statistics["embedding_distribution"]["embedding_samples"] > 0:
                norm_stats = [
                    statistics["embedding_distribution"]["embedding_norms_min"],
                    statistics["embedding_distribution"]["embedding_norms_mean"] - statistics["embedding_distribution"]["embedding_norms_std"],
                    statistics["embedding_distribution"]["embedding_norms_mean"],
                    statistics["embedding_distribution"]["embedding_norms_mean"] + statistics["embedding_distribution"]["embedding_norms_std"],
                    statistics["embedding_distribution"]["embedding_norms_max"]
                ]
                
                plt.figure(figsize=(10, 6))
                plt.boxplot([norm_stats], labels=['Embedding Norms'])
                plt.title('Distribution of Embedding Norms')
                plt.ylabel('Norm Value')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'embedding_norms_distribution.png'))
                plt.close()
                logger.info(f"Generated embedding norms distribution visualization")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def run_diagnostics() -> Dict[str, Any]:
    """Run all diagnostic tests and return results."""
    logger.info("Starting diagnostic tests")
    
    # Load data
    index, metadata = load_index_and_metadata()
    
    # Run analyses
    statistics = {}
    
    # Basic index statistics
    statistics["index_stats"] = analyze_index_statistics(index, metadata)
    logger.info(f"Index size: {statistics['index_stats'].get('index_size', 0)}, Metadata entries: {statistics['index_stats'].get('metadata_size', 0)}")
    
    # Embedding distribution analysis
    statistics["embedding_distribution"] = analyze_embedding_distribution(index, metadata)
    logger.info(f"Analyzed distribution of {statistics['embedding_distribution'].get('embedding_samples', 0)} embedding samples")
    
    # Chunk quality analysis
    statistics["chunk_quality"] = analyze_chunk_quality(metadata)
    logger.info(f"Analyzed {statistics['chunk_quality'].get('chunk_samples', 0)} sample chunks")
    
    # Search examples
    if server_modules_loaded:
        statistics["search_examples"] = analyze_search_examples(index, metadata)
        logger.info(f"Performed {statistics['search_examples'].get('searches_performed', 0)} example searches")
    
    # Generate visualizations
    generate_visualization(statistics)
    
    # Save results to JSON
    try:
        with open(DIAGNOSTIC_OUTPUT, 'w') as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"Saved diagnostic results to {DIAGNOSTIC_OUTPUT}")
    except Exception as e:
        logger.error(f"Error saving diagnostic results: {e}")
    
    return statistics

def print_diagnostic_summary(statistics):
    """Print a summary of the diagnostic results to console."""
    if not statistics:
        print("No diagnostic statistics available")
        return
    
    print("\n" + "="*80)
    print("STEERABLE SEARCH DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Index statistics
    print("\n--- INDEX STATISTICS ---")
    idx_stats = statistics.get("index_stats", {})
    print(f"Index size:               {idx_stats.get('index_size', 0):,} vectors")
    print(f"Metadata entries:         {idx_stats.get('metadata_size', 0):,} entries")
    print(f"Size match:               {'✓' if idx_stats.get('size_match', False) else '✗'}")
    print(f"Unique papers:            {idx_stats.get('unique_papers', 0):,}")
    print(f"Average chunks per paper: {idx_stats.get('chunks_per_paper_avg', 0):.2f}")
    print(f"Papers with titles:       {idx_stats.get('papers_with_titles_pct', 0):.2f}%")
    print(f"Average chunk length:     {idx_stats.get('chunk_length_avg', 0):,.0f} characters")
    print(f"Total text size:          {idx_stats.get('total_text_size_mb', 0):.2f} MB")
    
    # Embedding distribution
    print("\n--- EMBEDDING DISTRIBUTION ---")
    emb_stats = statistics.get("embedding_distribution", {})
    if emb_stats.get("embedding_samples", 0) > 0:
        print(f"Analyzed embedding samples: {emb_stats.get('embedding_samples', 0)}")
        print(f"Average embedding norm:     {emb_stats.get('embedding_norms_mean', 0):.4f}")
        print(f"Embedding norm std dev:     {emb_stats.get('embedding_norms_std', 0):.4f}")
        print(f"Min/Max embedding norm:     {emb_stats.get('embedding_norms_min', 0):.4f} / {emb_stats.get('embedding_norms_max', 0):.4f}")
        print(f"Average pairwise similarity: {emb_stats.get('avg_pairwise_similarity', 0):.4f}")
    else:
        print("No embedding samples analyzed")
    
    # Chunk quality
    print("\n--- CHUNK QUALITY ANALYSIS ---")
    chunk_stats = statistics.get("chunk_quality", {})
    if chunk_stats:
        print(f"Analyzed chunk samples:    {len(chunk_stats.get('chunk_samples', []))}")
        print(f"Average text length:       {chunk_stats.get('avg_text_length', 0):.2f} characters")
        print(f"Average sentences:         {chunk_stats.get('avg_sentences', 0):.2f} per chunk")
        print(f"Average citations:         {chunk_stats.get('avg_citations', 0):.2f} per chunk")
        print(f"Average references:        {chunk_stats.get('avg_references', 0):.2f} per chunk")
        print(f"Percentage with title:     {chunk_stats.get('pct_has_title', 0):.2f}%")
    else:
        print("No chunk quality analysis performed")
    
    # Search examples
    print("\n--- SEARCH EXAMPLES ---")
    search_stats = statistics.get("search_examples", {})
    if search_stats.get("searches_performed", 0) > 0:
        print(f"Searches performed: {search_stats.get('searches_performed', 0)}")
        
        for i, search in enumerate(search_stats.get("example_searches", [])):
            print(f"\nSearch {i+1}: '{search['query']}'")
            
            if not search.get("results"):
                print("  No results found")
                continue
                
            print(f"  Results found: {search.get('num_results', 0)}")
            for j, result in enumerate(search.get("results", [])[:3]):  # Show top 3
                print(f"  {j+1}. [{result.get('score', 0):.4f}] {result.get('paper_title', 'Unknown')}")
                snippet = result.get("text_snippet", "").replace("\n", " ")
                print(f"     {snippet}")
            
            if len(search.get("results", [])) > 3:
                print(f"  ... and {len(search.get('results', [])) - 3} more results")
    else:
        print("No search examples were performed")
    
    print("\n" + "="*80)
    print(f"Full results saved to: {DIAGNOSTIC_OUTPUT}")
    print("="*80 + "\n")

if __name__ == "__main__":
    statistics = run_diagnostics()
    print_diagnostic_summary(statistics)