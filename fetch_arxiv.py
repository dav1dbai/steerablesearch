#!/usr/bin/env python3
"""
Script to download the most recent papers from arXiv in the Computer Science (cs) category.
This script uses the arxiv package to interact with the arXiv API.

The script follows the arXiv API guidelines and best practices:
- Incorporates a delay between requests (recommended: 3 seconds)
- Respects API limits (max 2000 results per request, max 30000 total results)
- Properly formats search queries according to arXiv API specifications
- Provides options for sorting and batch sizes

Usage examples:
  python fetch_arxiv.py --max-results 1000 --category cs --subcategory AI
  python fetch_arxiv.py --sort-by submitted --sort-order descending
"""

import arxiv
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import concurrent.futures
import threading


def setup_argparse():
    parser = argparse.ArgumentParser(description='Download recent CS papers from arXiv')
    parser.add_argument(
        '--max-results', type=int, default=1000,
        help='Maximum number of papers to download (default: 1000, max allowed by API in a single session: 30000)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./arxiv',
        help='Directory where PDF papers will be saved (default: ./arxiv)'
    )
    parser.add_argument(
        '--category', type=str, default='cs',
        help='arXiv category to download papers from (default: cs)'
    )
    parser.add_argument(
        '--subcategory', type=str, default=None,
        help='arXiv subcategory (e.g., AI, LG for cs.AI, cs.LG) (default: None, which means all CS subcategories)'
    )
    parser.add_argument(
        '--sort-by', type=str, choices=['submitted', 'updated', 'relevance'], default='submitted',
        help='Sort criterion for papers (default: submitted)'
    )
    parser.add_argument(
        '--sort-order', type=str, choices=['ascending', 'descending'], default='descending',
        help='Sort order for results (default: descending - newest first)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1000,
        help='Number of results to retrieve per API request (default: 100, max allowed: 2000)'
    )
    parser.add_argument(
        '--delay', type=float, default=3.0,
        help='Delay between API requests in seconds to avoid rate limiting (recommended: 3.0)'
    )
    parser.add_argument(
        '--max-workers', type=int, default=10,
        help='Maximum number of parallel download threads (default: 10)'
    )
    return parser


def create_search_query(category, subcategory=None):
    """
    Create a category-based search query for arXiv using proper API syntax.
    
    Args:
        category: Main category (e.g., 'cs')
        subcategory: Optional subcategory (e.g., 'AI', 'LG')
        
    Returns:
        Properly formatted search query string for the arXiv API
    """
    # Construct the category query
    if subcategory:
        query = f"cat:{category}.{subcategory}"
    else:
        query = f"cat:{category}"
        
    return query


def create_output_dirs(base_dir):
    """Create output directory for PDFs."""
    pdf_dir = os.path.join(base_dir, "pdfs")
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)
    
    return pdf_dir


def sanitize_filename(title, max_length=100):
    """Create a safe filename from the paper title."""
    # Replace invalid filename characters
    safe_title = "".join(c if c.isalnum() or c in " -_.,;()[]{}'" else "_" for c in title)
    # Trim to reasonable length
    safe_title = safe_title[:max_length]
    # Remove trailing whitespace or dots
    safe_title = safe_title.rstrip(" .")
    return safe_title


def get_sort_criterion(sort_by):
    """
    Return the corresponding SortCriterion enum value based on API docs.
    
    The arXiv API supports three sort options:
    - submittedDate: Sort by date article was submitted
    - lastUpdatedDate: Sort by date article was last updated
    - relevance: Sort by relevance score (default for arXiv API)
    
    Args:
        sort_by: String indicating sort criterion
        
    Returns:
        The appropriate SortCriterion enum value
    """
    if sort_by == 'submitted':
        return arxiv.SortCriterion.SubmittedDate
    elif sort_by == 'updated':
        return arxiv.SortCriterion.LastUpdatedDate
    elif sort_by == 'relevance':
        return arxiv.SortCriterion.Relevance
    else:
        # Default to SubmittedDate for our downloader
        return arxiv.SortCriterion.SubmittedDate


def download_paper_task(result, pdf_dir):
    """Task function to download a single paper's PDF.
    Returns a tuple: (result, pdf_status, filename_base)
    """
    safe_title = sanitize_filename(result.title)
    filename_base = f"{result.get_short_id()}_{safe_title}"
    pdf_filename = f"{filename_base}.pdf"
    pdf_status = "Not Downloaded"
    try:
        result.download_pdf(dirpath=pdf_dir, filename=pdf_filename)
        pdf_status = "Success"
    except Exception as e:
        pdf_status = f"Failed: {str(e)}"
        print(f"\nError downloading {result.get_short_id()}: {str(e)}") # Print errors immediately
    return result, pdf_status, filename_base


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directories
    pdf_dir = create_output_dirs(args.output_dir)
    
    # Create metadata log file
    log_file = os.path.join(args.output_dir, f"arxiv_download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Build the search query with proper syntax according to API docs
    query = create_search_query(args.category, args.subcategory)
    
    # Set up client with appropriate parameters - respect API rate limits
    client = arxiv.Client(
        page_size=min(args.batch_size, 2000),  # API allows max 2000 per request
        delay_seconds=args.delay,
        num_retries=5
    )
    
    # Determine sort order 
    sort_order = arxiv.SortOrder.Descending if args.sort_order == 'descending' else arxiv.SortOrder.Ascending
    
    # Create search object with proper parameters
    search = arxiv.Search(
        query=query,
        max_results=min(args.max_results, 30000),  # API allows max 30000 results total
        sort_by=get_sort_criterion(args.sort_by),
        sort_order=sort_order
    )
    
    print(f"Searching for up to {args.max_results} papers matching: {query}")
    print(f"Sort: {args.sort_by} in {args.sort_order} order")
    
    # --- Fetch results first ---
    start_fetch_time = time.time()
    print("Fetching paper metadata...")
    results_iterator = client.results(search)
    all_results = []
    try:
        # Use API's max_results which considers args.max_results and 30k limit
        # The iterator respects the max_results set in Search
        for result in results_iterator:
             all_results.append(result)
             if len(all_results) >= args.max_results: # Explicitly break if user limit reached
                break
    except Exception as e:
        print(f"\nError during metadata fetch: {e}")

    actual_results_count = len(all_results)
    fetch_time = time.time() - start_fetch_time
    print(f"Metadata fetched for {actual_results_count} papers in {fetch_time:.1f} seconds.")

    if not all_results:
        print("No results found.")
        return

    print(f"Starting parallel download of {actual_results_count} PDFs using up to {args.max_workers} workers...")

    # --- Parallel Download ---    
    count_success = 0
    count_failed = 0
    start_download_time = time.time()
    log_lock = threading.Lock()

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"arXiv Download Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Query: {query}\n")
        log.write(f"Requested Max Results: {args.max_results}\n")
        log.write(f"Actual Results Found: {actual_results_count}\n")
        log.write(f"Sort By: {args.sort_by} ({args.sort_order})\n\n")
        log.write("ID, Title, Authors, Published, Updated, Category, Primary Category, PDF Download Status\n")
        log.write("-" * 120 + "\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all download tasks
            future_to_result = {executor.submit(download_paper_task, result, pdf_dir): result for result in all_results}
            
            processed_count = 0
            for future in concurrent.futures.as_completed(future_to_result):
                processed_count += 1
                try:
                    result_obj, pdf_status, filename_base = future.result()
                    
                    if pdf_status == "Success":
                        count_success += 1
                    else:
                        count_failed += 1

                    # Prepare log line
                    authors = ", ".join(str(author) for author in result_obj.authors)
                    categories = ", ".join(result_obj.categories)
                    primary_category = result_obj.primary_category
                    log_line = (f"{result_obj.get_short_id()}, \"{result_obj.title}\", \"{authors}\", "
                                f"{result_obj.published.date()}, {result_obj.updated.date()}, "
                                f"\"{categories}\", \"{primary_category}\", "
                                f"PDF: {pdf_status}\n")

                    # Write to log file safely
                    with log_lock:
                        log.write(log_line)
                    
                    # Update progress on screen
                    elapsed_download = time.time() - start_download_time
                    papers_per_sec = processed_count / elapsed_download if elapsed_download > 0 else 0
                    print(f"\rProgress: {processed_count}/{actual_results_count} ({count_success} success, {count_failed} failed) | Speed: {papers_per_sec:.2f} papers/sec", end="")

                except Exception as exc:
                    count_failed += 1
                    # Retrieve the original result object if possible to log the failure
                    result_obj = future_to_result[future]
                    print(f'\nError processing result for {result_obj.get_short_id()}: {exc}')
                    log_line = f"{result_obj.get_short_id()}, \"{result_obj.title}\", ,, , ,, , PDF: Failed - {exc}\n" # Log basic info on error
                    with log_lock:
                        log.write(log_line)

    # --- Final Summary ---                
    total_elapsed = time.time() - start_fetch_time
    print(f"\n\nDownload complete! {count_success} successful, {count_failed} failed.")
    print(f"Total time: {total_elapsed / 60:.1f} minutes ({fetch_time:.1f}s fetch + {(total_elapsed - fetch_time):.1f}s download).")
    print(f"Log file saved to: {log_file}")
    print(f"PDF files saved to: {pdf_dir}")


if __name__ == "__main__":
    main()