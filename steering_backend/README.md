# Steerable Search Backend

This directory contains the backend server for the steerable search application.

## ArXiv Paper Metadata

The backend now supports displaying paper titles in search results. To enable this feature, follow these steps:

1. First, install the required dependencies:

```bash
pip install requests xmltodict tqdm
```

2. Run the metadata fetching script to get paper titles for your existing ArXiv PDFs:

```bash
python fetch_arxiv_titles.py
```

This script will:
- Scan the `./arxiv` directory for PDF files
- Extract ArXiv IDs from the filenames
- Fetch paper metadata (titles, authors, etc.) from the ArXiv API
- Save the metadata to `./arxiv/arxiv_metadata.json`

3. Restart the server to load the metadata:

```bash
uvicorn steering_backend.server:app --reload --port 8000
```

The server will automatically load the metadata file and include paper titles in search results.

## Advanced Options

### Custom Directories

```bash
python fetch_arxiv_titles.py --directory /path/to/arxiv/pdfs --output /path/to/output.json
```

### Rate Limiting

The script respects ArXiv API rate limits by default. If you need to adjust the delay between requests:

```bash
# Edit RATE_LIMIT_SLEEP variable in fetch_arxiv_titles.py
```

## Troubleshooting

If paper titles aren't appearing in search results:

1. Verify the metadata file exists: `./arxiv/arxiv_metadata.json`
2. Check the server logs for errors loading the metadata
3. Ensure your PDF filenames match the ArXiv ID format: `YYMM.NNNNN.pdf` or `YYMM.NNNNNvN.pdf`
4. Try running the fetch_arxiv_titles.py script again