'use client';

interface ResultItemProps {
  index: number;
  score: number;
  metadata: {
    source?: string;
    arxiv_id?: string;
    paper_url?: string;
    paper_title?: string;
    display_text?: string;
    text?: string;
    chunk_index?: number;
  };
}

export default function ResultItem({ index, score, metadata }: ResultItemProps) {
  const { 
    source = 'Unknown source', 
    arxiv_id = '', 
    paper_url = '', 
    paper_title = '',
    display_text, 
    text = 'No text available', 
    chunk_index = 'N/A'
  } = metadata;
  
  const displayText = display_text || text;
  
  // Always use the paper title if it exists
  const displayTitle = paper_title || (arxiv_id ? `arXiv:${arxiv_id}` : source);
  
  return (
    <div className="result-item">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1 mr-3">
          <h3 className="text-lg font-medium mb-1 font-heading">
            {arxiv_id ? (
              <a 
                href={paper_url || `https://arxiv.org/abs/${arxiv_id}`} 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                {displayTitle}
              </a>
            ) : (
              <span>{displayTitle}</span>
            )}
          </h3>
          <div className="text-xs text-clay-orange-dark font-body">
            <span className="mr-2">arXiv:{arxiv_id}</span>
            <span>(Chunk {chunk_index})</span>
          </div>
        </div>
        <span className="text-sm bg-clay-orange-light text-clay-orange-dark px-2 py-1 rounded-full flex-shrink-0 whitespace-nowrap">
          Score: {score.toFixed(4)}
        </span>
      </div>
      <p className="text-foreground whitespace-pre-line font-body">{displayText}</p>
    </div>
  );
}