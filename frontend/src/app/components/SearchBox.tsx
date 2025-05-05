'use client';

import { useState, useEffect } from 'react';

interface SearchBoxProps {
  onSearch: (query: string, options: SearchOptions) => void;
  onModeChange?: (mode: 'normal' | 'manual-steered' | 'auto-steered') => void;
  isLoading?: boolean;
}

export interface SearchOptions {
  rewriteQuery: boolean;
  filterNoise: boolean;
  topK: number;
  searchMode: 'normal' | 'manual-steered' | 'auto-steered';
}

export default function SearchBox({ onSearch, onModeChange, isLoading = false }: SearchBoxProps) {
  const [query, setQuery] = useState('');
  const [rewriteQuery, setRewriteQuery] = useState(true);
  const [filterNoise, setFilterNoise] = useState(true);
  const [topK, setTopK] = useState(5);
  const [searchMode, setSearchMode] = useState<'normal' | 'manual-steered' | 'auto-steered'>('normal');

  // Memoize the search parameters to prevent unnecessary re-renders
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    onSearch(query, {
      rewriteQuery: true, // Always on
      filterNoise: true,  // Always on
      topK,
      searchMode
    });
  };

  return (
    <div className="card p-6">
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search papers..."
              className="claude-input font-heading text-lg"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 claude-btn py-1.5 px-3"
              disabled={isLoading || !query.trim()}
            >
              {isLoading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Searching
                </span>
              ) : 'Search'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">Search Mode</label>
            <select
              value={searchMode}
              onChange={(e) => {
                const newMode = e.target.value as 'normal' | 'manual-steered' | 'auto-steered';
                setSearchMode(newMode);
                // Notify parent component immediately when mode changes
                if (onModeChange) {
                  onModeChange(newMode);
                }
              }}
              className="claude-input claude-select"
              disabled={isLoading}
            >
              <option value="normal">Normal Search</option>
              <option value="manual-steered">Manual Steering</option>
              <option value="auto-steered">Auto-Steering</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-1">Number of Results</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              min={1}
              max={50}
              className="claude-input claude-select"
              disabled={isLoading}
            />
          </div>
        </div>
      </form>
    </div>
  );
}