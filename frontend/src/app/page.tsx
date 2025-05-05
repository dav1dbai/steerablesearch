'use client';

import { useState, useEffect, useCallback } from 'react';
import SearchBox, { SearchOptions } from './components/SearchBox';
import SteeringControls from './components/SteeringControls';
import AutoSteeringControls from './components/AutoSteeringControls';
import ResultItem from './components/ResultItem';
import QueryInfo from './components/QueryInfo';
import { 
  fetchFeatures, 
  searchDocuments, 
  SearchResponse, 
  Feature, 
  checkPendingRequests,
  setupSSEListener
} from './api/searchApi';

export default function Home() {
  // State variables
  const [features, setFeatures] = useState<Feature[]>([]);
  const [filteredFeatures, setFilteredFeatures] = useState<Feature[]>([]);
  const [featureSearchQuery, setFeatureSearchQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [searchMode, setSearchMode] = useState<'normal' | 'manual-steered' | 'auto-steered'>('normal');
  const [featuresLoading, setFeaturesLoading] = useState<boolean>(true);
  
  // Handle search mode change separately to prevent infinite renders
  // This gets called immediately when the dropdown value changes
  const handleSearchModeChange = useCallback((mode: 'normal' | 'manual-steered' | 'auto-steered') => {
    console.log('Search mode changed to:', mode);
    
    // If trying to use a steering mode without features loaded yet, show an error
    if ((mode === 'manual-steered' || mode === 'auto-steered') && 
        (featuresLoading || features.length === 0)) {
      // Create a local error message to avoid circular dependency
      const errorMsg = 'Features are still loading from the backend. Please try again in a moment.';
      setError(errorMsg);
      return;
    }
    
    setSearchMode(mode);
  }, [featuresLoading, features.length]);
  const [steeringParams, setSteeringParams] = useState<Array<{ feature_id: number; strength: number }>>([]);
  const [autoSteeringParams, setAutoSteeringParams] = useState<{ maxFeatures: number; maxStrength: number }>({
    maxFeatures: 5,
    maxStrength: 8.0,
  });
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // State for pending requests
  const [hasPendingRequests, setHasPendingRequests] = useState<boolean>(false);
  const [pendingRequestsInfo, setPendingRequestsInfo] = useState<any[]>([]);

  // Fetch features on component mount and check for pending requests
  useEffect(() => {
    // Check for any pending requests from previous sessions
    try {
      const { hasPending, requests } = checkPendingRequests();
      setHasPendingRequests(hasPending);
      setPendingRequestsInfo(requests);
      
      if (hasPending) {
        console.log('Found pending requests:', requests);
        // Use local variable for error message to avoid potential circular reference
        const pendingErrorMsg = 'You have pending search requests from a previous session. SAE inference may still be running. Please wait or try again.';
        setError(pendingErrorMsg);
      }
    } catch (err) {
      console.error('Error checking pending requests:', err);
    }
    
    // Feature loading function with retry mechanism
    const loadFeatures = async () => {
      setFeaturesLoading(true);
      try {
        console.log("Fetching features...");
        const featuresData = await fetchFeatures();
        console.log("Features received:", featuresData);
        
        if (featuresData.length > 0) {
          setFeatures(featuresData);
          setFilteredFeatures(featuresData);
          setError(null); // Clear any feature-related errors
        } else {
          // Use local variable for error message to avoid potential circular reference
          const noFeaturesMsg = 'No steering features were loaded. Please make sure the backend server is running at http://localhost:8000';
          setError(noFeaturesMsg);
          
          // Setup feature recovery monitoring
          setupFeatureRecovery();
        }
      } catch (err) {
        console.error('Error loading features:', err);
        // Use local variable for error message to avoid potential circular reference
        const loadErrorMsg = 'Failed to load steering features. Please check if the backend server is running at http://localhost:8000';
        setError(loadErrorMsg);
        
        // Setup feature recovery monitoring
        setupFeatureRecovery();
      } finally {
        setFeaturesLoading(false);
      }
    };
    
    // Setup a recovery mechanism that keeps trying to fetch features
    // This helps recover from temporary backend unavailability
    const setupFeatureRecovery = () => {
      // We'll attempt recovery every 10 seconds
      const recoveryInterval = setInterval(async () => {
        console.log("Attempting feature recovery...");
        try {
          const recoveredFeatures = await fetchFeatures();
          if (recoveredFeatures.length > 0) {
            console.log(`Feature recovery successful: ${recoveredFeatures.length} features loaded`);
            setFeatures(recoveredFeatures);
            setFilteredFeatures(recoveredFeatures);
            setError(null); // Clear any feature-related errors
            clearInterval(recoveryInterval); // Stop recovery attempts
          }
        } catch (err) {
          console.warn("Feature recovery attempt failed:", err);
        }
      }, 10000); // Try every 10 seconds
      
      // Store the interval ID for cleanup
      featureRecoveryInterval = recoveryInterval;
      return recoveryInterval;
    };

    // Initial feature loading
    loadFeatures();
    
    // Set up Server-Sent Events listener for real-time updates from the server
    const cleanupSSE = setupSSEListener((data) => {
      console.log('Received server update:', data);
      
      // If we receive a completed request update
      if (data.type === 'request_completed' && data.request_id) {
        try {
          // Update localStorage
          const key = `search_request_${data.request_id}`;
          const savedRequest = localStorage.getItem(key);
          if (savedRequest) {
            const parsed = JSON.parse(savedRequest);
            parsed.status = 'completed';
            parsed.result = data.result;
            localStorage.setItem(key, JSON.stringify(parsed));
            
            // If we have results, update the UI
            if (data.result) {
              setSearchResults(data.result);
              setIsLoading(false);
              
              // Update the pending requests state
              const { hasPending, requests } = checkPendingRequests();
              setHasPendingRequests(hasPending);
              setPendingRequestsInfo(requests);
            }
          }
        } catch (e) {
          console.warn('Error processing server update:', e);
        }
      }
    });
    
    // Set up a polling interval to check for pending requests every 5 seconds
    const intervalId = setInterval(() => {
      const { hasPending, requests } = checkPendingRequests();
      setHasPendingRequests(hasPending);
      setPendingRequestsInfo(requests);
    }, 5000);
    
    // Store any feature recovery interval for cleanup
    let featureRecoveryInterval: NodeJS.Timeout | null = null;
    
    return () => {
      clearInterval(intervalId);
      cleanupSSE(); // Clean up SSE listener
      
      // Clean up feature recovery interval if it exists
      if (featureRecoveryInterval) {
        clearInterval(featureRecoveryInterval);
      }
    };
  }, []);
  
  // Filter features based on search query
  useEffect(() => {
    if (!featureSearchQuery.trim()) {
      setFilteredFeatures(features);
      return;
    }
    
    const query = featureSearchQuery.toLowerCase();
    const filtered = features.filter(feature => 
      feature.description.toLowerCase().includes(query) || 
      feature.feature_id.toString().includes(query)
    );
    setFilteredFeatures(filtered);
  }, [featureSearchQuery, features]);

  // Handle search - memoized with dependencies to avoid unnecessary re-renders
  const handleSearch = useCallback(async (query: string, options: SearchOptions) => {
    setIsLoading(true);
    setError(null);
    
    // First try to change the search mode
    // If this returns an error (e.g., features not loaded), don't continue with search
    const prevMode = searchMode;
    handleSearchModeChange(options.searchMode);
    
    // If the mode didn't change due to an error, abort the search
    if (options.searchMode !== 'normal' && searchMode === prevMode && 
        (featuresLoading || features.length === 0)) {
      setIsLoading(false);
      return;
    }

    try {
      const results = await searchDocuments(query, {
        ...options,
        steeringParams,
        autoSteeringParams,
      });
      
      setSearchResults(results);
    } catch (err) {
      console.error('Search error:', err);
      // Use local variable for error message to avoid potential circular reference
      const errorMsg = 'An error occurred during search. Please try again or check if the backend server is running.';
      setError(errorMsg);
      setSearchResults(null);
    } finally {
      setIsLoading(false);
    }
  }, [steeringParams, autoSteeringParams, handleSearchModeChange, searchMode, featuresLoading, features.length]);

  // Handle steering parameter changes - memoized to prevent infinite loops
  const handleSteeringChange = useCallback((params: Array<{ feature_id: number; strength: number }>) => {
    setSteeringParams(params);
  }, []);

  // Handle auto-steering parameter changes - memoized to prevent infinite loops
  const handleAutoSteeringChange = useCallback((params: { maxFeatures: number; maxStrength: number }) => {
    setAutoSteeringParams(params);
  }, []);
  
  // Handle feature search - memoized to prevent infinite loops
  const handleFeatureSearch = useCallback((query: string) => {
    setFeatureSearchQuery(query);
  }, []);

  return (
    <div>
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-heading mb-3 font-medium tracking-tight">Weaver</h1>
        <div className="inline-flex items-center claude-pill mb-6">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1.5">
            <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
            <path fillRule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
          </svg>
          Steerable Search with Claude
        </div>
        <p className="text-foreground opacity-80 max-w-2xl mx-auto font-body text-lg">
          Discover academic papers with intelligent search steering powered by sparse autoencoders.
          Control how information is conveyed through manual or automated embedding features.
        </p>
        <div className="mt-3 text-xs text-foreground/60 max-w-xl mx-auto">
          <p><em>Named after Warren Weaver (1894-1978), Shannon's collaborator who made information theory accessible to broader audiences.</em></p>
        </div>
      </div>

      {/* Search Box */}
      <SearchBox 
        onSearch={handleSearch} 
        onModeChange={handleSearchModeChange}
        isLoading={isLoading || featuresLoading} 
      />

      {/* Error Message */}
      {error && (
        <div className="my-4 p-4 bg-red-50 border border-red-200 rounded-md text-red-700 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>{error}</span>
        </div>
      )}
      
      {/* Pending Requests Notification */}
      {hasPendingRequests && (
        <div className="my-4 p-4 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-700">
          <div className="flex items-center mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 flex-shrink-0 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium">SAE inference in progress...</span>
          </div>
          <p className="text-sm">
            This page has {pendingRequestsInfo.length} pending search request(s). SAE inference may take several minutes. The page will update when results are ready.
          </p>
        </div>
      )}

      {/* Always render SteeringControls but hide them when not in manual mode */}
      <div className={searchMode === 'manual-steered' ? 'block' : 'hidden'}>
        {featuresLoading ? (
          <div className="card p-4 my-4 flex justify-center items-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-clay-orange mr-3"></div>
            <span className="text-foreground">Loading steering features...</span>
          </div>
        ) : (
          <SteeringControls
            features={filteredFeatures}
            onSteeringChange={handleSteeringChange}
            onFeatureSearch={handleFeatureSearch}
            isLoading={isLoading}
          />
        )}
      </div>

      {/* Always render AutoSteeringControls but hide them when not in auto mode */}
      <div className={searchMode === 'auto-steered' ? 'block' : 'hidden'}>
        {featuresLoading ? (
          <div className="card p-4 my-4 flex justify-center items-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-clay-orange mr-3"></div>
            <span className="text-foreground">Loading steering features...</span>
          </div>
        ) : (
          <AutoSteeringControls
            onAutoSteeringChange={handleAutoSteeringChange}
            isLoading={isLoading}
          />
        )}
      </div>
      
      {/* Two-Column Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
        {/* Sidebar Controls */}
        <div className="md:col-span-1">
          {/* Query Analysis Info */}
          {searchResults && (
            <>
              <QueryInfo
                rewrittenQuery={searchResults.rewritten_query}
                filteredCount={searchResults.filtered_count ? searchResults.filtered_count : undefined}
                autoSteeringInfo={searchResults.auto_steering}
              />
            </>
          )}
        </div>

        {/* Results Column */}
        <div className="md:col-span-2">
          {/* No title needed */}

          {/* Loading State */}
          {isLoading && (
            <div className="flex justify-center items-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-clay-orange"></div>
            </div>
          )}

          {/* No Results */}
          {!isLoading && searchResults && searchResults.results.length === 0 && (
            <div className="bg-clay-orange-light border border-clay-orange-light rounded-md p-8 text-center">
              <p className="text-clay-orange-dark">No results found for your query.</p>
            </div>
          )}

          {/* Results List */}
          {!isLoading && searchResults && searchResults.results.length > 0 && (
            <div>
              {/* Steering Info */}
              {searchMode === 'manual-steered' && searchResults.steering_info && (
                <div className="mb-4 p-3 bg-clay-orange-light border border-clay-orange-light rounded-md">
                  <h3 className="text-sm font-medium mb-2 font-heading">Applied Steering Features:</h3>
                  {searchResults.steering_info.length === 0 ? (
                    <p className="text-sm text-clay-orange-dark italic">No steering features were applied by the backend.</p>
                  ) : (
                    <div className="space-y-2">
                      {searchResults.steering_info.map((info, idx) => (
                        <div key={idx} className="text-sm">
                          <span className="font-medium">Feature ID:</span> {info.feature_id} | 
                          <span className="font-medium ml-2">Strength:</span> {info.strength.toFixed(2)}
                          {info.explanation && (
                            <div className="text-xs text-clay-orange-dark mt-1">
                              {info.explanation}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Results */}
              <div className="space-y-4">
                {searchResults.results.map((result, idx) => (
                  <ResultItem
                    key={idx}
                    index={idx}
                    score={result.score}
                    metadata={result.metadata}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
