'use client';

// Use environment variable or default to localhost for the API URL
const API_BASE_URL = 'http://localhost:8000';

// Store active requests by ID
const pendingRequests = new Map();

// Long timeout for SAE inference (3 minutes)
const LONG_TIMEOUT = 3 * 60 * 1000; 

// Generate a unique ID for each request
const generateRequestId = () => {
  return `req_${Math.random().toString(36).substring(2, 9)}_${Date.now()}`;
};

export interface SearchOptions {
  rewriteQuery: boolean;
  filterNoise: boolean;
  topK: number;
  searchMode: 'normal' | 'manual-steered' | 'auto-steered';
  steeringParams?: Array<{ feature_id: number; strength: number }>;
  autoSteeringParams?: { maxFeatures: number; maxStrength: number };
}

export interface SearchResult {
  score: number;
  metadata: {
    source?: string;
    arxiv_id?: string;
    paper_url?: string;
    paper_title?: string;
    display_text?: string;
    text?: string;
    chunk_index?: number;
    [key: string]: any;
  };
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  filtered_count?: number;
  rewritten_query?: string;
  original_query?: string;
  steering_info?: Array<{
    feature_id: number;
    strength: number;
    explanation?: string;
    max_activation_used?: number;
  }>;
  auto_steering?: {
    query_intent: {
      key_concepts?: string[];
      technical_level?: string;
      perspective?: string;
      domains?: string[];
      content_type?: string;
    };
    selected_features: Array<{
      feature_id: number;
      strength: number;
      explanation?: string;
      relevance?: string;
    }>;
  };
}

export interface Feature {
  feature_id: number;
  description: string;
}

// Setup a server-sent events (SSE) listener for status updates
export const setupSSEListener = (onUpdate: (data: any) => void): (() => void) => {
  try {
    const eventSource = new EventSource(`${API_BASE_URL}/status-updates`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.warn('SSE connection error:', error);
      // Auto-reconnect is handled by the browser
    };
    
    // Return a cleanup function
    return () => {
      eventSource.close();
    };
  } catch (error) {
    console.warn('Failed to setup SSE listener:', error);
    return () => {}; // Empty cleanup function if setup fails
  }
};

// Check if the server is running
export const checkServerStatus = async (): Promise<boolean> => {
  try {
    console.log('Checking if server is running at:', API_BASE_URL);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${API_BASE_URL}`, {
      method: 'GET',
      signal: controller.signal,
    }).finally(() => clearTimeout(timeoutId));
    
    return response.ok;
  } catch (error) {
    console.error('Server connection failed:', error);
    return false;
  }
};

// Function to check for any pending requests and clean up old ones
export const checkPendingRequests = (): {hasPending: boolean, requests: any[]} => {
  const now = Date.now();
  const pendingRequestsData: any[] = [];
  let hasPendingRequests = false;
  
  try {
    // Find all search requests in localStorage
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('search_request_')) {
        try {
          const savedRequest = JSON.parse(localStorage.getItem(key) || '{}');
          const requestAge = now - savedRequest.timestamp;
          
          // If the request is from the last hour and still pending, we consider it active
          if (savedRequest.status === 'pending' && requestAge < 60 * 60 * 1000) {
            hasPendingRequests = true;
            pendingRequestsData.push(savedRequest);
          } 
          // Clean up completed or very old requests
          else if (savedRequest.status === 'completed' || requestAge > 24 * 60 * 60 * 1000) {
            localStorage.removeItem(key);
          }
        } catch (e) {
          console.warn('Error parsing saved request:', e);
          localStorage.removeItem(key); // Remove invalid entries
        }
      }
    }
  } catch (e) {
    console.warn('Error checking pending requests:', e);
  }
  
  return {
    hasPending: hasPendingRequests,
    requests: pendingRequestsData
  };
};

// Store the features in memory to prevent them from disappearing
// We use a Map with a sentinel value in case we need to debug feature disappearance
const FEATURE_CACHE = new Map<string, Feature[]>();
const CACHE_KEY = 'global_features_v1';

// Function to get cached features
const getCachedFeatures = (): Feature[] => {
  if (FEATURE_CACHE.has(CACHE_KEY)) {
    return FEATURE_CACHE.get(CACHE_KEY) || [];
  }
  
  // Try to load from localStorage if not in memory
  try {
    const storedFeatures = localStorage.getItem('cached_steering_features');
    if (storedFeatures) {
      const parsed = JSON.parse(storedFeatures);
      if (Array.isArray(parsed) && parsed.length > 0) {
        // Store in the in-memory cache
        FEATURE_CACHE.set(CACHE_KEY, parsed);
        console.log(`Loaded ${parsed.length} cached features from localStorage`);
        return parsed;
      }
    }
  } catch (e) {
    console.warn('Failed to load cached features from localStorage:', e);
  }
  
  return [];
};

// Function to set cached features
const setCachedFeatures = (features: Feature[]): void => {
  if (!Array.isArray(features) || features.length === 0) {
    console.warn('Attempted to cache invalid feature data');
    return;
  }
  
  // Set in memory cache
  FEATURE_CACHE.set(CACHE_KEY, features);
  console.log(`Set ${features.length} features in memory cache`);
  
  // Also persist to localStorage
  try {
    localStorage.setItem('cached_steering_features', JSON.stringify(features));
    console.log(`Saved ${features.length} features to localStorage`);
  } catch (e) {
    console.warn('Failed to cache features in localStorage:', e);
  }
};

// Initialize cache from localStorage if available
getCachedFeatures();

export const fetchFeatures = async (): Promise<Feature[]> => {
  try {
    // Return cached features immediately if we have them
    // This provides instant access to features even if the server is temporarily unavailable
    const cachedFeatures = getCachedFeatures();
    if (cachedFeatures.length > 0) {
      console.log(`Returning ${cachedFeatures.length} cached features while fetching fresh data`);
      
      // Start a background fetch to update the cache for next time
      fetchFreshFeatures().catch(err => 
        console.warn('Background feature refresh failed:', err)
      );
      
      return cachedFeatures;
    }
    
    // First check if server is running
    const isServerRunning = await checkServerStatus();
    if (!isServerRunning) {
      console.error('Server is not running');
      return getCachedFeatures(); // Return whatever we have in cache, even if empty
    }
    
    console.log('Fetching features from:', `${API_BASE_URL}/features`);
    
    // Use a longer timeout for features (30 seconds)
    // Features can take a long time to load on first server startup
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    
    // Try up to 3 times with exponential backoff
    let attempt = 0;
    const maxAttempts = 3;
    const baseDelay = 2000; // Start with 2 seconds delay
    
    while (attempt < maxAttempts) {
      try {
        const response = await fetch(`${API_BASE_URL}/features`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
          mode: 'cors',
          signal: controller.signal,
        });
        
        if (response.ok) {
          const data = await response.json();
          clearTimeout(timeoutId);
          
          if (Array.isArray(data) && data.length > 0) {
            console.log(`Successfully fetched ${data.length} features on attempt ${attempt + 1}`);
            
            // Update our cache
            setCachedFeatures(data);
            
            // We need to also return this exact data reference, not get from cache
            // This ensures the component gets the exact same reference it's expecting
            return data;
          } else if (Array.isArray(data) && data.length === 0) {
            console.warn('Server returned empty features array, will retry...');
          } else {
            console.warn('Server returned invalid features data, will retry...');
          }
        } else {
          console.warn(`Error fetching features (attempt ${attempt + 1}): ${response.status} ${response.statusText}`);
        }
      } catch (err) {
        console.warn(`Fetch error on attempt ${attempt + 1}:`, err);
      }
      
      // Exponential backoff
      const delay = baseDelay * Math.pow(2, attempt);
      console.log(`Retrying in ${delay/1000} seconds...`);
      
      // Wait before next attempt
      await new Promise(resolve => setTimeout(resolve, delay));
      attempt++;
    }
    
    // If we get here, all attempts failed
    clearTimeout(timeoutId);
    console.error(`Failed to fetch features after ${maxAttempts} attempts`);
    
    // Return cached features as fallback
    return getCachedFeatures();
  } catch (error) {
    console.error('Failed to fetch features:', error);
    return getCachedFeatures(); // Return cache even if empty
  }
};

// Function to fetch fresh features in the background without blocking the UI
const fetchFreshFeatures = async (): Promise<Feature[]> => {
  try {
    console.log('Background refresh: Fetching fresh features from backend');
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    
    const response = await fetch(`${API_BASE_URL}/features`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      mode: 'cors',
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      
      if (Array.isArray(data) && data.length > 0) {
        console.log(`Background refresh: Updated ${data.length} features`);
        
        // Update cache using our utility function
        setCachedFeatures(data);
        
        return data;
      }
    }
    
    return getCachedFeatures();
  } catch (error) {
    console.warn('Background feature refresh failed:', error);
    return getCachedFeatures();
  }
};

export const searchDocuments = async (
  query: string,
  options: SearchOptions
): Promise<SearchResponse> => {
  const { 
    rewriteQuery, 
    filterNoise, 
    topK, 
    searchMode, 
    steeringParams,
    autoSteeringParams
  } = options;
  
  // Generate a unique request ID that will persist across page reloads
  const requestId = generateRequestId();
  
  try {
    // First check if server is running
    const isServerRunning = await checkServerStatus();
    if (!isServerRunning) {
      throw new Error("Cannot connect to backend server at " + API_BASE_URL);
    }
    
    let endpoint: string;
    let payload: any = { 
      query, 
      top_k: topK, 
      filter_noise: filterNoise,
      rewrite_query: rewriteQuery,
      request_id: requestId // Include request ID for tracking
    };
    
    // Save request to localStorage for persistence across reloads
    try {
      localStorage.setItem(`search_request_${requestId}`, JSON.stringify({
        timestamp: Date.now(),
        query,
        options,
        status: 'pending'
      }));
    } catch (e) {
      console.warn('Could not save request to localStorage:', e);
    }
    
    switch (searchMode) {
      case 'manual-steered':
        endpoint = `${API_BASE_URL}/steered_search`;
        payload = {
          ...payload,
          steering_params: steeringParams || []
        };
        break;
      
      case 'auto-steered':
        endpoint = `${API_BASE_URL}/auto_steered_search`;
        if (autoSteeringParams) {
          payload = {
            ...payload,
            max_features: autoSteeringParams.maxFeatures,
            max_strength: autoSteeringParams.maxStrength
          };
        }
        break;
      
      default:
        endpoint = `${API_BASE_URL}/search`;
    }
    
    console.log('Sending search request to:', endpoint, 'with ID:', requestId);
    const controller = new AbortController();
    
    // Store the controller so we can cancel the request if needed
    pendingRequests.set(requestId, {
      controller,
      timestamp: Date.now(),
      query,
      endpoint
    });
    
    // Use a much longer timeout for SAE inference
    const timeoutId = setTimeout(() => controller.abort(), LONG_TIMEOUT);
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-Request-ID': requestId, // Include request ID in headers too
        },
        body: JSON.stringify(payload),
        mode: 'cors',
        signal: controller.signal,
        // Indicate preference for keeping the connection alive
        keepalive: true
      }).finally(() => {
        clearTimeout(timeoutId);
        pendingRequests.delete(requestId);
        
        // Update localStorage status
        try {
          const savedRequest = localStorage.getItem(`search_request_${requestId}`);
          if (savedRequest) {
            const parsed = JSON.parse(savedRequest);
            parsed.status = 'completed';
            localStorage.setItem(`search_request_${requestId}`, JSON.stringify(parsed));
          }
        } catch (e) {
          console.warn('Could not update request in localStorage:', e);
        }
      });
      
      if (!response.ok) {
        throw new Error(`Error searching documents: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      return result;
    } catch (fetchError) {
      // Update localStorage status on error
      try {
        const savedRequest = localStorage.getItem(`search_request_${requestId}`);
        if (savedRequest) {
          const parsed = JSON.parse(savedRequest);
          parsed.status = 'error';
          parsed.error = fetchError.message;
          localStorage.setItem(`search_request_${requestId}`, JSON.stringify(parsed));
        }
      } catch (e) {
        console.warn('Could not update request in localStorage:', e);
      }
      
      if (fetchError.name === 'AbortError') {
        throw new Error('Search request timed out. SAE inference may take longer than expected. Please try again later.');
      }
      throw fetchError;
    }
  } catch (error) {
    console.error('Failed to search documents:', error);
    throw error;
  }
};