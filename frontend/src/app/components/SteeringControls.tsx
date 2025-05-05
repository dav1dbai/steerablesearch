'use client';

import { useState, useEffect, useRef, useCallback, memo } from 'react';

interface SteeringFeature {
  feature_id: number;
  description: string;
}

interface SteeringParameter {
  feature_id: number;
  strength: number;
}

interface SteeringControlsProps {
  features: SteeringFeature[];
  onSteeringChange: (params: SteeringParameter[]) => void;
  onFeatureSearch?: (query: string) => void;
  isLoading?: boolean;
}

// Emergency feature recovery function
// This function can be used to retrieve features directly from localStorage
// when the props-based features array mysteriously becomes empty
const emergencyFeatureRecovery = (): SteeringFeature[] => {
  try {
    console.log('Attempting emergency feature recovery from localStorage...');
    const storedFeatures = localStorage.getItem('cached_steering_features');
    if (storedFeatures) {
      const parsed = JSON.parse(storedFeatures);
      if (Array.isArray(parsed) && parsed.length > 0) {
        console.log(`SUCCESS: Recovered ${parsed.length} features from localStorage!`);
        return parsed;
      }
    }
    console.warn('Emergency recovery failed: No features in localStorage');
    return [];
  } catch (e) {
    console.error('Emergency feature recovery error:', e);
    return [];
  }
};

// Create a memoized feature item component to prevent unnecessary re-renders
const FeatureItem = memo(({ 
  feature, 
  isSelected, 
  isDisabled, 
  onToggle 
}: { 
  feature: SteeringFeature; 
  isSelected: boolean; 
  isDisabled: boolean; 
  onToggle: () => void; 
}) => {
  // Format feature for display 
  const formatFeatureDisplay = () => {
    const { feature_id, description } = feature;
    const maxLength = 70;
    const truncatedDesc = description.length > maxLength
      ? `${description.substring(0, maxLength)}...`
      : description;
    
    return `ID: ${feature_id} - ${truncatedDesc}`;
  };

  return (
    <div className="flex items-center py-1">
      <input
        type="checkbox"
        id={`feature-${feature.feature_id}`}
        checked={isSelected}
        onChange={onToggle}
        className="claude-checkbox mr-1.5"
        disabled={isDisabled}
      />
      <label 
        htmlFor={`feature-${feature.feature_id}`} 
        className="ml-2 block text-sm text-foreground"
        title={feature.description}
      >
        {formatFeatureDisplay()}
      </label>
    </div>
  );
});

// Create a memoized strength slider component
const StrengthSlider = memo(({ 
  featureId, 
  feature, 
  currentStrength, 
  isDisabled, 
  onStrengthChange 
}: { 
  featureId: number; 
  feature: SteeringFeature | undefined; 
  currentStrength: number; 
  isDisabled: boolean; 
  onStrengthChange: (value: number) => void; 
}) => {
  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium">
          Feature {featureId}:
        </span>
        <span className="text-sm text-clay-orange">
          {currentStrength.toFixed(1)}
        </span>
      </div>
      <input
        type="range"
        min="-10"
        max="10"
        step="0.1"
        value={currentStrength}
        onChange={(e) => onStrengthChange(parseFloat(e.target.value))}
        className="w-full claude-range"
        disabled={isDisabled}
      />
      <div className="text-xs text-foreground opacity-80 truncate mt-1" title={feature?.description}>
        {feature?.description || 'No description available'}
      </div>
    </div>
  );
});

// Limit the number of features that can be displayed to improve performance
const MAX_VISIBLE_FEATURES = 50;

export default function SteeringControls({
  features,
  onSteeringChange,
  onFeatureSearch = () => {},
  isLoading = false,
}: SteeringControlsProps) {
  // Main states
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([]);
  const [strengthValues, setStrengthValues] = useState<Record<number, number>>({});
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // Three-tier feature management:
  // 1. masterFeatures: stable copy of all features that never gets filtered or modified
  // 2. visibleFeatures: filtered subset for display (changes with search)
  const [masterFeatures, setMasterFeatures] = useState<SteeringFeature[]>([]);
  const [visibleFeatures, setVisibleFeatures] = useState<SteeringFeature[]>([]);
  
  // Initialize master features only when we get valid data
  // This ensures we have a stable copy that never disappears
  useEffect(() => {
    if (features && features.length > 0) {
      console.log(`Setting master features (${features.length} items)`);
      setMasterFeatures(features);
      
      // Only update visible features if no search is active
      if (!searchQuery) {
        setVisibleFeatures(features.slice(0, MAX_VISIBLE_FEATURES));
      }
    }
  }, [features, searchQuery]);

  // When selections change, prepare and send the steering parameters to parent
  // We use a debounce ref to limit how often we update
  const prevParamsRef = useRef<string>('');
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Emergency recovery only runs once at startup or if everything is empty
  useEffect(() => {
    if (masterFeatures.length === 0 && features.length === 0) {
      console.log('No features available - attempting recovery from localStorage');
      try {
        const recoveredFeatures = emergencyFeatureRecovery();
        if (recoveredFeatures.length > 0) {
          setMasterFeatures(recoveredFeatures);
          setVisibleFeatures(recoveredFeatures.slice(0, MAX_VISIBLE_FEATURES));
        }
      } catch (e) {
        console.error('Emergency recovery failed:', e);
      }
    }
  }, [masterFeatures.length, features.length]);
  
  // Debounced update function
  const updateSteeringParams = useCallback(() => {
    // Notify parent component of steering changes only when there's an actual change
    const steeringParams = selectedFeatures.map(featureId => ({
      feature_id: featureId,
      strength: strengthValues[featureId] || 2.0,
    }));
    
    // Only call the parent function if there's an actual change
    const paramsString = JSON.stringify(steeringParams);
    if (paramsString !== prevParamsRef.current) {
      prevParamsRef.current = paramsString;
      onSteeringChange(steeringParams);
    }
  }, [selectedFeatures, strengthValues, onSteeringChange]);

  // Schedule debounced updates on parameter changes
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    debounceTimerRef.current = setTimeout(() => {
      updateSteeringParams();
    }, 150); // 150ms debounce
    
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [selectedFeatures, strengthValues, updateSteeringParams]);

  // Feature toggle handler with memoization
  const handleFeatureToggle = useCallback((featureId: number) => {
    setSelectedFeatures(prev => {
      if (prev.includes(featureId)) {
        return prev.filter(id => id !== featureId);
      } else {
        // Limit to max 5 selected features to avoid performance issues
        if (prev.length >= 5) {
          alert('Performance warning: You can select up to 5 features at a time for better responsiveness.');
          return prev;
        }
        
        // Initialize strength to 2.0 when feature is selected
        setStrengthValues(prevStrengths => ({
          ...prevStrengths,
          [featureId]: 2.0,
        }));
        return [...prev, featureId];
      }
    });
  }, []);

  // Strength change handler with memoization  
  const handleStrengthChange = useCallback((featureId: number, strength: number) => {
    setStrengthValues(prev => ({
      ...prev,
      [featureId]: strength,
    }));
  }, []);

  // Memoized search handler to reduce re-renders
  const handleSearch = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value.toLowerCase();
    setSearchQuery(query); // Store query in component state
    onFeatureSearch(query); // Notify parent
    
    // Always search against the master features to maintain consistency
    if (!query.trim()) {
      setVisibleFeatures(masterFeatures.slice(0, MAX_VISIBLE_FEATURES));
    } else {
      const filtered = masterFeatures
        .filter(feature => 
          feature.description.toLowerCase().includes(query) || 
          feature.feature_id.toString().includes(query)
        )
        .slice(0, MAX_VISIBLE_FEATURES);
      
      setVisibleFeatures(filtered);
    }
  }, [masterFeatures, onFeatureSearch]);

  // Handle no features available state - only show this if we have no features at all
  // (but we should never reach this because of all our backups)
  if (visibleFeatures.length === 0 && masterFeatures.length === 0) {
    return (
      <div className="card p-4 my-4">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-clay-orange mb-4"></div>
          <p className="text-foreground text-center">
            Waiting for steering features from backend...
          </p>
          <p className="text-sm text-foreground opacity-70 mt-2 text-center">
            This may take a moment. The features will be cached once loaded.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-4 my-4">
      <h3 className="text-lg font-medium mb-4 font-heading">Manual Steering Controls</h3>
      
      {/* Feature Search */}
      <div className="mb-4">
        <div className="relative mb-3">
          <input
            type="text"
            placeholder="Search features..."
            className="claude-input w-full"
            onChange={handleSearch}
            defaultValue=""
            disabled={isLoading}
          />
          <svg className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        
        {/* Feature Selection */}
        <label className="block text-sm font-medium text-foreground mb-2">
          Select Feature(s) to Steer: <span className="text-clay-orange">(max 5 recommended)</span>
        </label>
        <div className="max-h-60 overflow-y-auto border border-neutral-200 rounded-md p-2">
          {visibleFeatures.map(feature => (
            <FeatureItem 
              key={feature.feature_id}
              feature={feature}
              isSelected={selectedFeatures.includes(feature.feature_id)}
              isDisabled={isLoading}
              onToggle={() => handleFeatureToggle(feature.feature_id)}
            />
          ))}
          {features.length > MAX_VISIBLE_FEATURES && (
            <div className="text-sm text-clay-orange mt-2">
              Showing {visibleFeatures.length} of {features.length} features. Use the search to find more.
            </div>
          )}
        </div>
      </div>
      
      {/* Strength Sliders */}
      {selectedFeatures.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Adjust Strength:
          </label>
          {selectedFeatures.map(featureId => {
            const feature = features.find(f => f.feature_id === featureId);
            const currentStrength = strengthValues[featureId] || 2.0;
            
            return (
              <StrengthSlider
                key={featureId}
                featureId={featureId}
                feature={feature}
                currentStrength={currentStrength}
                isDisabled={isLoading}
                onStrengthChange={(value) => handleStrengthChange(featureId, value)}
              />
            );
          })}
        </div>
      )}
      
      {selectedFeatures.length === 0 && (
        <p className="text-foreground italic">
          Select one or more features above to enable steering.
        </p>
      )}
    </div>
  );
}