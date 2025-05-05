'use client';

import { useEffect, useRef } from 'react';

interface AutoSteeringControlsProps {
  onAutoSteeringChange: (params: { maxFeatures: number; maxStrength: number }) => void;
  isLoading?: boolean;
}

export default function AutoSteeringControls({
  onAutoSteeringChange,
  isLoading = false,
}: AutoSteeringControlsProps) {
  // Default values
  const DEFAULT_MAX_FEATURES = 5;
  const DEFAULT_MAX_STRENGTH = 8.0;
  
  // Send default values to parent only once
  const sentDefaultsRef = useRef(false);
  
  useEffect(() => {
    if (!sentDefaultsRef.current) {
      onAutoSteeringChange({ 
        maxFeatures: DEFAULT_MAX_FEATURES, 
        maxStrength: DEFAULT_MAX_STRENGTH 
      });
      sentDefaultsRef.current = true;
    }
  }, [onAutoSteeringChange]);

  return (
    <div className="card card-highlight p-5 my-4">
      <div className="flex items-center mb-4">
        <div className="mr-3">
          <div className="w-10 h-10 bg-clay-orange rounded-full flex items-center justify-center shadow-md">
            <span className="text-xl">🤖</span>
          </div>
        </div>
        <h3 className="text-xl font-medium font-heading text-clay-orange-dark">
          Claude Takes the Wheel
        </h3>
      </div>
      
      <div className="p-4 bg-clay-orange-light border border-clay-orange-light rounded-md shadow-sm">
        <div className="flex items-start">
          <div className="mt-1 mr-3">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9.5 14.5L12 17L14.5 14.5M12 7V17" stroke="#B55F41" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="12" r="9" stroke="#B55F41" strokeWidth="2"/>
            </svg>
          </div>
          <div>
            <p className="text-sm text-clay-orange-dark">
              Claude will analyze your query and automatically select the most relevant features 
              and strengths. Just sit back and <span className="font-medium">enjoy the ride!</span>
            </p>
          </div>
        </div>
      </div>
      
      {isLoading && (
        <div className="mt-4 flex justify-center items-center">
          <div className="inline-flex items-center px-3 py-2 text-sm text-clay-orange-dark bg-clay-orange-light bg-opacity-30 rounded-full">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-clay-orange" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Claude is analyzing your query and selecting the best features...
          </div>
        </div>
      )}
    </div>
  );
}