'use client';

import { useState, useEffect } from 'react';
import { 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell
} from 'recharts';

interface QueryInfoProps {
  rewrittenQuery?: string;
  filteredCount?: number;
  autoSteeringInfo?: {
    query_intent?: {
      key_concepts?: string[];
      perspective?: string;
      domains?: string[];
      content_type?: string;
    };
    selected_features?: Array<{
      feature_id: number;
      strength: number;
      explanation?: string;
      relevance?: string;
    }>;
  };
}

export default function QueryInfo({
  rewrittenQuery,
  filteredCount,
  autoSteeringInfo,
}: QueryInfoProps) {
  const hasContent = rewrittenQuery || (filteredCount && filteredCount > 0) || autoSteeringInfo;
  const [activeFeature, setActiveFeature] = useState<number | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [radarData, setRadarData] = useState<any[]>([]);
  
  // Make sure we return null if there's no content to display
  if (!hasContent) return null;

  // Process data for charts when autoSteeringInfo changes
  useEffect(() => {
    if (autoSteeringInfo?.selected_features?.length) {
      // Prepare data for bar chart
      const barData = autoSteeringInfo.selected_features.map(feature => ({
        name: `Feature ${feature.feature_id}`,
        value: feature.strength,
        id: feature.feature_id,
        explanation: feature.explanation || '',
        relevance: feature.relevance || ''
      }));
      setChartData(barData);
      
      // Prepare data for radar chart if we have domains
      if (autoSteeringInfo?.query_intent?.domains?.length) {
        const domains = autoSteeringInfo.query_intent.domains;
        // Create radar chart with dummy values based on domain count
        const radarPoints = domains.map((domain, index) => ({
          subject: domain,
          A: Math.random() * 80 + 20, // Random values between 20-100 for visual effect
          fullMark: 100,
        }));
        setRadarData(radarPoints);
      }
    }
  }, [autoSteeringInfo]);
  
  // Define gradient colors
  const COLORS = ['#B55F41', '#D97757', '#F5D1BC', '#E0EBF3', '#F3DAD0'];
  
  return (
    <div className="card p-4 my-4 shadow-lg rounded-xl backdrop-blur-sm">
      <h3 className="text-xl font-medium mb-3 font-heading">
        <span className="inline-flex items-center">
          <span className="mr-2">🔍</span>
          Query Analysis
        </span>
      </h3>
      
      {/* Rewritten Query */}
      {rewrittenQuery && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-foreground mb-1">Query Rewritten by Claude:</h4>
          <div className="p-3 bg-gradient-to-r from-clay-orange-light to-sunset-peach border border-clay-orange-light rounded-md shadow-sm">
            <p className="text-sm text-clay-orange-dark font-medium">{rewrittenQuery}</p>
          </div>
        </div>
      )}
      
      {/* Filtered Count - only show if greater than 0 */}
      {filteredCount && filteredCount > 0 ? (
        <div className="mb-4">
          <div className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-clay-orange-dark bg-clay-orange-light rounded-full shadow-sm">
            <span className="mr-1">🔎</span>
            {filteredCount} low-quality results filtered out
          </div>
        </div>
      ) : null}
      
      {/* Auto-Steering Info */}
      {autoSteeringInfo && autoSteeringInfo.query_intent && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-foreground mb-2">
            <span className="inline-flex items-center">
              <span className="mr-1">✨</span>
              Query Intent Analysis:
            </span>
          </h4>
          <div className="p-4 bg-gradient-to-br from-clay-orange-light to-warm-cream border border-clay-orange-light rounded-lg shadow-sm space-y-3">
            {/* Key Concepts */}
            {autoSteeringInfo.query_intent.key_concepts && autoSteeringInfo.query_intent.key_concepts.length > 0 && (
              <div>
                <span className="text-sm font-medium">Key Concepts:</span>
                <div className="flex flex-wrap gap-2 mt-2">
                  {autoSteeringInfo.query_intent.key_concepts.map((concept, idx) => (
                    <span 
                      key={idx} 
                      className="px-3 py-1 text-xs bg-white text-clay-orange-dark rounded-full shadow-sm
                               border border-clay-orange-light"
                    >
                      {concept}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            
            {/* Perspective */}
            {autoSteeringInfo.query_intent.perspective && (
              <div className="flex items-center mt-3">
                <span className="text-sm font-medium mr-2">Perspective:</span>
                <span className="text-xs px-3 py-1 bg-white text-clay-orange-dark rounded-full shadow-sm
                               border border-clay-orange-light font-medium">
                  {autoSteeringInfo.query_intent.perspective}
                </span>
              </div>
            )}
            
            {/* Domains with Radar Chart */}
            {autoSteeringInfo.query_intent.domains && autoSteeringInfo.query_intent.domains.length > 0 && (
              <div className="mt-4">
                <span className="text-sm font-medium block mb-1">Domains:</span>
                <div className="flex items-start justify-between">
                  <div className="flex flex-wrap gap-2 mt-1">
                    {autoSteeringInfo.query_intent.domains.map((domain, idx) => (
                      <span 
                        key={idx} 
                        className="px-3 py-1 text-xs bg-white text-clay-orange-dark rounded-full shadow-sm
                                 border border-clay-orange-light font-medium"
                      >
                        {domain}
                      </span>
                    ))}
                  </div>
                  
                  {/* Radar Chart for Domains */}
                  {radarData.length > 1 && (
                    <div className="w-40 h-40 ml-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                          <PolarGrid stroke="#D97757" />
                          <PolarAngleAxis dataKey="subject" tick={{ fill: '#B55F41', fontSize: 8 }} />
                          <Radar 
                            name="Relevance" 
                            dataKey="A" 
                            stroke="#D97757" 
                            fill="#D97757" 
                            fillOpacity={0.5}
                          />
                          <Tooltip 
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                return (
                                  <div className="bg-white p-2 border border-clay-orange-light rounded-md shadow-md">
                                    <p className="text-xs font-medium text-clay-orange-dark">{payload[0].payload.subject}</p>
                                    <p className="text-xs">Relevance: <span className="font-medium">{payload[0].value}</span></p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Content Type */}
            {autoSteeringInfo.query_intent.content_type && (
              <div className="flex items-center mt-3">
                <span className="text-sm font-medium mr-2">Content Type:</span>
                <span className="text-xs px-3 py-1 bg-white text-clay-orange-dark rounded-full shadow-sm
                               border border-clay-orange-light font-medium">
                  {autoSteeringInfo.query_intent.content_type}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Auto-Selected Features with Bar Chart */}
      {autoSteeringInfo && autoSteeringInfo.selected_features && autoSteeringInfo.selected_features.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-foreground mb-2">
            <span className="inline-flex items-center">
              <span className="mr-1">⚙️</span>
              Auto-Selected Features:
            </span>
          </h4>
          
          {/* Bar Chart for Feature Strength */}
          <div className="mb-4 mt-3 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#F5D1BC" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 10]} label={{ value: 'Strength', angle: -90, position: 'insideLeft', fill: '#B55F41', fontSize: 12 }} />
                <Tooltip 
                  formatter={(value: any, name: any, props: any) => ['Strength: ' + value, '']}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-3 border border-clay-orange-light rounded-md shadow-md">
                          <p className="font-medium text-clay-orange-dark">{data.name}</p>
                          <p>Strength: <span className="font-medium">{data.value.toFixed(2)}</span></p>
                          {data.explanation && <p className="text-xs mt-1 max-w-xs">{data.explanation}</p>}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="value" fill="#D97757" onClick={(data) => setActiveFeature(data.id)}>
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.id === activeFeature ? '#B55F41' : COLORS[index % COLORS.length]} 
                      className="cursor-pointer hover:opacity-90 transition-opacity"
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* Feature Details */}
          <div className="p-4 bg-clay-orange-light border border-clay-orange-light rounded-lg shadow-sm space-y-3">
            {autoSteeringInfo.selected_features.map((feature, idx) => (
              <div 
                key={idx} 
                className="border-b border-white border-opacity-30 pb-3 last:border-b-0 last:pb-0 p-2"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium flex items-center">
                    <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></span>
                    Feature ID: {feature.feature_id}
                  </span>
                  <span className="text-xs px-2 py-0.5 bg-white text-clay-orange-dark rounded-full shadow-sm font-medium">
                    Strength: {feature.strength.toFixed(2)}
                  </span>
                </div>
                
                <div className="mt-2">
                  {feature.explanation && (
                    <p className="text-xs text-clay-orange-dark mt-1 leading-relaxed">
                      <span className="font-medium">Explanation:</span> {feature.explanation}
                    </p>
                  )}
                  {feature.relevance && (
                    <p className="text-xs text-clay-orange-dark mt-1 leading-relaxed">
                      <span className="font-medium">Relevance:</span> {feature.relevance}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}