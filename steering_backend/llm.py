# steering agent (Claude)
'''
Responsible for autosteering the SAE given a query + other things
'''

import os
import logging
import json
from typing import List, Dict, Any, Optional
import anthropic
import dotenv

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompts for Claude API
QUERY_REWRITE_PROMPT_TEMPLATE = """
Your task is to improve the search effectiveness of the given query by rewriting it to produce better results in an arXiv papers search system.

Original query: "{query}"

As an expert in information retrieval, please rewrite this query to:
1. Add synonyms or related technical terms that would enhance recall
2. Include specific technical terminology from relevant domains
3. Disambiguate any potentially ambiguous terms
4. Expand abbreviations where helpful
5. Remove filler words and focus on key concepts
6. Format for vector search effectiveness (focus on content-bearing terms)

Your rewritten query should:
- Be clear and concise (generally within 2-3 times the length of the original)
- Maintain the original intent but enhance the technical precision
- Include specialized academic terminology where appropriate
- Be optimized for searching academic papers on arXiv

Return ONLY the rewritten query text with no explanations, preamble, or quotes.
"""

QUERY_INTENT_PROMPT_TEMPLATE = """
Analyze this search query: "{query}"

Extract the following aspects of the query intent:
1. Key concepts and topics
2. Desired level of technical depth
3. Perspective or approach being sought
4. Any specific domains, fields, or disciplines relevant to the query
5. Any implied preferences about the type of content (e.g., theoretical, practical, tutorial, etc.)

Format your response as a single JSON object like this:
{{
    "key_concepts": ["concept1", "concept2", ...],
    "technical_level": "low|medium|high",
    "perspective": "theoretical|practical|critical|neutral|etc",
    "domains": ["domain1", "domain2", ...],
    "content_type": "theoretical|practical|tutorial|overview|etc"
}}

Provide ONLY the JSON with no other text.
"""

FEATURE_MATCHING_PROMPT_TEMPLATE = """
I need you to match semantic search features to a query intent.

QUERY INTENT:
{intent_json}

AVAILABLE FEATURES:
{features_json}

Select the most relevant features that would help steer the search toward more relevant results for this query.
For each selected feature, determine an appropriate steering strength between -10 and 10:
- Positive values (1 to 10) enhance the feature's presence
- Negative values (-1 to -10) reduce the feature's presence
- The magnitude (1-10) indicates how strongly to apply the steering

Consider these factors:
1. How well the feature's description matches key concepts in the query
2. Whether the feature matches the desired technical level and perspective
3. Whether enhancing or diminishing the feature would better serve the query intent
4. The feature's activation value (higher activation means the feature is more present in the query)

Return your selections as a JSON array like this:
[
    {{
        "feature_id": 123,
        "strength": 8.5,
        "explanation": "The feature description",
        "relevance": "Why this feature was selected and why this strength was chosen"
    }},
    ...
]

Select up to 5 features that would most effectively steer the search.
Provide ONLY the JSON with no other text.
"""

CONTENT_FILTERING_PROMPT_TEMPLATE = """
I need you to evaluate if the following text chunk from an arXiv paper is primarily citation text, references, or low-quality content that should be filtered out from search results.

TEXT CHUNK:
---
{text_chunk}
---

Evaluate if this chunk should be filtered out based on these criteria:
1. High density of citation patterns like [1], (2020), et al.
2. Primarily reference or bibliography section content
3. Mostly numbered references (lines starting with numbers followed by dots or parentheses)
4. Mostly whitespace or very short content with low information value
5. Boilerplate text with little scientific value

Return ONLY a JSON with a boolean "should_filter" and a brief "reason":
{{"should_filter": true/false, "reason": "Brief explanation of why this should or should not be filtered"}}
"""

class ClaudeAPIClient:
    """Client for interacting with Claude API"""
    
    def __init__(self):
        """Initialize the Claude API client"""
        try:
            self.client = anthropic.Anthropic()
            # The client automatically uses the ANTHROPIC_API_KEY environment variable
        except anthropic.AnthropicError as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None # Indicate client initialization failed

        # Cache for query analyses to avoid repeated API calls
        self.query_cache = {}
        
    def call_claude_api(self, prompt: str) -> Optional[str]:
        """
        Call the Claude API with the given prompt using the anthropic client.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            The model's response text or None if the call fails
        """
        if not self.client: # Check if client was initialized
            logger.error("Cannot call Claude API, client not initialized.")
            return None
            
        try:
            # Replace the old requests logic:
            # url = "https://api.anthropic.com/v1/messages"
            # headers = { ... }
            # data = { ... }
            # response = requests.post(url, headers=headers, json=data)
            # response.raise_for_status()
            # response_data = response.json()
            # content = response_data.get("content", [])
            # if content and isinstance(content, list) and len(content) > 0:
            #     text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]
            #     return "".join(text_blocks)
            # return None

            # With the new anthropic client logic:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022", # Use the latest Sonnet model
                max_tokens=1024, # Keep max_tokens, or adjust as needed
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract text from the response object
            if message.content and isinstance(message.content, list) and len(message.content) > 0:
                 # Assuming the first content block is the text response
                return message.content[0].text
            else:
                logger.warning("Claude API returned empty or unexpected content format.")
                return None
            
        except anthropic.APIError as e: # Catch specific anthropic errors
            logger.error(f"Anthropic API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Claude API call: {e}")
            return None
            
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Use Claude API to analyze the query intent.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary containing query intent analysis
        """
        # Check cache first
        if query in self.query_cache:
            logger.info(f"Using cached query intent analysis for: '{query}'")
            return self.query_cache[query]
        
        try:
            logger.info(f"Analyzing query intent with Claude: '{query}'")
            
            # Format the prompt
            prompt = QUERY_INTENT_PROMPT_TEMPLATE.format(query=query)
            
            # Call Claude API
            response = self.call_claude_api(prompt)
            
            if not response:
                logger.error("Failed to get response from Claude API")
                return {}
            
            # Parse the JSON response
            try:
                intent_data = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in query intent response: {response}")
                return {}
            
            # Cache the result
            self.query_cache[query] = intent_data
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Error analyzing query intent for '{query}': {e}")
            return {}
            
    def match_features_to_intent(
        self, 
        features: List[Dict[str, Any]], 
        query_intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Match features to query intent using Claude API.
        
        Args:
            features: List of feature dictionaries
            query_intent: Dictionary with query intent analysis
            
        Returns:
            List of selected features with steering strengths
        """
        if not features or not query_intent:
            logger.warning("Empty features or query intent for matching")
            return []
            
        try:
            # Format the data for the prompt
            intent_json = json.dumps(query_intent, indent=2)
            features_json = json.dumps(features, indent=2)
            
            # Format the prompt
            prompt = FEATURE_MATCHING_PROMPT_TEMPLATE.format(
                intent_json=intent_json,
                features_json=features_json
            )
            
            # Call Claude API
            response = self.call_claude_api(prompt)
            
            if not response:
                logger.error("Failed to get feature matching response from Claude API")
                return []
                
            # Parse the JSON response
            try:
                selected_features = json.loads(response)
                
                if not isinstance(selected_features, list):
                    logger.error(f"Invalid feature matching response format: {response}")
                    return []
                    
                # Process and normalize the selected features
                for feature in selected_features:
                    if "feature_id" in feature:
                        feature["feature_id"] = int(feature["feature_id"])
                    if "strength" in feature:
                        feature["strength"] = float(feature["strength"])
                        
                return selected_features
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in feature matching response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error in feature matching: {e}")
            return []
            
    def rewrite_query(self, query: str) -> str:
        """
        Use Claude API to rewrite the query for better search effectiveness.
        
        Args:
            query: The original search query text
            
        Returns:
            A rewritten query that should produce better search results, or the original query if rewriting fails
        """
        # Use a cache key specific to query rewriting
        cache_key = f"rewrite_{query}"
        if cache_key in self.query_cache:
            logger.info(f"Using cached query rewrite for: '{query}'")
            return self.query_cache[cache_key]
        
        try:
            logger.info(f"Rewriting query with Claude: '{query}'")
            
            # Format the prompt
            prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(query=query)
            
            # Call Claude API
            response = self.call_claude_api(prompt)
            
            if not response or not response.strip():
                logger.error("Failed to get query rewrite from Claude API")
                return query  # Return original query if rewriting fails
            
            # Clean up the response
            rewritten_query = response.strip()
            
            # Cache the result
            self.query_cache[cache_key] = rewritten_query
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query '{query}': {e}")
            return query  # Return original query if an error occurs
    
    def filter_content(self, text_chunk: str) -> Dict[str, Any]:
        """
        Use Claude to determine if content should be filtered out.
        
        Args:
            text_chunk: The text content to evaluate
            
        Returns:
            Dictionary with 'should_filter' boolean and 'reason' string
        """
        if not self.client:
            logger.error("Cannot filter content, client not initialized.")
            return {"should_filter": False, "reason": "Unable to connect to Claude API"}
        
        # Create a simple cache key based on the hash of the text
        cache_key = f"filter_{hash(text_chunk[:100])}"
        if cache_key in self.query_cache:
            logger.info("Using cached content filtering decision")
            return self.query_cache[cache_key]
        
        # Truncate very long text chunks to keep prompt within limits
        MAX_CHUNK_LENGTH = 4000
        text_to_analyze = text_chunk
        if len(text_chunk) > MAX_CHUNK_LENGTH:
            text_to_analyze = text_chunk[:MAX_CHUNK_LENGTH] + "... [truncated]"
        
        try:
            logger.info("Filtering content with Claude")
            
            # Format the prompt
            prompt = CONTENT_FILTERING_PROMPT_TEMPLATE.format(
                text_chunk=text_to_analyze
            )
            
            # Call Claude API
            response = self.call_claude_api(prompt)
            
            if not response or not response.strip():
                logger.error("Failed to get content filtering response from Claude API")
                return {"should_filter": False, "reason": "No response from filtering API"}
            
            # Parse the JSON response
            try:
                filter_decision = json.loads(response.strip())
                
                if not isinstance(filter_decision, dict) or 'should_filter' not in filter_decision:
                    logger.error(f"Invalid content filtering response format: {response}")
                    return {"should_filter": False, "reason": "Invalid filtering response format"}
                
                # Cache the result
                self.query_cache[cache_key] = filter_decision
                
                return filter_decision
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in content filtering response: {e}, response: {response}")
                return {"should_filter": False, "reason": "Invalid JSON response from filtering API"}
            
        except Exception as e:
            logger.error(f"Error in content filtering: {e}")
            return {"should_filter": False, "reason": f"Error during filtering: {str(e)}"}
