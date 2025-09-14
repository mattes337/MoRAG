"""Query intent analysis for graph-guided retrieval."""

import re
import logging
from typing import Dict, List

from .models import QueryEntity

logger = logging.getLogger(__name__)


class QueryIntentAnalyzer:
    """Analyze query intent with pattern matching and confidence scoring."""
    
    def __init__(self):
        """Initialize the intent analyzer with predefined patterns."""
        self.intent_patterns = {
            'factual': [
                r'\bwhat is\b', r'\bwho is\b', r'\bwhere is\b', r'\bwhen did\b',
                r'\bdefine\b', r'\btell me about\b', r'\bwhat are\b', r'\bwho are\b'
            ],
            'comparative': [
                r'\bcompare\b', r'\bdifference between\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter than\b', r'\bsimilar to\b', r'\bcontrast\b'
            ],
            'procedural': [
                r'\bhow to\b', r'\bsteps to\b', r'\bprocess of\b', r'\bway to\b',
                r'\bmethod for\b', r'\bprocedure\b'
            ],
            'causal': [
                r'\bwhy does\b', r'\bcause of\b', r'\breason for\b', r'\bdue to\b',
                r'\bwhy is\b', r'\bwhat causes\b', r'\bresult of\b'
            ],
            'exploratory': [
                r'\bfind\b', r'\bsearch\b', r'\bshow\b', r'\blist\b',
                r'\bdiscover\b', r'\bexplore\b', r'\bbrowse\b'
            ],
            'explanatory': [
                r'\bexplain\b', r'\bdescribe\b', r'\bdetail\b', r'\belaborate\b',
                r'\bclarify\b', r'\bbreak down\b'
            ]
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_intent(self, query: str, entities: List[QueryEntity]) -> Dict[str, float]:
        """Analyze query intent with confidence scores.
        
        Args:
            query: User query text
            entities: Extracted entities from the query
            
        Returns:
            Dictionary mapping intent types to confidence scores
        """
        intent_scores = {}
        query_lower = query.lower()
        
        self.logger.debug(f"Analyzing intent for query: {query}")
        
        # Pattern-based scoring
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score = max(score, 0.8)
                    matched_patterns.append(pattern)
            
            # Adjust based on entity types and count
            if entities:
                entity_types = [e.entity_type for e in entities]
                entity_count = len(entities)
                
                # Intent-specific adjustments
                if intent == 'factual' and 'PERSON' in entity_types:
                    score += 0.1
                elif intent == 'comparative' and entity_count >= 2:
                    score += 0.2
                elif intent == 'exploratory' and entity_count == 1:
                    score += 0.1
                elif intent == 'causal' and any(t in entity_types for t in ['EVENT', 'CONCEPT']):
                    score += 0.1
            
            # Question word adjustments
            if intent == 'factual' and any(word in query_lower for word in ['what', 'who', 'where', 'when']):
                score += 0.1
            elif intent == 'explanatory' and any(word in query_lower for word in ['how', 'why']):
                score += 0.1
            
            intent_scores[intent] = min(score, 1.0)
            
            if matched_patterns:
                self.logger.debug(f"Intent '{intent}' matched patterns: {matched_patterns}, score: {score:.3f}")
        
        # Ensure at least one intent has a reasonable score
        if all(score < 0.3 for score in intent_scores.values()):
            # Default to general intent
            intent_scores['general'] = 0.5
        
        return intent_scores
    
    def get_primary_intent(self, intent_scores: Dict[str, float]) -> str:
        """Get the primary intent with highest confidence.
        
        Args:
            intent_scores: Dictionary of intent scores
            
        Returns:
            Primary intent type
        """
        if not intent_scores:
            return 'general'
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        return primary_intent[0]
    
    def is_complex_query(self, query: str, entities: List[QueryEntity]) -> bool:
        """Determine if query is complex based on structure and entities.
        
        Args:
            query: User query text
            entities: Extracted entities
            
        Returns:
            True if query is considered complex
        """
        # Multiple entities suggest complexity
        if len(entities) > 2:
            return True
        
        # Long queries are often complex
        if len(query.split()) > 15:
            return True
        
        # Multiple question words suggest complexity
        question_words = ['what', 'who', 'where', 'when', 'why', 'how']
        question_count = sum(1 for word in question_words if word in query.lower())
        if question_count > 1:
            return True
        
        # Comparative queries are often complex
        if any(word in query.lower() for word in ['compare', 'difference', 'versus', 'vs']):
            return True
        
        return False
