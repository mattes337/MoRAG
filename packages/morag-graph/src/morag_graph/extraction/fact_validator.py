"""Validation logic for extracted facts."""

import re
from typing import List, Tuple, Dict, Any
import structlog

from ..models.fact import Fact, FactType


class FactValidator:
    """Validate quality and completeness of extracted facts."""
    
    def __init__(self, min_confidence: float = 0.7):
        """Initialize validator with configuration.
        
        Args:
            min_confidence: Minimum confidence threshold for facts
        """
        self.min_confidence = min_confidence
        self.logger = structlog.get_logger(__name__)
        
        # Generic words that indicate low specificity
        self.generic_subjects = {
            'it', 'this', 'that', 'they', 'these', 'those', 'something', 
            'anything', 'everything', 'nothing', 'someone', 'anyone',
            'everyone', 'thing', 'stuff', 'item', 'element', 'aspect'
        }
        
        # Words that indicate meta-commentary rather than facts
        self.meta_indicators = {
            'document', 'text', 'paper', 'article', 'chapter', 'section',
            'paragraph', 'sentence', 'author', 'writer', 'researcher',
            'study shows', 'research indicates', 'paper discusses'
        }
    
    def validate_fact(self, fact: Fact) -> Tuple[bool, List[str]]:
        """Validate a fact and return validation result with issues.
        
        Args:
            fact: Fact to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check basic completeness
        completeness_issues = self._check_completeness(fact)
        issues.extend(completeness_issues)
        
        # Check specificity
        specificity_issues = self._check_specificity(fact)
        issues.extend(specificity_issues)
        
        # Check actionability
        actionability_issues = self._check_actionability(fact)
        issues.extend(actionability_issues)
        
        # Check confidence threshold
        if fact.extraction_confidence < self.min_confidence:
            issues.append(f"Confidence {fact.extraction_confidence:.2f} below threshold {self.min_confidence}")
        
        # Check for meta-commentary
        meta_issues = self._check_meta_commentary(fact)
        issues.extend(meta_issues)
        
        # Check fact type validity
        type_issues = self._check_fact_type(fact)
        issues.extend(type_issues)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.debug(
                "Fact validation failed",
                fact_id=fact.id,
                issues=issues,
                subject=fact.subject[:50] + "..." if len(fact.subject) > 50 else fact.subject
            )
        
        return is_valid, issues
    
    def _check_completeness(self, fact: Fact) -> List[str]:
        """Check if fact has required components.

        Args:
            fact: Fact to check

        Returns:
            List of completeness issues
        """
        issues = []

        if not fact.subject or not fact.subject.strip():
            issues.append("Missing or empty subject")

        if not fact.object or not fact.object.strip():
            issues.append("Missing or empty object")

        # Check minimum length requirements
        if fact.subject and len(fact.subject.strip()) < 3:
            issues.append("Subject too short (minimum 3 characters)")

        if fact.object and len(fact.object.strip()) < 3:
            issues.append("Object too short (minimum 3 characters)")

        return issues
    
    def _check_specificity(self, fact: Fact) -> List[str]:
        """Check if fact is specific enough to be useful.

        Args:
            fact: Fact to check

        Returns:
            List of specificity issues
        """
        issues = []

        # Check for generic entities in primary entities
        for entity in fact.structured_metadata.primary_entities:
            entity_lower = entity.lower().strip()
            entity_words = set(entity_lower.split())
            if entity_words.intersection(self.generic_subjects):
                issues.append(f"Generic entity: '{entity}'")
        
        # Check for overly vague language
        vague_patterns = [
            r'\b(some|many|several|various|different|certain)\b',
            r'\b(generally|usually|often|sometimes|typically)\b',
            r'\b(might|could|may|possibly|potentially)\b'
        ]
        
        combined_text = f"{fact.subject} {fact.object}"
        if fact.approach:
            combined_text += f" {fact.approach}"
        if fact.solution:
            combined_text += f" {fact.solution}"
        
        for pattern in vague_patterns:
            if re.search(pattern, combined_text.lower()):
                issues.append(f"Contains vague language: {pattern}")
                break  # Only report one vague language issue
        
        # Check for sufficient detail
        total_length = len(fact.subject) + len(fact.object)
        if fact.approach:
            total_length += len(fact.approach)
        if fact.solution:
            total_length += len(fact.solution)
        
        if total_length < 20:
            issues.append("Fact too brief - lacks sufficient detail")
        
        return issues
    
    def _check_actionability(self, fact: Fact) -> List[str]:
        """Check if fact provides actionable information.
        
        Args:
            fact: Fact to check
            
        Returns:
            List of actionability issues
        """
        issues = []
        
        # Check for actionable verbs or concrete information
        actionable_indicators = [
            # Process verbs
            'implement', 'configure', 'install', 'setup', 'create', 'build',
            'develop', 'design', 'analyze', 'measure', 'test', 'validate',
            'optimize', 'improve', 'reduce', 'increase', 'enhance', 'use',
            'apply', 'employ', 'utilize', 'perform', 'execute', 'achieve',
            # Result verbs
            'achieves', 'produces', 'results', 'leads', 'causes', 'enables',
            'provides', 'delivers', 'generates', 'yields', 'accuracy',
            'performance', 'efficiency', 'effectiveness', 'success',
            # Specific nouns
            'method', 'technique', 'approach', 'procedure', 'process',
            'algorithm', 'formula', 'equation', 'model', 'framework',
            'system', 'tool', 'software', 'hardware', 'device', 'network',
            'learning', 'training', 'classification', 'recognition', 'detection',
            # Medical/Health terms
            'treatment', 'therapy', 'medication', 'dosage', 'dose', 'intake',
            'administration', 'prescription', 'supplement', 'extract', 'compound',
            'concentration', 'mg', 'gram', 'daily', 'twice', 'morning', 'evening',
            'before', 'after', 'meals', 'empty stomach', 'with food',
            # Herbal/Natural terms
            'herb', 'plant', 'botanical', 'natural', 'organic', 'standardized',
            'tincture', 'capsule', 'tablet', 'tea', 'infusion', 'decoction',
            'preparation', 'formulation', 'blend', 'mixture', 'combination',
            # Action/Effect terms
            'improves', 'supports', 'helps', 'assists', 'promotes', 'enhances',
            'reduces', 'decreases', 'increases', 'boosts', 'strengthens',
            'calms', 'soothes', 'relieves', 'alleviates', 'prevents',
            # Measurement terms
            'study', 'research', 'clinical', 'trial', 'evidence', 'shown',
            'demonstrated', 'proven', 'effective', 'beneficial', 'safe',
            # German medical terms (for multilingual support)
            'behandlung', 'therapie', 'medikament', 'dosierung', 'einnahme',
            'anwendung', 'extrakt', 'standardisiert', 'täglich', 'morgens',
            'abends', 'vor', 'nach', 'mahlzeiten', 'verbessert', 'unterstützt',
            'hilft', 'reduziert', 'erhöht', 'stärkt', 'beruhigt', 'lindert'
        ]
        
        # Use fact_text for checking
        fact_text_lower = fact.fact_text.lower()

        # Check for actionable content - be more lenient
        has_actionable_content = any(
            indicator in fact_text_lower for indicator in actionable_indicators
        )

        # Accept fact if it has actionable indicators or is substantial
        if not has_actionable_content:
            # Only fail if the fact is very generic and brief
            if len(fact.fact_text.strip()) < 30:
                issues.append("Fact too brief and lacks actionable content")

        return issues
    
    def _check_meta_commentary(self, fact: Fact) -> List[str]:
        """Check for meta-commentary about the document itself.
        
        Args:
            fact: Fact to check
            
        Returns:
            List of meta-commentary issues
        """
        issues = []
        
        combined_text = f"{fact.subject} {fact.object}"
        if fact.approach:
            combined_text += f" {fact.approach}"
        if fact.solution:
            combined_text += f" {fact.solution}"
        
        combined_lower = combined_text.lower()
        
        for meta_indicator in self.meta_indicators:
            if meta_indicator in combined_lower:
                issues.append(f"Contains meta-commentary: '{meta_indicator}'")
                break  # Only report one meta-commentary issue
        
        return issues
    
    def _check_fact_type(self, fact: Fact) -> List[str]:
        """Check if fact type is valid.
        
        Args:
            fact: Fact to check
            
        Returns:
            List of fact type issues
        """
        issues = []
        
        valid_types = FactType.all_types()
        if fact.fact_type not in valid_types:
            issues.append(f"Invalid fact type: '{fact.fact_type}'. Valid types: {valid_types}")
        
        return issues
    
    def validate_facts_batch(self, facts: List[Fact]) -> Dict[str, Any]:
        """Validate a batch of facts and return summary statistics.
        
        Args:
            facts: List of facts to validate
            
        Returns:
            Dictionary with validation statistics
        """
        valid_facts = []
        invalid_facts = []
        all_issues = []
        
        for fact in facts:
            is_valid, issues = self.validate_fact(fact)
            if is_valid:
                valid_facts.append(fact)
            else:
                invalid_facts.append((fact, issues))
                all_issues.extend(issues)
        
        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            'total_facts': len(facts),
            'valid_facts': len(valid_facts),
            'invalid_facts': len(invalid_facts),
            'validation_rate': len(valid_facts) / len(facts) if facts else 0,
            'valid_fact_objects': valid_facts,
            'invalid_fact_objects': invalid_facts,
            'issue_counts': issue_counts,
            'most_common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def get_quality_score(self, fact: Fact) -> float:
        """Calculate a quality score for a fact.
        
        Args:
            fact: Fact to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        is_valid, issues = self.validate_fact(fact)
        
        if not is_valid:
            # Penalize based on number and severity of issues
            penalty = min(0.8, len(issues) * 0.2)
            return max(0.0, fact.extraction_confidence - penalty)
        
        # Bonus for completeness
        completeness_bonus = 0.0
        if fact.approach and fact.solution:
            completeness_bonus += 0.1
        if fact.remarks:
            completeness_bonus += 0.05
        if fact.keywords:
            completeness_bonus += 0.05
        
        return min(1.0, fact.extraction_confidence + completeness_bonus)
