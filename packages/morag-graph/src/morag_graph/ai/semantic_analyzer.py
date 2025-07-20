"""Semantic analyzer for deep relation understanding and enhancement."""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.ai import Relation
from ..models import Entity as GraphEntity

logger = structlog.get_logger(__name__)


class SemanticPattern(str, Enum):
    """Semantic patterns for relation analysis."""
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    CONDITIONAL = "conditional"
    COMPARATIVE = "comparative"
    MODAL = "modal"
    NEGATION = "negation"
    INTENSIFICATION = "intensification"


@dataclass
class SemanticSignal:
    """A semantic signal found in text."""
    pattern: SemanticPattern
    text: str
    position: int
    strength: float
    context: str


@dataclass
class RelationEnhancement:
    """Enhancement information for a relation."""
    original_type: str
    enhanced_type: str
    confidence_adjustment: float
    semantic_signals: List[SemanticSignal]
    reasoning: str


class SemanticRelationAnalyzer:
    """Analyzes text for semantic patterns that indicate deeper relationships."""

    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self._build_pattern_library()

    def _build_pattern_library(self):
        """Build library of semantic patterns for relation analysis."""
        
        # Causal patterns
        self.causal_patterns = {
            'strong_cause': [
                r'\b(?:causes?|results? in|leads? to|triggers?|produces?)\b',
                r'\b(?:due to|because of|as a result of|owing to)\b',
                r'\b(?:consequently|therefore|thus|hence)\b'
            ],
            'weak_cause': [
                r'\b(?:contributes? to|influences?|affects?)\b',
                r'\b(?:may cause|might lead to|can result in)\b',
                r'\b(?:associated with|linked to|related to)\b'
            ],
            'prevention': [
                r'\b(?:prevents?|stops?|blocks?|inhibits?|reduces?)\b',
                r'\b(?:protects? against|guards? against)\b'
            ],
            'enablement': [
                r'\b(?:enables?|allows?|facilitates?|supports?)\b',
                r'\b(?:makes? possible|helps? to)\b'
            ]
        }

        # Temporal patterns
        self.temporal_patterns = {
            'sequence': [
                r'\b(?:before|after|then|next|subsequently|following)\b',
                r'\b(?:first|second|third|finally|lastly)\b',
                r'\b(?:earlier|later|previously|afterwards)\b'
            ],
            'simultaneity': [
                r'\b(?:while|during|simultaneously|at the same time)\b',
                r'\b(?:concurrent|parallel|together)\b'
            ],
            'initiation': [
                r'\b(?:starts?|begins?|initiates?|launches?)\b',
                r'\b(?:kicks off|sets in motion)\b'
            ],
            'termination': [
                r'\b(?:ends?|finishes?|terminates?|concludes?)\b',
                r'\b(?:stops?|ceases?|discontinues?)\b'
            ]
        }

        # Hierarchical patterns
        self.hierarchical_patterns = {
            'ownership': [
                r'\b(?:owns?|possesses?|belongs? to|property of)\b',
                r'\b(?:has|contains?|includes?)\b'
            ],
            'management': [
                r'\b(?:manages?|supervises?|oversees?|controls?)\b',
                r'\b(?:leads?|heads?|directs?|runs?)\b'
            ],
            'membership': [
                r'\b(?:member of|part of|component of|element of)\b',
                r'\b(?:within|inside|under)\b'
            ]
        }

        # Functional patterns
        self.functional_patterns = {
            'operation': [
                r'\b(?:operates?|functions?|works?|runs?)\b',
                r'\b(?:performs?|executes?|carries? out)\b'
            ],
            'transformation': [
                r'\b(?:transforms?|converts?|changes?|modifies?)\b',
                r'\b(?:turns? into|becomes?|evolves? into)\b'
            ],
            'dependency': [
                r'\b(?:depends? on|relies? on|requires?|needs?)\b',
                r'\b(?:based on|built on|founded on)\b'
            ]
        }

        # Comparative patterns
        self.comparative_patterns = {
            'similarity': [
                r'\b(?:similar to|like|resembles?|comparable to)\b',
                r'\b(?:same as|identical to|equivalent to)\b'
            ],
            'difference': [
                r'\b(?:different from|unlike|distinct from|opposite of)\b',
                r'\b(?:contrasts? with|differs? from)\b'
            ],
            'superiority': [
                r'\b(?:better than|superior to|exceeds?|outperforms?)\b',
                r'\b(?:more than|greater than|higher than)\b'
            ]
        }

        # Modal patterns (indicating certainty/uncertainty)
        self.modal_patterns = {
            'certainty': [
                r'\b(?:definitely|certainly|clearly|obviously)\b',
                r'\b(?:always|never|must|will)\b'
            ],
            'uncertainty': [
                r'\b(?:possibly|probably|likely|perhaps|maybe)\b',
                r'\b(?:might|could|may|seems? to)\b'
            ]
        }

        # Negation patterns
        self.negation_patterns = [
            r'\b(?:not|no|never|neither|nor)\b',
            r'\b(?:doesn\'t|don\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\b(?:cannot|can\'t|won\'t|wouldn\'t|shouldn\'t)\b'
        ]

    def analyze_relation_context(
        self,
        relation: Relation,
        full_text: str,
        entities: Optional[List[GraphEntity]] = None
    ) -> RelationEnhancement:
        """Analyze the context around a relation to enhance its semantic understanding."""
        
        # Find the context window around the relation
        context_window = self._extract_relation_context(
            relation, full_text, window_size=200
        )
        
        # Detect semantic signals
        semantic_signals = self._detect_semantic_signals(context_window)
        
        # Enhance relation type based on signals
        enhanced_type, confidence_adjustment, reasoning = self._enhance_relation_type(
            relation.relation_type, semantic_signals, context_window
        )
        
        return RelationEnhancement(
            original_type=relation.relation_type,
            enhanced_type=enhanced_type,
            confidence_adjustment=confidence_adjustment,
            semantic_signals=semantic_signals,
            reasoning=reasoning
        )

    def _extract_relation_context(
        self,
        relation: Relation,
        full_text: str,
        window_size: int = 200
    ) -> str:
        """Extract context window around the relation mention."""
        # Try to find the entities in the text
        source_pos = full_text.lower().find(relation.source_entity.lower())
        target_pos = full_text.lower().find(relation.target_entity.lower())
        
        if source_pos == -1 or target_pos == -1:
            # If entities not found, use the relation context if available
            return getattr(relation, 'context', '')
        
        # Find the span covering both entities
        start_pos = min(source_pos, target_pos)
        end_pos = max(source_pos + len(relation.source_entity), 
                     target_pos + len(relation.target_entity))
        
        # Expand window
        context_start = max(0, start_pos - window_size // 2)
        context_end = min(len(full_text), end_pos + window_size // 2)
        
        return full_text[context_start:context_end]

    def _detect_semantic_signals(self, context: str) -> List[SemanticSignal]:
        """Detect semantic signals in the context."""
        signals = []
        context_lower = context.lower()
        
        # Check for causal patterns
        for pattern_type, patterns in self.causal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    strength = self._get_pattern_strength(pattern_type, 'causal')
                    signals.append(SemanticSignal(
                        pattern=SemanticPattern.CAUSAL,
                        text=match.group(),
                        position=match.start(),
                        strength=strength,
                        context=pattern_type
                    ))
        
        # Check for temporal patterns
        for pattern_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    strength = self._get_pattern_strength(pattern_type, 'temporal')
                    signals.append(SemanticSignal(
                        pattern=SemanticPattern.TEMPORAL,
                        text=match.group(),
                        position=match.start(),
                        strength=strength,
                        context=pattern_type
                    ))
        
        # Check for hierarchical patterns
        for pattern_type, patterns in self.hierarchical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    strength = self._get_pattern_strength(pattern_type, 'hierarchical')
                    signals.append(SemanticSignal(
                        pattern=SemanticPattern.CAUSAL,  # Using CAUSAL as placeholder
                        text=match.group(),
                        position=match.start(),
                        strength=strength,
                        context=pattern_type
                    ))
        
        # Check for modal patterns
        for pattern_type, patterns in self.modal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    strength = self._get_pattern_strength(pattern_type, 'modal')
                    signals.append(SemanticSignal(
                        pattern=SemanticPattern.MODAL,
                        text=match.group(),
                        position=match.start(),
                        strength=strength,
                        context=pattern_type
                    ))
        
        # Check for negation
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, context_lower, re.IGNORECASE)
            for match in matches:
                signals.append(SemanticSignal(
                    pattern=SemanticPattern.NEGATION,
                    text=match.group(),
                    position=match.start(),
                    strength=0.8,
                    context='negation'
                ))
        
        return signals

    def _get_pattern_strength(self, pattern_type: str, category: str) -> float:
        """Get the strength score for a pattern type."""
        strength_map = {
            'causal': {
                'strong_cause': 0.9,
                'weak_cause': 0.6,
                'prevention': 0.8,
                'enablement': 0.7
            },
            'temporal': {
                'sequence': 0.8,
                'simultaneity': 0.7,
                'initiation': 0.8,
                'termination': 0.8
            },
            'hierarchical': {
                'ownership': 0.9,
                'management': 0.8,
                'membership': 0.7
            },
            'modal': {
                'certainty': 0.9,
                'uncertainty': 0.4
            }
        }
        
        return strength_map.get(category, {}).get(pattern_type, 0.5)

    def _enhance_relation_type(
        self,
        original_type: str,
        signals: List[SemanticSignal],
        context: str
    ) -> Tuple[str, float, str]:
        """Enhance relation type based on semantic signals - now fully dynamic."""

        enhanced_type = original_type
        confidence_adjustment = 0.0
        reasoning_parts = []

        # Analyze causal signals and suggest dynamic relation types
        causal_signals = [s for s in signals if s.pattern == SemanticPattern.CAUSAL]
        if causal_signals:
            strongest_causal = max(causal_signals, key=lambda s: s.strength)

            if strongest_causal.context == 'strong_cause':
                enhanced_type = self._create_causal_relation_type(strongest_causal.text, context)
                confidence_adjustment += 0.2
                reasoning_parts.append(f"Strong causal language detected: '{strongest_causal.text}'")

            elif strongest_causal.context == 'prevention':
                enhanced_type = self._create_prevention_relation_type(strongest_causal.text, context)
                confidence_adjustment += 0.15
                reasoning_parts.append(f"Prevention language detected: '{strongest_causal.text}'")

            elif strongest_causal.context == 'enablement':
                enhanced_type = self._create_enablement_relation_type(strongest_causal.text, context)
                confidence_adjustment += 0.15
                reasoning_parts.append(f"Enablement language detected: '{strongest_causal.text}'")

        # Analyze temporal signals and create dynamic temporal relations
        temporal_signals = [s for s in signals if s.pattern == SemanticPattern.TEMPORAL]
        if temporal_signals:
            strongest_temporal = max(temporal_signals, key=lambda s: s.strength)

            if strongest_temporal.context == 'sequence':
                enhanced_type = self._create_temporal_relation_type(strongest_temporal.text, context)
                confidence_adjustment += 0.1
                reasoning_parts.append(f"Temporal sequence detected: '{strongest_temporal.text}'")

            elif strongest_temporal.context == 'initiation':
                enhanced_type = self._create_initiation_relation_type(strongest_temporal.text, context)
                confidence_adjustment += 0.1
                reasoning_parts.append(f"Initiation language detected: '{strongest_temporal.text}'")

        # Check for negation and adjust relation type dynamically
        negation_signals = [s for s in signals if s.pattern == SemanticPattern.NEGATION]
        if negation_signals:
            enhanced_type = self._create_negated_relation_type(enhanced_type, context)
            reasoning_parts.append("Negation detected, adjusting relation type")

        # Check for modal uncertainty
        modal_signals = [s for s in signals if s.pattern == SemanticPattern.MODAL]
        uncertainty_signals = [s for s in modal_signals if s.context == 'uncertainty']
        if uncertainty_signals:
            confidence_adjustment -= 0.1
            reasoning_parts.append("Uncertainty language detected, reducing confidence")

        certainty_signals = [s for s in modal_signals if s.context == 'certainty']
        if certainty_signals:
            confidence_adjustment += 0.1
            reasoning_parts.append("Certainty language detected, increasing confidence")

        # If no enhancement was made, create a context-specific relation type
        if enhanced_type == original_type and original_type in ['MENTIONS', 'RELATED_TO']:
            enhanced_type = self._create_context_specific_relation(context, signals)
            if enhanced_type != original_type:
                reasoning_parts.append(f"Enhanced generic relation to context-specific type: {enhanced_type}")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No semantic enhancement applied"

        return enhanced_type, confidence_adjustment, reasoning

    def _create_causal_relation_type(self, signal_text: str, context: str) -> str:
        """Create a dynamic causal relation type based on context."""
        context_lower = context.lower()
        signal_lower = signal_text.lower()

        # Analyze the specific causal language to create precise relation type
        if 'prevent' in signal_lower or 'stop' in signal_lower:
            return 'prevents'
        elif 'cause' in signal_lower or 'result' in signal_lower:
            return 'causes'
        elif 'trigger' in signal_lower or 'initiate' in signal_lower:
            return 'triggers'
        elif 'enable' in signal_lower or 'allow' in signal_lower:
            return 'enables'
        else:
            return 'influences'

    def _create_prevention_relation_type(self, signal_text: str, context: str) -> str:
        """Create a dynamic prevention relation type."""
        return 'prevents'

    def _create_enablement_relation_type(self, signal_text: str, context: str) -> str:
        """Create a dynamic enablement relation type."""
        return 'enables'

    def _create_temporal_relation_type(self, signal_text: str, context: str) -> str:
        """Create a dynamic temporal relation type based on context."""
        signal_lower = signal_text.lower()

        if 'before' in signal_lower or 'prior' in signal_lower:
            return 'precedes'
        elif 'after' in signal_lower or 'following' in signal_lower:
            return 'follows'
        elif 'during' in signal_lower or 'while' in signal_lower:
            return 'occurs_during'
        elif 'start' in signal_lower or 'begin' in signal_lower:
            return 'initiates'
        elif 'end' in signal_lower or 'finish' in signal_lower:
            return 'terminates'
        else:
            return 'temporally_related_to'

    def _create_initiation_relation_type(self, signal_text: str, context: str) -> str:
        """Create a dynamic initiation relation type."""
        return 'initiates'

    def _create_negated_relation_type(self, original_type: str, context: str) -> str:
        """Create a negated version of the relation type."""
        if 'causes' in original_type:
            return 'prevents'
        elif 'enables' in original_type:
            return 'inhibits'
        elif 'supports' in original_type:
            return 'opposes'
        else:
            return f"not_{original_type}"

    def _create_context_specific_relation(self, context: str, signals: List[SemanticSignal]) -> str:
        """Create a context-specific relation type based on analysis."""
        context_lower = context.lower()

        # Analyze context for relationship indicators and create dynamic types
        if any(word in context_lower for word in ['manage', 'supervise', 'lead', 'head', 'direct']):
            return 'manages'
        elif any(word in context_lower for word in ['own', 'possess', 'belong', 'property']):
            return 'owns'
        elif any(word in context_lower for word in ['create', 'produce', 'generate', 'make', 'build']):
            return 'creates'
        elif any(word in context_lower for word in ['use', 'utilize', 'employ', 'apply']):
            return 'uses'
        elif any(word in context_lower for word in ['teach', 'instruct', 'educate', 'train']):
            return 'teaches'
        elif any(word in context_lower for word in ['collaborate', 'partner', 'work with', 'cooperate']):
            return 'collaborates_with'
        elif any(word in context_lower for word in ['locate', 'situate', 'base', 'position']):
            return 'located_in'
        elif any(word in context_lower for word in ['similar', 'like', 'resemble', 'comparable']):
            return 'similar_to'
        elif any(word in context_lower for word in ['different', 'unlike', 'opposite', 'contrast']):
            return 'differs_from'
        elif any(word in context_lower for word in ['treat', 'cure', 'heal', 'therapy']):
            return 'treats'
        elif any(word in context_lower for word in ['study', 'research', 'investigate', 'examine']):
            return 'studies'
        elif any(word in context_lower for word in ['develop', 'design', 'engineer', 'implement']):
            return 'develops'
        elif any(word in context_lower for word in ['connect', 'link', 'join', 'attach']):
            return 'connects_to'
        elif any(word in context_lower for word in ['transform', 'convert', 'change', 'modify']):
            return 'transforms'
        elif any(word in context_lower for word in ['control', 'regulate', 'govern', 'command']):
            return 'controls'

        # If no specific indicators found, create a general but meaningful type
        if signals:
            return 'interacts_with'

        return 'related_to'

    def get_enhancement_summary(self, enhancement: RelationEnhancement) -> str:
        """Get a human-readable summary of the relation enhancement."""
        if enhancement.original_type == enhancement.enhanced_type:
            return f"No enhancement: {enhancement.original_type}"

        confidence_change = ""
        if enhancement.confidence_adjustment > 0:
            confidence_change = f" (confidence +{enhancement.confidence_adjustment:.2f})"
        elif enhancement.confidence_adjustment < 0:
            confidence_change = f" (confidence {enhancement.confidence_adjustment:.2f})"

        return f"Enhanced: {enhancement.original_type} → {enhancement.enhanced_type}{confidence_change}"
