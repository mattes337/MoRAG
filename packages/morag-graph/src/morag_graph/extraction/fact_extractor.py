"""Extract structured facts from document chunks."""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import structlog

from ..models.fact import Fact, FactType
from .fact_validator import FactValidator
from .fact_filter import FactFilter, DomainFilterConfig
from .fact_filter_config import create_domain_configs_for_language

# Import agents framework - required
try:
    from agents import get_agent
except ImportError:
    raise ImportError("Agents framework is required. Please install the agents package.")


class FactExtractor:
    """Extract structured facts from document chunks."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_confidence: float = 0.3,
        max_facts_per_chunk: int = 10,
        domain: str = "general",
        language: str = "en",
        max_workers: int = 5,
        allow_vague_language: bool = False,
        require_entities: bool = True,
        min_fact_length: int = 20,
        strict_validation: bool = True
    ):
        """Initialize fact extractor with LLM and configuration.

        Args:
            model_id: LLM model to use for extraction
            api_key: API key for LLM service
            min_confidence: Minimum confidence threshold for facts
            max_facts_per_chunk: Maximum facts to extract per chunk
            domain: Domain context for extraction
            language: Language of the text
            max_workers: Maximum worker threads for parallel processing
            allow_vague_language: Allow facts with vague language (mark with lower confidence)
            require_entities: Require primary entities in structured metadata
            min_fact_length: Minimum fact text length
            strict_validation: Enable strict quality validation
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_confidence = min_confidence
        self.max_facts_per_chunk = max_facts_per_chunk
        self.domain = domain
        self.language = language
        self.max_workers = max_workers

        self.logger = structlog.get_logger(__name__)
        self.validator = FactValidator(
            min_confidence=min_confidence,
            allow_vague_language=allow_vague_language,
            require_entities=require_entities,
            min_fact_length=min_fact_length,
            strict_validation=strict_validation
        )
        # Initialize fact filter with domain-specific configurations
        domain_configs = self._create_domain_filter_configs()
        self.fact_filter = FactFilter(domain_configs)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize fact extraction agent
        self.fact_agent = get_agent("fact_extraction")
        # Configure agent for this domain
        self.fact_agent.update_config(
            agent_config={
                "max_facts": max_facts_per_chunk,
                "domain": domain,
                "language": language,
                "min_confidence": min_confidence
            }
        )
    
    async def extract_facts(
        self, 
        chunk_text: str, 
        chunk_id: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Fact]:
        """Extract structured facts from a document chunk.
        
        Args:
            chunk_text: Text content to extract facts from
            chunk_id: Unique identifier for the chunk
            document_id: Identifier for the source document
            context: Optional context information (domain, metadata, etc.)
            
        Returns:
            List of validated Fact objects
        """
        if not chunk_text or not chunk_text.strip():
            return []
        
        # Update context from parameters
        base_domain = context.get('domain', self.domain) if context else self.domain

        # Infer domain from text if not explicitly provided or if it's 'general'
        if base_domain == 'general':
            inferred_domain = self._infer_domain_from_text(chunk_text)
            if inferred_domain != 'general':
                base_domain = inferred_domain
                self.logger.info(f"Inferred domain '{inferred_domain}' from text content")

        extraction_context = {
            'domain': base_domain,
            'language': context.get('language', self.language) if context else self.language,
            'chunk_id': chunk_id,
            'document_id': document_id
        }
        
        self.logger.info(
            "Starting fact extraction",
            chunk_id=chunk_id,
            document_id=document_id,
            text_length=len(chunk_text),
            domain=extraction_context['domain']
        )
        
        try:
            # Preprocess the text
            processed_text = self._preprocess_chunk(chunk_text)
            
            # Extract fact candidates using LLM
            fact_candidates = await self._extract_fact_candidates(processed_text, extraction_context)

            if not fact_candidates:
                self.logger.warning(
                    "No fact candidates extracted",
                    chunk_id=chunk_id,
                    text_length=len(chunk_text)
                )
                return []

            # Debug: Log what we got from LLM
            self.logger.debug(
                "Fact candidates extracted",
                chunk_id=chunk_id,
                candidates_count=len(fact_candidates),
                candidates_types=[type(c).__name__ for c in fact_candidates],
                first_candidate_preview=str(fact_candidates[0])[:100] if fact_candidates else "None"
            )

            # Structure facts from candidates
            facts = self._structure_facts(fact_candidates, chunk_id, document_id, extraction_context)
            
            # Validate facts
            validated_facts = []
            for fact in facts:
                is_valid, issues = self.validator.validate_fact(fact)
                if is_valid:
                    validated_facts.append(fact)
                else:
                    self.logger.debug(
                        "Fact validation failed",
                        fact_id=fact.id,
                        issues=issues
                    )
            
            # Generate keywords for validated facts
            for fact in validated_facts:
                if not fact.keywords:
                    fact.keywords = self._generate_fact_keywords(fact)

            # Filter facts for domain relevance
            domain_for_filtering = self._determine_filter_domain(extraction_context.get('domain', 'general'))
            document_context = {
                'topics': extraction_context.get('topics', []),
                'domain': extraction_context.get('domain'),
                'source_file_name': extraction_context.get('source_file_name', '')
            }

            filtered_facts = self.fact_filter.filter_facts(
                validated_facts,
                domain=domain_for_filtering,
                document_context=document_context
            )

            self.logger.info(
                "Fact extraction completed",
                chunk_id=chunk_id,
                candidates_found=len(fact_candidates),
                facts_structured=len(facts),
                facts_validated=len(validated_facts),
                facts_filtered=len(filtered_facts)
            )

            return filtered_facts
            
        except Exception as e:
            self.logger.error(
                "Fact extraction failed",
                chunk_id=chunk_id,
                error=str(e),
                error_type=type(e).__name__
            )
            # Log the full stack trace for debugging
            import traceback
            self.logger.debug(
                "Full fact extraction error traceback",
                chunk_id=chunk_id,
                traceback=traceback.format_exc()
            )
            return []
    
    def _preprocess_chunk(self, text: str) -> str:
        """Clean and prepare text for fact extraction.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned text ready for extraction
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove markdown artifacts that might confuse extraction
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code

        # Clean up common artifacts - but preserve timestamps!
        # Only remove markdown links that have URLs (contain http or www or end with common extensions)
        text = re.sub(r'\[([^\]]+)\]\((?:https?://|www\.|[^)]*\.[a-z]{2,4}[^)]*)\)', r'\1', text)  # Links with URLs
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Images

        # Preserve timestamp formats like [00:21 - 00:25] or [01:23] by not removing them
        # The original regex was too broad and removed timestamps

        return text.strip()

    def _infer_domain_from_text(self, text: str) -> str:
        """Infer domain from text content using keyword analysis.

        Args:
            text: Text to analyze for domain inference

        Returns:
            Inferred domain string
        """
        text_lower = text.lower()

        # Medical/health keywords (English and German)
        medical_keywords = [
            # English terms
            'patient', 'doctor', 'medical', 'health', 'disease', 'treatment', 'medication',
            'symptom', 'diagnosis', 'therapy', 'clinical', 'hospital', 'medicine',
            'toxin', 'detox', 'vitamin', 'mineral', 'supplement', 'enzyme', 'hormone',
            'thyroid', 'mercury', 'aluminum', 'heavy metal', 'herb', 'herbal',
            'adhd', 'attention', 'hyperactivity', 'concentration', 'focus',
            'ginkgo', 'biloba', 'ginseng', 'rhodiola', 'chlorella', 'extract',
            'dosage', 'standardized', 'bioavailability', 'cognitive', 'neurotransmitter',

            # German terms
            'patient', 'arzt', 'medizinisch', 'gesundheit', 'krankheit', 'behandlung', 'medikament',
            'symptom', 'diagnose', 'therapie', 'klinisch', 'krankenhaus', 'medizin',
            'toxin', 'entgiftung', 'vitamin', 'mineral', 'nahrungsergänzung', 'enzym', 'hormon',
            'schilddrüse', 'quecksilber', 'aluminium', 'schwermetall', 'kraut', 'kräuter', 'pflanzlich',
            'adhs', 'aufmerksamkeit', 'hyperaktivität', 'konzentration', 'fokus',
            'ginkgo', 'biloba', 'ginseng', 'rhodiola', 'chlorella', 'extrakt',
            'dosierung', 'standardisiert', 'bioverfügbarkeit', 'kognitiv', 'neurotransmitter',
            'aufmerksamkeitsdefizit', 'hyperaktivitätsstörung', 'durchblutung', 'gehirn',
            'flavonglykoside', 'terpenlactone', 'ginsenoside', 'adaptogen', 'beruhigend',
            'nebenwirkungen', 'wechselwirkungen', 'kontraindikation', 'sicherheit'
        ]

        # Technical keywords
        technical_keywords = [
            'system', 'software', 'database', 'server', 'network', 'algorithm',
            'programming', 'code', 'api', 'framework', 'technology', 'computer'
        ]

        # Legal keywords
        legal_keywords = [
            'law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'contract',
            'agreement', 'regulation', 'compliance', 'statute', 'jurisdiction'
        ]

        # Business keywords
        business_keywords = [
            'business', 'company', 'corporation', 'market', 'sales', 'revenue',
            'profit', 'customer', 'client', 'strategy', 'management', 'finance'
        ]

        # Count keyword matches
        medical_score = sum(1 for keyword in medical_keywords if keyword in text_lower)
        technical_score = sum(1 for keyword in technical_keywords if keyword in text_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in text_lower)
        business_score = sum(1 for keyword in business_keywords if keyword in text_lower)

        # Determine domain based on highest score
        scores = {
            'medical': medical_score,
            'technical': technical_score,
            'legal': legal_score,
            'business': business_score
        }

        max_score = max(scores.values())
        if max_score >= 3:  # Minimum threshold for domain inference
            return max(scores, key=scores.get)

        return 'general'  # Default to general if no clear domain

    async def _extract_fact_candidates(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to extract fact candidates from text.
        
        Args:
            text: Preprocessed text
            context: Extraction context
            
        Returns:
            List of fact candidate dictionaries
        """
        domain = context.get('domain', 'general')
        language = context.get('language', 'en')
        
        # Extract query context for agent
        query_context = context.get('query_context')
        
        try:
            # Use the agents framework - ALWAYS
            result = await self.fact_agent.extract_facts(
                text=text,
                domain=domain,
                query_context=query_context,
                max_facts=self.max_facts_per_chunk
            )

            # Convert agent result to legacy format
            fact_candidates = []
            for fact in result.facts:
                fact_candidates.append({
                    'subject': fact.subject,
                    'object': fact.object,
                    'approach': fact.approach,
                    'solution': fact.solution,
                    'condition': fact.condition,
                    'remarks': fact.remarks,
                    'fact_type': fact.fact_type,
                    'confidence': fact.confidence,
                    'keywords': fact.keywords,
                    'source_text': fact.source_text
                })

            self.logger.debug(
                "Agent fact extraction completed",
                candidates_found=len(fact_candidates),
                domain=domain,
                total_facts_from_agent=result.total_facts
            )

            return fact_candidates

        except Exception as e:
            self.logger.error(
                "Agent fact extraction failed",
                error=str(e),
                domain=domain
            )
            return []
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract fact candidates.

        Args:
            response: Raw LLM response

        Returns:
            List of parsed fact dictionaries
        """
        if not response or not response.strip():
            self.logger.warning("Empty LLM response received")
            return []

        # Clean the response
        response = response.strip()

        # Log the raw response for debugging
        self.logger.debug(
            "Parsing LLM response",
            response_length=len(response),
            response_preview=response[:300]
        )

        try:
            # Use the centralized JSON parser from agents framework
            from agents.base.response_parser import LLMResponseParser

            parsed_data = LLMResponseParser.parse_json_response(
                response=response,
                fallback_value={"facts": []},
                context="legacy_fact_extraction"
            )

            # Handle both old format (direct array) and new format (object with facts array)
            if isinstance(parsed_data, list):
                # Old format: direct array of facts
                valid_candidates = []
                for i, candidate in enumerate(parsed_data):
                    if isinstance(candidate, dict):
                        valid_candidates.append(candidate)
                    else:
                        self.logger.warning(f"Candidate {i} is not a dictionary: {type(candidate)}")
                return valid_candidates
            elif isinstance(parsed_data, dict):
                # New format: object with facts array
                if "facts" in parsed_data and isinstance(parsed_data["facts"], list):
                    valid_candidates = []
                    for i, candidate in enumerate(parsed_data["facts"]):
                        if isinstance(candidate, dict):
                            valid_candidates.append(candidate)
                        else:
                            self.logger.warning(f"Fact {i} is not a dictionary: {type(candidate)}")
                    return valid_candidates
                else:
                    # Single fact object
                    return [parsed_data]

            self.logger.warning("Unexpected response format", parsed_data_type=type(parsed_data))
            return []

        except Exception as e:
            self.logger.error(
                "Failed to parse LLM response",
                error=str(e),
                error_type=type(e).__name__,
                response_preview=response[:500],
                response_length=len(response)
            )
            return []
    
    def _structure_facts(
        self,
        candidates: List[Dict[str, Any]],
        chunk_id: str,
        document_id: str,
        context: Dict[str, Any]
    ) -> List[Fact]:
        """Convert LLM output to structured Fact objects using hybrid approach.

        Args:
            candidates: List of fact candidate dictionaries
            chunk_id: Source chunk identifier
            document_id: Source document identifier
            context: Extraction context

        Returns:
            List of structured Fact objects
        """
        from ..models.fact import StructuredMetadata

        facts = []

        for i, candidate in enumerate(candidates):
            try:
                # Ensure candidate is a dictionary
                if not isinstance(candidate, dict):
                    self.logger.warning(
                        f"Candidate {i} is not a dictionary",
                        candidate_type=type(candidate).__name__,
                        candidate_value=str(candidate)[:100]
                    )
                    continue

                # Extract or generate fact_text field
                fact_text = candidate.get('fact_text', '')

                # If fact_text is missing, generate it from structured fields
                if not fact_text or not fact_text.strip():
                    fact_text = self._generate_fact_text_from_candidate(candidate)

                if not fact_text or not fact_text.strip():
                    self.logger.debug(
                        f"Candidate {i} missing fact_text and could not generate from fields",
                        candidate=candidate
                    )
                    continue

                # Extract structured metadata
                structured_metadata_data = candidate.get('structured_metadata', {})
                if not isinstance(structured_metadata_data, dict):
                    structured_metadata_data = {}

                structured_metadata = StructuredMetadata(**structured_metadata_data)

                # Helper function to safely handle optional string fields
                def safe_strip(value):
                    if value is None:
                        return None
                    elif isinstance(value, str):
                        stripped = value.strip()
                        return stripped if stripped else None
                    else:
                        stripped = str(value).strip()
                        return stripped if stripped else None

                # Create fact with hybrid approach
                fact_data = {
                    'fact_text': fact_text.strip(),
                    'structured_metadata': structured_metadata,
                    'source_chunk_id': chunk_id,
                    'source_document_id': document_id,
                    'extraction_confidence': float(candidate.get('confidence', 0.8)),
                    'fact_type': candidate.get('fact_type', FactType.DEFINITION),
                    'domain': context.get('domain'),
                    'language': context.get('language', 'en'),
                    'keywords': candidate.get('keywords', []),
                    # Enhanced extraction metadata
                    'query_relevance': candidate.get('query_relevance'),
                    'evidence_strength': candidate.get('evidence_strength'),
                    'source_span': candidate.get('source_span'),
                    # Detailed source metadata from context
                    'source_file_path': context.get('source_file_path'),
                    'source_file_name': context.get('source_file_name'),
                    'page_number': context.get('page_number'),
                    'chapter_title': context.get('chapter_title'),
                    'chapter_index': context.get('chapter_index'),
                    'paragraph_index': context.get('paragraph_index'),
                    'timestamp_start': context.get('timestamp_start'),
                    'timestamp_end': context.get('timestamp_end'),
                    'topic_header': context.get('topic_header'),
                    'speaker_label': context.get('speaker_label'),
                    'source_text_excerpt': context.get('source_text_excerpt')
                }

                # Validate fact type
                if fact_data['fact_type'] not in FactType.all_types():
                    fact_data['fact_type'] = FactType.DEFINITION

                fact = Fact(**fact_data)
                facts.append(fact)

            except Exception as e:
                self.logger.warning(
                    "Failed to structure fact candidate",
                    candidate=candidate,
                    error=str(e)
                )
                continue

        return facts

    def _generate_fact_text_from_candidate(self, candidate: Dict[str, Any]) -> str:
        """Generate fact_text from structured candidate fields.

        Args:
            candidate: Fact candidate dictionary with structured fields

        Returns:
            Generated fact text string
        """
        # Try to use source_text if available
        if candidate.get('source_text'):
            return candidate['source_text'].strip()

        # Generate from structured fields
        subject = candidate.get('subject', '')
        object_val = candidate.get('object', '')
        approach = candidate.get('approach', '')
        solution = candidate.get('solution', '')
        remarks = candidate.get('remarks', '')

        # Build fact text from available components
        fact_parts = []

        if subject and object_val:
            if approach:
                fact_parts.append(f"{subject} {approach} {object_val}")
            else:
                fact_parts.append(f"{subject} involves {object_val}")
        elif subject:
            fact_parts.append(subject)

        if solution and solution != object_val:
            fact_parts.append(f"This results in {solution}")

        if remarks:
            fact_parts.append(f"({remarks})")

        # Join parts with appropriate punctuation
        if fact_parts:
            fact_text = '. '.join(part.strip() for part in fact_parts if part.strip())
            # Ensure proper sentence ending
            if fact_text and not fact_text.endswith('.'):
                fact_text += '.'
            return fact_text

        # Fallback: try to construct from any available text fields
        text_fields = ['statement', 'description', 'content', 'text']
        for field in text_fields:
            if candidate.get(field):
                return candidate[field].strip()

        return ""

    def _create_domain_filter_configs(self) -> Dict[str, DomainFilterConfig]:
        """Create domain filter configurations based on extraction context.

        Returns:
            Dictionary of domain filter configurations
        """
        # Create standard configurations for the extractor's language
        return create_domain_configs_for_language(self.language)

    def _determine_filter_domain(self, extraction_domain: str) -> str:
        """Determine the appropriate domain for fact filtering.

        Args:
            extraction_domain: Domain used for extraction

        Returns:
            Domain key for filtering configuration
        """
        # Map extraction domains to filter domains
        domain_mapping = {
            'medical': 'medical',
            'herbal': 'adhd_herbal',
            'adhd': 'adhd_herbal',
            'research': 'medical',
            'technical': 'general',
            'general': 'general'
        }

        return domain_mapping.get(extraction_domain, 'general')

    def _generate_fact_keywords(self, fact: Fact) -> List[str]:
        """Generate keywords for fact indexing.

        Args:
            fact: Fact to generate keywords for

        Returns:
            List of keywords
        """
        # Combine fact text and metadata
        text_parts = [fact.fact_text]
        text_parts.extend(fact.structured_metadata.primary_entities)
        text_parts.extend(fact.structured_metadata.domain_concepts)
        
        combined_text = ' '.join(text_parts).lower()
        
        # Extract meaningful words (simple keyword extraction)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'men', 'oil', 'run', 'say', 'she', 'too', 'use', 'way',
            'with', 'this', 'that', 'they', 'have', 'from', 'will', 'been', 'each',
            'which', 'their', 'said', 'there', 'what', 'were', 'when', 'where'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates and limit count
        unique_keywords = list(dict.fromkeys(keywords))[:10]
        
        return unique_keywords
