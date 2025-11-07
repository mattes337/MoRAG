"""Extract keywords from document chunks and create keyword entities."""

import hashlib
import re
from typing import List, Optional, Set, Tuple

import structlog

from ..models.entity import Entity
from ..models.relation import Relation


class ChunkKeywordExtractor:
    """Extract keywords from document chunks and create keyword entities."""

    def __init__(self, domain: str = "general"):
        """Initialize the extractor.

        Args:
            domain: Domain context for keyword extraction
        """
        self.domain = domain
        self.logger = structlog.get_logger(__name__)

        # Initialize LLM agent for intelligent keyword extraction
        from morag_core.agents import AgentRegistry

        self.agent_registry = AgentRegistry()
        self.keyword_agent = self.agent_registry.get_agent("keyword_extraction")

    def extract_keywords_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        document_id: str,
        domain: Optional[str] = None,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract keywords from a document chunk and create entities/relationships.

        Args:
            chunk_text: Text content of the chunk
            chunk_id: ID of the chunk
            document_id: ID of the source document
            domain: Domain context (optional)

        Returns:
            Tuple of (keyword entities, relationships to chunk)
        """
        effective_domain = domain or self.domain

        # Extract domain-specific keywords
        keywords = self._extract_domain_keywords(chunk_text, effective_domain)

        if not keywords:
            return [], []

        entities = []
        relationships = []

        self.logger.info(f"Extracted {len(keywords)} keywords from chunk {chunk_id}")

        for keyword in keywords:
            # Create keyword entity
            keyword_entity = self._create_keyword_entity(
                keyword, chunk_id, document_id, effective_domain
            )
            entities.append(keyword_entity)

            # Create relationship: Keyword -> DESCRIBES -> Chunk
            relationship = self._create_keyword_chunk_relationship(
                keyword_entity.id, chunk_id, keyword
            )
            relationships.append(relationship)

        return entities, relationships

    def _extract_domain_keywords(self, text: str, domain: str) -> List[str]:
        """Extract domain-specific keywords from text using LLM.

        Args:
            text: Text to extract keywords from
            domain: Domain context

        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []

        try:
            # Use LLM agent for intelligent keyword extraction
            result = self.keyword_agent.process(
                {
                    "text": text,
                    "domain": domain,
                    "max_keywords": 15,
                    "language": "auto",
                    "focus": "domain_specific_terms",
                }
            )

            if result and "keywords" in result:
                return result["keywords"]
            else:
                self.logger.warning("LLM keyword extraction failed, using fallback")
                return self._fallback_keyword_extraction(text)

        except Exception as e:
            self.logger.error("Error in LLM keyword extraction", error=str(e))
            return self._fallback_keyword_extraction(text)

    def _fallback_keyword_extraction(self, text: str) -> List[str]:
        """Simple fallback keyword extraction when LLM fails.

        Args:
            text: Text to extract keywords from

        Returns:
            List of basic keywords
        """
        # Simple extraction: find words longer than 4 characters, capitalized words, etc.
        import re

        words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{4,}\b", text)

        # Basic filtering: remove very common words and keep meaningful terms
        basic_stop_words = {
            "aber",
            "auch",
            "dann",
            "dass",
            "wenn",
            "wie",
            "was",
            "wer",
            "wo",
        }
        keywords = []

        for word in words:
            word_lower = word.lower()
            if (
                word_lower not in basic_stop_words
                and not word_lower.isdigit()
                and len(word) >= 4
            ):
                keywords.append(word_lower)

        # Remove duplicates and return top 10
        return list(set(keywords))[:10]

    def _extract_medical_keywords(self, text: str) -> Set[str]:
        """Extract medical/health-specific keywords.

        Args:
            text: Lowercase text

        Returns:
            Set of medical keywords
        """
        keywords = set()

        # Medical substances and compounds
        medical_terms = [
            # Supplements and herbs (English and German)
            "ginkgo biloba",
            "ginkgo",
            "biloba",
            "chlorella",
            "spirulina",
            "ashwagandha",
            "rhodiola",
            "bacopa monnieri",
            "bacopa",
            "panax ginseng",
            "ginseng",
            "curcumin",
            "turmeric",
            "omega-3",
            "fish oil",
            "vitamin d",
            "vitamin b6",
            "vitamin b12",
            "magnesium",
            "zinc",
            "iron",
            "selenium",
            "chromium",
            "passionsblume",
            "passiflora",
            "kräuter",
            "pflanzlich",
            "extrakt",
            # Medical conditions (English and German)
            "adhd",
            "attention deficit",
            "hyperactivity",
            "autism",
            "depression",
            "anxiety",
            "insomnia",
            "fatigue",
            "cognitive decline",
            "memory loss",
            "alzheimer",
            "dementia",
            "parkinson",
            "multiple sclerosis",
            "adhs",
            "aufmerksamkeitsdefizit",
            "hyperaktivitätsstörung",
            "hyperaktivität",
            "aufmerksamkeit",
            "konzentration",
            "fokus",
            "unruhe",
            "müdigkeit",
            # Biological processes (English and German)
            "detoxification",
            "detox",
            "chelation",
            "bioavailability",
            "absorption",
            "metabolism",
            "neurotransmitter",
            "dopamine",
            "serotonin",
            "norepinephrine",
            "acetylcholine",
            "gaba",
            "glutamate",
            "inflammation",
            "oxidative stress",
            "antioxidant",
            "neuroprotective",
            "adaptogenic",
            "nootropic",
            "entgiftung",
            "bioverfügbarkeit",
            "stoffwechsel",
            "durchblutung",
            "gehirn",
            "kognitiv",
            "adaptogen",
            "beruhigend",
            # Heavy metals and toxins (English and German)
            "heavy metal",
            "mercury",
            "aluminum",
            "lead",
            "cadmium",
            "arsenic",
            "toxin",
            "pollutant",
            "pesticide",
            "herbicide",
            "schwermetall",
            "quecksilber",
            "aluminium",
            "blei",
            "kadmium",
            # Body systems (English and German)
            "thyroid",
            "adrenal",
            "liver",
            "kidney",
            "brain",
            "nervous system",
            "immune system",
            "cardiovascular",
            "digestive system",
            "gut microbiome",
            "schilddrüse",
            "leber",
            "niere",
            "gehirn",
            "nervensystem",
            # Therapeutic terms (English and German)
            "standardized extract",
            "bioactive compound",
            "active ingredient",
            "therapeutic dose",
            "clinical trial",
            "placebo-controlled",
            "double-blind",
            "efficacy",
            "safety profile",
            "contraindication",
            "side effect",
            "drug interaction",
            "synergistic effect",
            "standardisiert",
            "extrakt",
            "dosierung",
            "wirkstoff",
            "nebenwirkung",
            "wechselwirkung",
            "kontraindikation",
            "sicherheit",
            "qualität",
            "flavonglykoside",
            "terpenlactone",
            "ginsenoside",
            "rosavine",
            "salidrosid",
        ]

        for term in medical_terms:
            if term in text:
                keywords.add(term)
                # Also add individual words from multi-word terms
                if " " in term:
                    for word in term.split():
                        if len(word) >= 3:
                            keywords.add(word)

        # Extract dosage-related terms
        dosage_patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:mg|g|ml|mcg|iu|units?)",
            r"(\d+(?:\.\d+)?)\s*(?:times?|x)\s*(?:daily|per day|weekly)",
            r"standardized\s+extract",
            r"(\d+)%\s*(?:extract|concentration)",
        ]

        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    keywords.update(match)
                else:
                    keywords.add(match)

        return keywords

    def _extract_technical_keywords(self, text: str) -> Set[str]:
        """Extract technical keywords.

        Args:
            text: Lowercase text

        Returns:
            Set of technical keywords
        """
        keywords = set()

        technical_terms = [
            "algorithm",
            "database",
            "server",
            "network",
            "api",
            "framework",
            "software",
            "hardware",
            "programming",
            "code",
            "system",
            "application",
            "interface",
            "protocol",
            "security",
            "encryption",
            "authentication",
            "authorization",
            "scalability",
            "performance",
            "optimization",
        ]

        for term in technical_terms:
            if term in text:
                keywords.add(term)

        return keywords

    def _extract_general_keywords(self, text: str) -> Set[str]:
        """Extract general keywords using enhanced analysis for German and English.

        Args:
            text: Lowercase text

        Returns:
            Set of general keywords
        """
        keywords = set()

        # Domain-specific terms that should always be considered keywords
        important_terms = {
            # Scientific/Medical terms
            "toxoplasma",
            "gondii",
            "parasiten",
            "verstandesparasiten",
            "biophysik",
            "biophysiker",
            "amygdala",
            "dopamin",
            "neurotransmitter",
            "schizophrenie",
            "bewusstsein",
            "gehirn",
            "blut-hirn-schranke",
            "immunsystem",
            "mikroorganismen",
            "parasitologie",
            "infektion",
            "beta-carboline",
            "harmene",
            "noharmene",
            "steppenraute",
            "soma",
            "upanishaden",
            "zirbeldrüse",
            "ekstase",
            "freude",
            "emotionen",
            "aggressionen",
            "neokortex",
            "reptiliengehirn",
            "lymmische",
            "systeme",
            "metastudie",
            "universitäten",
            "forschung",
            "wissenschaft",
            "studien",
            "ergebnisse",
            "statistik",
            "prozent",
            "weltbevölkerung",
            "deutschland",
            "österreich",
            "schweiz",
            "island",
            "staatspräsident",
            "herausforderung",
            "menschheit",
            "gesellschaft",
            "kollektiv",
            "individuum",
            "verhalten",
            "psyche",
            "depression",
            "psychopharmaka",
            "behandlung",
            "therapie",
            "therapeut",
            "medizin",
            "nahrung",
            "fleisch",
            "schinken",
            "salami",
            "obst",
            "gemüse",
            "trinkwasser",
            "katzen",
            "mäuse",
            "endwirt",
            "zwischenwirt",
            "evolution",
            "übertragung",
            "infektion",
            "schwangerschaft",
            "schwangere",
            "risiko",
            "gefahr",
            "prävention",
            "schutz",
            # Names and proper nouns
            "dieter",
            "broers",
            "dirk",
            "schumann",
            "armin",
            "risi",
            "pfleger",
            "prag",
            "tschechien",
            "iran",
            "südamerika",
            "griechenland",
            "ministerium",
            "who",
            # Technical terms
            "elektronenrastermikroskop",
            "vergrößerung",
            "geometrie",
            "kristall",
            "oszillator",
            "schwingkreis",
            "widerstand",
            "kondensator",
            "spule",
            "halbmeiter",
            "bauteile",
            "proto-zooinen",
            "lebewesen",
            "milliarden",
            "jahre",
            "rna",
            "dna",
            "klassifizierung",
            "virologie",
            "viren",
            "funktionsautomaten",
            "intelligenz",
            "schwarmbewusstsein",
            "kommunikation",
            "eigenschaften",
            "verhalten",
            "manipulation",
            "kontrolle",
        }

        # Extract words that appear multiple times and are significant
        words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text)  # Include German umlauts
        word_freq = {}

        for word in words:
            word_lower = word.lower()
            if word_lower not in self.stop_words:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        # Add important terms that appear in text
        for term in important_terms:
            if term in text:
                keywords.add(term)

        # Select words that appear at least twice or are long or are scientifically relevant
        for word, freq in word_freq.items():
            if (
                freq >= 2
                or len(word) >= 7
                or any(
                    scientific in word
                    for scientific in [
                        "parasit",
                        "toxo",
                        "neuro",
                        "bio",
                        "psych",
                        "medizin",
                        "wissenschaft",
                    ]
                )
            ):
                keywords.add(word)

        # Filter out remaining common words that might have slipped through
        filtered_keywords = set()
        for keyword in keywords:
            # Keep if it's a compound word, proper noun, or scientific term
            if (
                len(keyword) >= 6
                or keyword[0].isupper()
                or any(
                    sci in keyword
                    for sci in ["parasit", "toxo", "neuro", "bio", "psych", "medizin"]
                )
                or keyword in important_terms
            ):
                filtered_keywords.add(keyword)

        return filtered_keywords

    def _create_keyword_entity(
        self, keyword: str, chunk_id: str, document_id: str, domain: str
    ) -> Entity:
        """Create a keyword entity.

        Args:
            keyword: Keyword text
            chunk_id: Source chunk ID
            document_id: Source document ID
            domain: Domain context

        Returns:
            Keyword entity
        """
        # Generate deterministic ID based on normalized keyword
        normalized_keyword = keyword.lower().strip()
        content_hash = hashlib.md5(normalized_keyword.encode()).hexdigest()[:12]
        entity_id = f"ent_{content_hash}"

        return Entity(
            id=entity_id,
            name=keyword,
            type="ENTITY",  # Use generic ENTITY type for consistency
            confidence=0.7,
            source_doc_id=document_id,
            attributes={
                "keyword_type": "domain_specific",
                "domain": domain,
                "source_chunk_id": chunk_id,
                "normalized_name": normalized_keyword,
                "original_type": "KEYWORD",  # Keep original semantic type for reference
            },
        )

    def _create_keyword_chunk_relationship(
        self, keyword_entity_id: str, chunk_id: str, keyword: str
    ) -> Relation:
        """Create a relationship between keyword entity and chunk.

        Args:
            keyword_entity_id: Keyword entity ID
            chunk_id: Chunk ID
            keyword: Keyword text

        Returns:
            Relation object
        """
        # Generate relation ID
        content_for_hash = f"{keyword_entity_id}_DESCRIBES_{chunk_id}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
        rel_id = f"rel_{content_hash}"

        return Relation(
            id=rel_id,
            source_entity_id=keyword_entity_id,
            target_entity_id=chunk_id,
            type="DESCRIBES",
            description=f"Keyword '{keyword}' describes content in this chunk",
            confidence=0.8,
            attributes={"relationship_category": "keyword_chunk"},
        )
