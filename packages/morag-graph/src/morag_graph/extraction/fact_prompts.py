"""Prompts for fact extraction using LLMs."""

from typing import Dict, List, Any


class FactExtractionPrompts:
    """Prompts for fact extraction using LLMs."""
    
    @staticmethod
    def get_fact_extraction_prompt(domain: str = "general", language: str = "en", query_context: str = None) -> str:
        """Get the enhanced fact extraction prompt with few-shot examples and query context.

        Args:
            domain: Domain context for extraction
            language: Language of the text
            query_context: Optional query context for relevance-aware extraction

        Returns:
            Formatted prompt for fact extraction
        """
        # Create language-specific instruction
        language_names = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        language_name = language_names.get(language, language)

        language_instruction = f"""
CRITICAL LANGUAGE REQUIREMENT:
You MUST extract and write ALL fact content (subject, object, approach, solution, condition, remarks, keywords) in {language_name} ({language}).
Do NOT translate to English or any other language.
The extracted facts must be in the SAME language as the input text.
If the input text is in {language_name}, your response must be entirely in {language_name}.
"""

        # Add query context awareness if provided
        context_instruction = ""
        if query_context:
            context_instruction = f"""
QUERY CONTEXT AWARENESS:
This extraction is being performed to help answer: "{query_context}"
Prioritize facts that are directly relevant to this query context.
Focus on extracting information that would help answer this specific question or topic.
"""

        return language_instruction + context_instruction + f"""
You are a knowledge extraction expert. Extract self-contained facts from the following text that represent specific, domain-relevant information that can be used to answer questions or understand concepts.

HYBRID FACT APPROACH:
Each fact should be a complete, self-contained statement that includes all necessary context and can stand alone. Additionally, extract structured metadata to support knowledge graph construction.

FACT STRUCTURE:
- fact_text: A comprehensive, complete statement that includes all relevant information with full context
- structured_metadata: Key entities, relationships, and concepts for graph building
  * primary_entities: Main entities mentioned in the fact
  * relationships: Important actions, connections, or relationships
  * domain_concepts: Domain-specific technical terms and concepts

QUALITY REQUIREMENTS:
- Facts must be SPECIFIC and DOMAIN-RELEVANT, not overly generic
- Include exact dosages, measurements, frequencies, and durations when mentioned
- Each fact must be SELF-CONTAINED and usable without the original text
- Write facts as complete sentences that provide full context
- AVOID overly generic advice like "take a break", "consult a doctor", "eat healthy", "exercise regularly"
- Focus on domain-specific knowledge including:
  * Medical conditions, symptoms, and diagnostic criteria
  * Treatment methods, medications, and therapeutic approaches
  * Educational content about diseases, disorders, and health conditions
  * Specific techniques, procedures, and interventions
  * Characteristic features, types, and classifications
- INCLUDE educational facts about medical conditions, symptoms, types, and characteristics
- REJECT only facts that are completely generic life advice with no domain specificity
- If no domain-relevant facts are found, return an empty array []

FEW-SHOT EXAMPLES:

Example 1 (Medical/Herbal):
Text: "For chronic stress and anxiety, use 300-600mg of standardized Ashwagandha extract (containing 5% withanolides) twice daily with meals. Treatment should be continued for 8-12 weeks for optimal results. Do not exceed 2 weeks without a break."

Extracted Fact:
{{
  "fact_text": "For chronic stress and anxiety management, standardized Ashwagandha extract containing 5% withanolides should be taken at 300-600mg twice daily with meals for 8-12 weeks for optimal results, with breaks not exceeding 2 weeks to ensure safety and effectiveness.",
  "structured_metadata": {{
    "primary_entities": ["Ashwagandha extract", "chronic stress", "anxiety", "withanolides"],
    "relationships": ["treats", "manages", "reduces"],
    "domain_concepts": ["standardized extract", "dosage", "treatment duration", "herbal medicine"]
  }},
  "fact_type": "methodological",
  "confidence": 0.9,
  "keywords": ["ashwagandha", "withanolides", "adaptogen", "chronic stress", "anxiety management"]
}}

Example 2 (Technical/Engineering):
Text: "To optimize PostgreSQL query performance, create a B-tree index on frequently queried columns using CREATE INDEX idx_name ON table_name (column_name). For composite queries, use multi-column indexes with the most selective column first. Rebuild indexes monthly using REINDEX."

Extracted Fact:
{{
  "fact_text": "PostgreSQL query performance can be optimized by creating B-tree indexes on frequently queried columns using CREATE INDEX syntax, with composite queries requiring multi-column indexes where the most selective column is placed first, and indexes should be rebuilt monthly using REINDEX for optimal performance.",
  "structured_metadata": {{
    "primary_entities": ["PostgreSQL", "B-tree index", "query performance", "database optimization"],
    "relationships": ["optimizes", "improves", "requires"],
    "domain_concepts": ["CREATE INDEX", "composite queries", "column selectivity", "REINDEX", "database maintenance"]
  }},
  "fact_type": "methodological",
  "confidence": 0.95,
  "keywords": ["PostgreSQL", "B-tree index", "query optimization", "database performance", "REINDEX"]
}}

Example 3 (Medical/Educational):
Text: "ADHS zeigt sich in drei Haupttypen: der unaufmerksame Typ mit Konzentrationsschwierigkeiten und Vergesslichkeit, der hyperaktiv-impulsive Typ mit Unruhe und spontanen Handlungen, und der Mischtyp, der beide Symptomgruppen kombiniert. Der Mischtyp hat besondere Herausforderungen im Alltag."

Extracted Fact:
{{
  "fact_text": "ADHS manifestiert sich in drei Haupttypen: dem unaufmerksamen Typ mit Konzentrationsschwierigkeiten und Vergesslichkeit, dem hyperaktiv-impulsiven Typ mit Unruhe und spontanen Handlungen, und dem Mischtyp, der beide Symptomgruppen kombiniert und besondere Herausforderungen im Alltag mit sich bringt.",
  "structured_metadata": {{
    "primary_entities": ["ADHS", "unaufmerksamer Typ", "hyperaktiv-impulsiver Typ", "Mischtyp"],
    "relationships": ["manifestiert sich", "kombiniert", "verursacht"],
    "domain_concepts": ["Konzentrationsschwierigkeiten", "Vergesslichkeit", "Unruhe", "spontane Handlungen", "Alltagsherausforderungen"]
  }},
  "fact_type": "descriptive",
  "confidence": 0.9,
  "keywords": ["ADHS", "Mischtyp", "Unaufmerksamkeit", "Hyperaktivität", "Impulsivität", "Symptomatik"]
}}

Example 4 (Business/Finance):
Text: "For small business cash flow management, maintain 3-6 months of operating expenses in reserve. Use the 50/30/20 rule: 50% for essential expenses, 30% for growth investments, 20% for emergency fund. Review monthly and adjust based on seasonal patterns."

Extracted Fact:
{{
  "fact_text": "Small business cash flow management requires maintaining 3-6 months of operating expenses in reserve and following the 50/30/20 rule where 50% goes to essential expenses, 30% to growth investments, and 20% to emergency fund, with monthly reviews and seasonal adjustments to ensure financial stability.",
  "structured_metadata": {{
    "primary_entities": ["small business", "cash flow management", "50/30/20 rule", "operating expenses", "emergency fund"],
    "relationships": ["requires", "allocates", "maintains"],
    "domain_concepts": ["essential expenses", "growth investments", "financial reserves", "seasonal adjustments", "monthly review"]
  }},
  "fact_type": "methodological",
  "confidence": 0.85,
  "keywords": ["cash flow management", "50/30/20 rule", "small business", "financial planning", "emergency fund"]
}}

Example 5 (Legal/Compliance):
Text: "Under GDPR Article 17, individuals have the right to erasure (right to be forgotten) within 30 days of request. Organizations must delete personal data unless legal obligations require retention. Document all deletion requests and maintain audit logs for compliance verification."

Extracted Fact:
{{
  "fact_text": "Under GDPR Article 17, individuals have the right to erasure (right to be forgotten) which requires organizations to delete personal data within 30 days of request unless legal obligations require retention, with all deletion requests documented and audit logs maintained for compliance verification.",
  "structured_metadata": {{
    "primary_entities": ["GDPR Article 17", "right to erasure", "personal data", "organizations", "legal obligations"],
    "relationships": ["requires", "delete", "document", "maintain"],
    "domain_concepts": ["data deletion", "compliance verification", "audit logs", "retention requirements", "privacy rights"]
  }},
  "fact_type": "regulatory",
  "confidence": 0.95,
  "keywords": ["GDPR", "right to erasure", "data deletion", "compliance", "audit logs"]
}}

Domain: {domain}
Target Language: {language_name} ({language})

REMEMBER: Extract ALL content in {language_name}. Do NOT use English or any other language.

Text: {{{{chunk_text}}}}

Respond with JSON array of facts in {language_name}:
[
  {{{{
    "fact_text": "Complete, self-contained fact statement with full context",
    "structured_metadata": {{
      "primary_entities": ["key", "entities", "mentioned"],
      "relationships": ["important", "actions", "connections"],
      "domain_concepts": ["domain-specific", "technical", "terms"]
    }},
    "fact_type": "process|definition|causal|methodological|safety|descriptive|regulatory",
    "confidence": 0.0-1.0,
    "query_relevance": 0.0-1.0,
    "evidence_strength": "strong|moderate|weak",
    "keywords": ["searchable", "terms"],
    "source_span": "exact text span supporting this fact"
  }}}}
]

CRITICAL GUIDELINES:
- LANGUAGE: Extract ALL content in {language_name} ({language}). Never use English or other languages.
- Write complete, self-contained fact_text that includes all necessary context
- Only extract facts with high practical value and specificity
- Include exact measurements, frequencies, and specifications in the fact_text
- Extract meaningful entities and relationships in structured_metadata
- Keywords should become separate entities in the knowledge graph
- Ensure confidence >0.7 for all extracted facts
- Avoid trivial or commonly known information

STRICT FILTERING RULES:
- REJECT any fact that mentions generic advice like "take breaks", "rest", "relax", "exercise"
- REJECT any fact about general lifestyle recommendations
- REJECT any fact that doesn't mention specific substances, dosages, techniques, or procedures
- ACCEPT only facts that contain domain-specific knowledge with measurable parameters
- ACCEPT only facts that mention specific plants, compounds, techniques, or medical procedures
- If the text contains only general advice, return an empty array: []

EXAMPLES OF INVALID FACTS TO REJECT:
- "Take regular breaks to reduce stress" (generic advice)
- "Exercise helps with ADHD" (no specific measurements)
- "Healthy diet is important" (vague recommendation)
- "Get enough sleep" (no specific guidance)
- "Consult your doctor" (generic referral)
- "Use best practices" (no specific practices mentioned)
- "Follow industry standards" (no specific standards)
- "Implement proper security" (no specific measures)

EXAMPLES OF VALID FACTS TO EXTRACT:
- "Ginkgo biloba 120mg twice daily improves concentration in ADHD patients" (specific dosage, frequency, condition)
- "PostgreSQL connection pooling with 20 max connections reduces response time by 40%" (specific configuration, measurable outcome)
- "GDPR requires data breach notification within 72 hours to supervisory authorities" (specific timeframe, legal requirement)
- "React useEffect with empty dependency array runs only on component mount" (specific technical behavior)
- "Compound annual growth rate (CAGR) formula: (Ending Value/Beginning Value)^(1/years) - 1" (specific calculation method)

Remember: If no specific, measurable, domain-relevant facts are found, return []"""

    @staticmethod
    def get_fact_validation_prompt() -> str:
        """Get prompt for validating extracted facts.
        
        Returns:
            Prompt for fact quality validation
        """
        return """Evaluate the quality of this extracted fact:

Fact: {fact_json}

Rate the fact on:
1. Specificity (0-1): Is it specific rather than generic?
2. Actionability (0-1): Does it provide useful, applicable information?
3. Completeness (0-1): Does it contain sufficient context?
4. Verifiability (0-1): Can it be traced to source text?

Respond with JSON:
{
  "overall_score": 0.0-1.0,
  "specificity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "completeness": 0.0-1.0,
  "verifiability": 0.0-1.0,
  "issues": ["list of specific issues"],
  "suggestions": ["improvement suggestions"]
}"""

    @staticmethod
    def get_fact_type_classification_prompt() -> str:
        """Get prompt for classifying fact types.
        
        Returns:
            Prompt for fact type classification
        """
        return """Classify the type of this fact:

Fact: {fact_json}

Available fact types:
- research: Findings, studies, experimental results
- process: How-to information, procedures, workflows
- definition: What something is, characteristics, properties
- causal: Cause-effect relationships, dependencies
- comparative: Comparisons, evaluations, rankings
- temporal: Time-based information, sequences, chronology
- statistical: Numbers, measurements, quantitative data
- methodological: Methods, techniques, approaches

Respond with JSON:
{
  "fact_type": "most_appropriate_type",
  "confidence": 0.0-1.0,
  "reasoning": "explanation for the classification"
}"""

    @staticmethod
    def get_fact_relationship_prompt(language: str = "en") -> str:
        """Get prompt for identifying relationships between facts.

        Args:
            language: Language for the response

        Returns:
            Prompt for fact relationship extraction
        """
        # Create language-specific instruction
        language_names = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        language_name = language_names.get(language, language)

        language_instruction = f"""
CRITICAL LANGUAGE REQUIREMENT:
You MUST write ALL relationship context and explanations in {language_name} ({language}).
Do NOT use English or any other language. The response must be in {language_name}.
"""

        return language_instruction + """
Given these facts from the same document, identify semantic relationships:

Facts: {facts_list}

Identify relationships like:
- SUPPORTS: One fact provides evidence for another
- ELABORATES: One fact provides more detail about another
- CONTRADICTS: Facts that present conflicting information
- SEQUENCE: Facts that represent steps in a process
- COMPARISON: Facts that compare different approaches/solutions
- CAUSATION: One fact describes the cause of another
- PREREQUISITE: Required condition/dependency
- ALTERNATIVE: Alternative approach/solution
- HIERARCHY: Parent-child or containment relationship

For each relationship, assess:
- Relationship strength: direct (explicitly stated) | inferred (logically derivable) | contextual (domain knowledge)
- Evidence quality: explicit (clearly stated) | implicit (requires inference) | speculative (uncertain)

Only create relationships that are clearly supported by the text.

Respond with JSON array:
[
  {{
    "source_fact_id": "fact_id_1",
    "target_fact_id": "fact_id_2",
    "relation_type": "SUPPORTS|ELABORATES|CONTRADICTS|SEQUENCE|COMPARISON|CAUSATION|PREREQUISITE|ALTERNATIVE|HIERARCHY",
    "confidence": 0.0-1.0,
    "relationship_strength": "direct|inferred|contextual",
    "evidence_quality": "explicit|implicit|speculative",
    "context": "explanation of the relationship",
    "source_evidence": "text span supporting this relationship"
  }}
]"""

    @staticmethod
    def get_keyword_extraction_prompt() -> str:
        """Get prompt for extracting keywords from facts.

        Returns:
            Prompt for keyword extraction
        """
        return """Extract domain-specific technical keywords from this fact for indexing and search:

Fact: {fact_json}

Extract keywords that are:
1. Technical terms and scientific names (e.g., "ginsenosides", "flavonglycosides", "terpenlactones")
2. Medical/therapeutic terms (e.g., "adaptogenic", "nootropic", "bioavailability")
3. Measurement units and specifications (e.g., "standardized extract", "mg/kg", "bioactive compounds")
4. Condition-specific terminology (e.g., "cognitive enhancement", "neuroprotective", "anxiolytic")
5. NOT simple words from subject/object (avoid duplicating basic terms)

Focus on terms that would help someone search for similar information or related facts.

Respond with JSON:
{
  "keywords": ["technical_term1", "scientific_name2", "therapeutic_class3"],
  "primary_keywords": ["most_important", "for_search"],
  "domain_keywords": ["highly_specific", "technical_terms"]
}"""

    @staticmethod
    def create_extraction_prompt(
        chunk_text: str,
        domain: str = "general",
        language: str = "en",
        max_facts: int = 10,
        query_context: str = None
    ) -> str:
        """Create a complete extraction prompt with text.

        Args:
            chunk_text: Text to extract facts from
            domain: Domain context
            language: Language of the text
            max_facts: Maximum number of facts to extract
            query_context: Optional query context for relevance-aware extraction

        Returns:
            Complete prompt ready for LLM
        """
        base_prompt = FactExtractionPrompts.get_fact_extraction_prompt(domain, language, query_context)

        # Replace the placeholder with actual text
        complete_prompt = base_prompt.replace("{{chunk_text}}", chunk_text)

        # Debug: Log the prompt replacement
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Prompt replacement debug - chunk_text length: {len(chunk_text)}")
        logger.debug(f"Base prompt contains {{{{chunk_text}}}}: {'{{chunk_text}}' in base_prompt}")
        logger.debug(f"Complete prompt contains chunk text: {chunk_text[:100] in complete_prompt if len(chunk_text) > 100 else chunk_text in complete_prompt}")

        return complete_prompt + f"\n\nExtract at most {max_facts} high-quality facts from the text."

    @staticmethod
    def create_validation_prompt(fact_dict: Dict[str, Any]) -> str:
        """Create a validation prompt for a specific fact.
        
        Args:
            fact_dict: Fact data as dictionary
            
        Returns:
            Complete validation prompt
        """
        import json
        fact_json = json.dumps(fact_dict, indent=2)
        return FactExtractionPrompts.get_fact_validation_prompt().format(
            fact_json=fact_json
        )

    @staticmethod
    def create_relationship_prompt(facts: List[Dict[str, Any]], language: str = "en") -> str:
        """Create a relationship extraction prompt for multiple facts.

        Args:
            facts: List of fact dictionaries
            language: Language for the response

        Returns:
            Complete relationship extraction prompt
        """
        import json
        facts_json = json.dumps(facts, indent=2)
        return FactExtractionPrompts.get_fact_relationship_prompt(language).format(
            facts_list=facts_json
        )

    @staticmethod
    def create_keyword_prompt(fact_dict: Dict[str, Any]) -> str:
        """Create a keyword extraction prompt for a specific fact.
        
        Args:
            fact_dict: Fact data as dictionary
            
        Returns:
            Complete keyword extraction prompt
        """
        import json
        fact_json = json.dumps(fact_dict, indent=2)
        return FactExtractionPrompts.get_keyword_extraction_prompt().format(
            fact_json=fact_json
        )


class FactPromptTemplates:
    """Template variations for different domains and use cases."""
    
    RESEARCH_DOMAIN_PROMPT = """You are extracting facts from research literature. Focus on:
- Research findings and results
- Methodologies and approaches
- Statistical data and measurements
- Experimental procedures
- Theoretical frameworks
- Limitations and future work

{base_prompt}"""

    TECHNICAL_DOMAIN_PROMPT = """You are extracting facts from technical documentation. Focus on:
- Implementation details and procedures
- Configuration and setup instructions
- Technical specifications
- Troubleshooting information
- Best practices and recommendations
- System requirements and dependencies

{base_prompt}"""

    BUSINESS_DOMAIN_PROMPT = """You are extracting facts from business documents. Focus on:
- Processes and workflows
- Policies and procedures
- Performance metrics and KPIs
- Strategic decisions and rationale
- Market analysis and insights
- Organizational information

{base_prompt}"""

    @classmethod
    def get_domain_prompt(cls, domain: str, base_prompt: str) -> str:
        """Get domain-specific prompt template.
        
        Args:
            domain: Domain name
            base_prompt: Base extraction prompt
            
        Returns:
            Domain-enhanced prompt
        """
        domain_templates = {
            "research": cls.RESEARCH_DOMAIN_PROMPT,
            "technical": cls.TECHNICAL_DOMAIN_PROMPT,
            "business": cls.BUSINESS_DOMAIN_PROMPT,
        }
        
        template = domain_templates.get(domain.lower())
        if template:
            return template.format(base_prompt=base_prompt)
        return base_prompt
