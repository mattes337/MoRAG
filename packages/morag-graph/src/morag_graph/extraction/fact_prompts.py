"""Prompts for fact extraction using LLMs."""

from typing import Dict, List, Any


class FactExtractionPrompts:
    """Prompts for fact extraction using LLMs."""
    
    @staticmethod
    def get_fact_extraction_prompt(domain: str = "general", language: str = "en") -> str:
        """Get the main fact extraction prompt.
        
        Args:
            domain: Domain context for extraction
            language: Language of the text
            
        Returns:
            Formatted prompt for fact extraction
        """
        return """You are a knowledge extraction expert. Extract structured facts from the following text that represent actionable, specific information.

A fact should contain:
- Subject: The main entity or concept the fact is about
- Object: What is being described, studied, or acted upon
- Approach: The method, technique, or way something is done (optional)
- Solution: The result, outcome, or answer provided (optional)
- Remarks: Important context, limitations, or qualifications (optional)

Extract only facts that are:
1. Specific and actionable (not generic statements)
2. Verifiable from the text
3. Useful for answering questions
4. Complete enough to stand alone

Domain: """ + domain + """
Language: """ + language + """

Text: {chunk_text}

Respond with JSON array of facts:
[
  {{
    "subject": "specific subject",
    "object": "what is being described",
    "approach": "how it's done (optional)",
    "solution": "outcome/result (optional)",
    "remarks": "context/limitations (optional)",
    "fact_type": "research|process|definition|causal|comparative|temporal|statistical|methodological",
    "confidence": 0.0-1.0,
    "keywords": ["key", "terms"]
  }}
]

Important guidelines:
- Only extract facts that are explicitly stated in the text
- Do not infer or hallucinate information
- Each fact must be self-contained and understandable without external context
- Focus on actionable knowledge rather than generic descriptions
- Ensure high confidence (>0.7) for extracted facts"""

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
{{
  "fact_type": "most_appropriate_type",
  "confidence": 0.0-1.0,
  "reasoning": "explanation for the classification"
}}"""

    @staticmethod
    def get_fact_relationship_prompt() -> str:
        """Get prompt for identifying relationships between facts.
        
        Returns:
            Prompt for fact relationship extraction
        """
        return """Given these facts from the same document, identify semantic relationships:

Facts: {facts_list}

Identify relationships like:
- SUPPORTS: One fact provides evidence for another
- ELABORATES: One fact provides more detail about another
- CONTRADICTS: Facts that present conflicting information
- SEQUENCE: Facts that represent steps in a process
- COMPARISON: Facts that compare different approaches/solutions
- CAUSATION: One fact describes the cause of another
- TEMPORAL_ORDER: Facts that have a time-based sequence

Only create relationships that are clearly supported by the text.

Respond with JSON array:
[
  {{
    "source_fact_id": "fact_id_1",
    "target_fact_id": "fact_id_2",
    "relation_type": "SUPPORTS|ELABORATES|CONTRADICTS|SEQUENCE|COMPARISON|CAUSATION|TEMPORAL_ORDER",
    "confidence": 0.0-1.0,
    "context": "explanation of the relationship"
  }}
]"""

    @staticmethod
    def get_keyword_extraction_prompt() -> str:
        """Get prompt for extracting keywords from facts.
        
        Returns:
            Prompt for keyword extraction
        """
        return """Extract relevant keywords from this fact for indexing and search:

Fact: {fact_json}

Extract keywords that are:
1. Specific and meaningful (not generic words)
2. Useful for search and retrieval
3. Representative of the fact's content
4. Include technical terms, proper nouns, and key concepts

Respond with JSON:
{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "primary_keywords": ["most_important", "keywords"],
  "domain_keywords": ["domain_specific", "terms"]
}"""

    @staticmethod
    def create_extraction_prompt(
        chunk_text: str,
        domain: str = "general",
        language: str = "en",
        max_facts: int = 10
    ) -> str:
        """Create a complete extraction prompt with text.
        
        Args:
            chunk_text: Text to extract facts from
            domain: Domain context
            language: Language of the text
            max_facts: Maximum number of facts to extract
            
        Returns:
            Complete prompt ready for LLM
        """
        base_prompt = FactExtractionPrompts.get_fact_extraction_prompt(domain, language)
        
        return base_prompt.format(
            chunk_text=chunk_text
        ) + f"\n\nExtract at most {max_facts} high-quality facts from the text."

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
    def create_relationship_prompt(facts: List[Dict[str, Any]]) -> str:
        """Create a relationship extraction prompt for multiple facts.
        
        Args:
            facts: List of fact dictionaries
            
        Returns:
            Complete relationship extraction prompt
        """
        import json
        facts_json = json.dumps(facts, indent=2)
        return FactExtractionPrompts.get_fact_relationship_prompt().format(
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
