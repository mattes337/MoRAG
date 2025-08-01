"""Domain-specific examples for LangExtract entity and relation extraction."""

from typing import List, Dict, Any

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None


class LangExtractExamples:
    """Factory for creating domain-specific LangExtract examples."""
    
    @staticmethod
    def get_entity_examples(domain: str = "general") -> List[Any]:
        """Get entity extraction examples for a specific domain.
        
        Args:
            domain: Domain name (general, medical, technical, legal, business, scientific)
            
        Returns:
            List of LangExtract ExampleData objects
        """
        if not LANGEXTRACT_AVAILABLE:
            return []
        
        if domain == "medical":
            return LangExtractExamples._get_medical_entity_examples()
        elif domain == "technical":
            return LangExtractExamples._get_technical_entity_examples()
        elif domain == "legal":
            return LangExtractExamples._get_legal_entity_examples()
        elif domain == "business":
            return LangExtractExamples._get_business_entity_examples()
        elif domain == "scientific":
            return LangExtractExamples._get_scientific_entity_examples()
        else:
            return LangExtractExamples._get_general_entity_examples()
    
    @staticmethod
    def get_relation_examples(domain: str = "general") -> List[Any]:
        """Get relation extraction examples for a specific domain.
        
        Args:
            domain: Domain name (general, medical, technical, legal, business, scientific)
            
        Returns:
            List of LangExtract ExampleData objects
        """
        if not LANGEXTRACT_AVAILABLE:
            return []
        
        if domain == "medical":
            return LangExtractExamples._get_medical_relation_examples()
        elif domain == "technical":
            return LangExtractExamples._get_technical_relation_examples()
        elif domain == "legal":
            return LangExtractExamples._get_legal_relation_examples()
        elif domain == "business":
            return LangExtractExamples._get_business_relation_examples()
        elif domain == "scientific":
            return LangExtractExamples._get_scientific_relation_examples()
        else:
            return LangExtractExamples._get_general_relation_examples()
    
    @staticmethod
    def _get_general_entity_examples() -> List[Any]:
        """General domain entity examples."""
        return [
            lx.data.ExampleData(
                text="Dr. Sarah Johnson works as a researcher at Google in Mountain View, California.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Dr. Sarah Johnson",
                        attributes={"title": "Dr.", "role": "researcher"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organization",
                        extraction_text="Google",
                        attributes={"type": "technology_company"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Mountain View",
                        attributes={"type": "city", "state": "California"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="California",
                        attributes={"type": "state", "country": "USA"}
                    ),
                ]
            )
        ]
    
    @staticmethod
    def _get_medical_entity_examples() -> List[Any]:
        """Medical domain entity examples."""
        return [
            lx.data.ExampleData(
                text="Patient John Smith was prescribed Metformin 500mg twice daily for Type 2 diabetes management by Dr. Emily Chen at Mayo Clinic.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="patient",
                        extraction_text="John Smith",
                        attributes={"role": "patient"}
                    ),
                    lx.data.Extraction(
                        extraction_class="medication",
                        extraction_text="Metformin",
                        attributes={"dosage": "500mg", "frequency": "twice daily"}
                    ),
                    lx.data.Extraction(
                        extraction_class="condition",
                        extraction_text="Type 2 diabetes",
                        attributes={"type": "chronic_disease", "category": "metabolic"}
                    ),
                    lx.data.Extraction(
                        extraction_class="doctor",
                        extraction_text="Dr. Emily Chen",
                        attributes={"title": "Dr.", "role": "physician"}
                    ),
                    lx.data.Extraction(
                        extraction_class="medical_facility",
                        extraction_text="Mayo Clinic",
                        attributes={"type": "hospital", "reputation": "renowned"}
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="The patient presented with chest pain and shortness of breath. Blood pressure was 140/90 mmHg.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="symptom",
                        extraction_text="chest pain",
                        attributes={"severity": "presenting", "type": "pain"}
                    ),
                    lx.data.Extraction(
                        extraction_class="symptom",
                        extraction_text="shortness of breath",
                        attributes={"type": "respiratory", "severity": "presenting"}
                    ),
                    lx.data.Extraction(
                        extraction_class="vital_sign",
                        extraction_text="Blood pressure",
                        attributes={"value": "140/90 mmHg", "status": "elevated"}
                    ),
                ]
            )
        ]
    
    @staticmethod
    def _get_technical_entity_examples() -> List[Any]:
        """Technical domain entity examples."""
        return [
            lx.data.ExampleData(
                text="The microservice architecture uses Docker containers deployed on Kubernetes cluster with PostgreSQL database and Redis cache.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="architecture_pattern",
                        extraction_text="microservice architecture",
                        attributes={"type": "software_architecture", "style": "distributed"}
                    ),
                    lx.data.Extraction(
                        extraction_class="technology",
                        extraction_text="Docker",
                        attributes={"type": "containerization", "category": "devops"}
                    ),
                    lx.data.Extraction(
                        extraction_class="platform",
                        extraction_text="Kubernetes",
                        attributes={"type": "orchestration", "category": "container_management"}
                    ),
                    lx.data.Extraction(
                        extraction_class="database",
                        extraction_text="PostgreSQL",
                        attributes={"type": "relational_database", "category": "storage"}
                    ),
                    lx.data.Extraction(
                        extraction_class="cache",
                        extraction_text="Redis",
                        attributes={"type": "in_memory_cache", "category": "performance"}
                    ),
                ]
            )
        ]
    
    @staticmethod
    def _get_legal_entity_examples() -> List[Any]:
        """Legal domain entity examples."""
        return [
            lx.data.ExampleData(
                text="The Supreme Court ruled on the contract dispute under Section 15 of the Commercial Code, with Justice Roberts writing the majority opinion.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="court",
                        extraction_text="Supreme Court",
                        attributes={"level": "highest", "jurisdiction": "federal"}
                    ),
                    lx.data.Extraction(
                        extraction_class="legal_document",
                        extraction_text="contract",
                        attributes={"type": "agreement", "status": "disputed"}
                    ),
                    lx.data.Extraction(
                        extraction_class="statute",
                        extraction_text="Section 15 of the Commercial Code",
                        attributes={"type": "law", "area": "commercial"}
                    ),
                    lx.data.Extraction(
                        extraction_class="judge",
                        extraction_text="Justice Roberts",
                        attributes={"title": "Justice", "role": "majority_author"}
                    ),
                ]
            )
        ]
    
    @staticmethod
    def _get_business_entity_examples() -> List[Any]:
        """Business domain entity examples."""
        return [
            lx.data.ExampleData(
                text="CEO John Davis announced Q3 revenue of $2.5M, representing 15% growth compared to Q2. The company plans to expand into the European market.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="executive",
                        extraction_text="CEO John Davis",
                        attributes={"title": "CEO", "role": "chief_executive"}
                    ),
                    lx.data.Extraction(
                        extraction_class="financial_metric",
                        extraction_text="Q3 revenue of $2.5M",
                        attributes={"type": "revenue", "period": "Q3", "amount": "$2.5M"}
                    ),
                    lx.data.Extraction(
                        extraction_class="performance_indicator",
                        extraction_text="15% growth",
                        attributes={"type": "growth_rate", "value": "15%", "direction": "positive"}
                    ),
                    lx.data.Extraction(
                        extraction_class="market",
                        extraction_text="European market",
                        attributes={"type": "geographic_market", "region": "Europe", "status": "target"}
                    ),
                ]
            )
        ]
    
    @staticmethod
    def _get_scientific_entity_examples() -> List[Any]:
        """Scientific domain entity examples."""
        return [
            lx.data.ExampleData(
                text="The research team at MIT published findings on quantum computing algorithms in Nature journal, showing 40% improvement in processing speed.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="research_team",
                        extraction_text="research team at MIT",
                        attributes={"institution": "MIT", "type": "academic_research"}
                    ),
                    lx.data.Extraction(
                        extraction_class="research_topic",
                        extraction_text="quantum computing algorithms",
                        attributes={"field": "computer_science", "subfield": "quantum_computing"}
                    ),
                    lx.data.Extraction(
                        extraction_class="publication",
                        extraction_text="Nature journal",
                        attributes={"type": "scientific_journal", "reputation": "high_impact"}
                    ),
                    lx.data.Extraction(
                        extraction_class="research_result",
                        extraction_text="40% improvement in processing speed",
                        attributes={"type": "performance_improvement", "metric": "processing_speed", "value": "40%"}
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_general_relation_examples() -> List[Any]:
        """General domain relation examples."""
        return [
            lx.data.ExampleData(
                text="Dr. Sarah Johnson works as a researcher at Google in Mountain View, California.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="employment",
                        extraction_text="Dr. Sarah Johnson works as a researcher at Google",
                        attributes={
                            "source_entity": "Dr. Sarah Johnson",
                            "target_entity": "Google",
                            "relationship_type": "WORKS_FOR",
                            "role": "researcher"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Google in Mountain View",
                        attributes={
                            "source_entity": "Google",
                            "target_entity": "Mountain View",
                            "relationship_type": "LOCATED_IN"
                        }
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_medical_relation_examples() -> List[Any]:
        """Medical domain relation examples."""
        return [
            lx.data.ExampleData(
                text="Patient John Smith was prescribed Metformin for Type 2 diabetes by Dr. Emily Chen.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="prescription",
                        extraction_text="John Smith was prescribed Metformin",
                        attributes={
                            "source_entity": "John Smith",
                            "target_entity": "Metformin",
                            "relationship_type": "PRESCRIBED",
                            "context": "medication"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="treatment",
                        extraction_text="Metformin for Type 2 diabetes",
                        attributes={
                            "source_entity": "Metformin",
                            "target_entity": "Type 2 diabetes",
                            "relationship_type": "TREATS",
                            "purpose": "management"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="medical_care",
                        extraction_text="prescribed by Dr. Emily Chen",
                        attributes={
                            "source_entity": "Dr. Emily Chen",
                            "target_entity": "John Smith",
                            "relationship_type": "TREATS_PATIENT",
                            "action": "prescribing"
                        }
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="Hashimoto's thyroiditis affects the thyroid gland, causing inflammation and autoimmune damage.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="disease_organ_relationship",
                        extraction_text="Hashimoto's thyroiditis affects the thyroid gland",
                        attributes={
                            "source_entity": "Hashimoto",
                            "target_entity": "Schilddrüse",
                            "relationship_type": "AFFECTS",
                            "mechanism": "autoimmune"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="pathological_process",
                        extraction_text="causing inflammation and autoimmune damage",
                        attributes={
                            "source_entity": "Hashimoto",
                            "target_entity": "Entzündung",
                            "relationship_type": "CAUSES",
                            "process": "inflammation"
                        }
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="Heavy metals like mercury accumulate in brain tissue and damage neurons.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="toxin_accumulation",
                        extraction_text="mercury accumulate in brain tissue",
                        attributes={
                            "source_entity": "Quecksilber",
                            "target_entity": "Gehirn",
                            "relationship_type": "ACCUMULATES_IN",
                            "mechanism": "bioaccumulation"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="toxic_damage",
                        extraction_text="damage neurons",
                        attributes={
                            "source_entity": "Quecksilber",
                            "target_entity": "Neuronen",
                            "relationship_type": "DAMAGES",
                            "effect": "cellular_damage"
                        }
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="Silizium detoxifies aluminum from the body and protects against heavy metal toxicity.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="detoxification",
                        extraction_text="Silizium detoxifies aluminum",
                        attributes={
                            "source_entity": "Silizium",
                            "target_entity": "Aluminium",
                            "relationship_type": "DETOXIFIES",
                            "mechanism": "chelation"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="protection",
                        extraction_text="protects against heavy metal toxicity",
                        attributes={
                            "source_entity": "Silizium",
                            "target_entity": "Schwermetalle",
                            "relationship_type": "PROTECTS",
                            "effect": "protective"
                        }
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_technical_relation_examples() -> List[Any]:
        """Technical domain relation examples."""
        return [
            lx.data.ExampleData(
                text="The microservice uses PostgreSQL database and connects to Redis cache for performance optimization.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="data_storage",
                        extraction_text="microservice uses PostgreSQL database",
                        attributes={
                            "source_entity": "microservice",
                            "target_entity": "PostgreSQL",
                            "relationship_type": "USES",
                            "purpose": "data_storage"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="system_connection",
                        extraction_text="connects to Redis cache",
                        attributes={
                            "source_entity": "microservice",
                            "target_entity": "Redis",
                            "relationship_type": "CONNECTS_TO",
                            "purpose": "performance_optimization"
                        }
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_legal_relation_examples() -> List[Any]:
        """Legal domain relation examples."""
        return [
            lx.data.ExampleData(
                text="The Supreme Court ruled on the contract dispute under Section 15 of the Commercial Code.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="legal_ruling",
                        extraction_text="Supreme Court ruled on the contract dispute",
                        attributes={
                            "source_entity": "Supreme Court",
                            "target_entity": "contract dispute",
                            "relationship_type": "RULED_ON",
                            "action": "judicial_decision"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="legal_basis",
                        extraction_text="ruled under Section 15 of the Commercial Code",
                        attributes={
                            "source_entity": "contract dispute",
                            "target_entity": "Section 15 of the Commercial Code",
                            "relationship_type": "GOVERNED_BY",
                            "context": "legal_authority"
                        }
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_business_relation_examples() -> List[Any]:
        """Business domain relation examples."""
        return [
            lx.data.ExampleData(
                text="CEO John Davis announced Q3 revenue growth and plans to expand into the European market.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="corporate_announcement",
                        extraction_text="CEO John Davis announced Q3 revenue growth",
                        attributes={
                            "source_entity": "John Davis",
                            "target_entity": "Q3 revenue growth",
                            "relationship_type": "ANNOUNCED",
                            "role": "CEO"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="business_expansion",
                        extraction_text="plans to expand into the European market",
                        attributes={
                            "source_entity": "company",
                            "target_entity": "European market",
                            "relationship_type": "PLANS_TO_EXPAND_INTO",
                            "strategy": "market_expansion"
                        }
                    ),
                ]
            )
        ]

    @staticmethod
    def _get_scientific_relation_examples() -> List[Any]:
        """Scientific domain relation examples."""
        return [
            lx.data.ExampleData(
                text="The research team at MIT published findings on quantum computing in Nature journal.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="research_publication",
                        extraction_text="research team at MIT published findings",
                        attributes={
                            "source_entity": "research team at MIT",
                            "target_entity": "findings",
                            "relationship_type": "PUBLISHED",
                            "context": "academic_research"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="research_focus",
                        extraction_text="findings on quantum computing",
                        attributes={
                            "source_entity": "findings",
                            "target_entity": "quantum computing",
                            "relationship_type": "FOCUSES_ON",
                            "field": "computer_science"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="publication_venue",
                        extraction_text="published in Nature journal",
                        attributes={
                            "source_entity": "findings",
                            "target_entity": "Nature journal",
                            "relationship_type": "PUBLISHED_IN",
                            "venue_type": "scientific_journal"
                        }
                    ),
                ]
            )
        ]


class DomainEntityTypes:
    """Predefined entity types for different domains."""

    GENERAL = {
        "person": "Human beings, individuals, professionals",
        "organization": "Companies, institutions, groups, agencies",
        "location": "Places, cities, countries, addresses, facilities",
        "event": "Meetings, conferences, incidents, processes",
        "product": "Items, services, offerings, substances, materials",
        "concept": "Ideas, theories, abstract notions, methods",
        "substance": "Chemical compounds, materials, elements",
        "condition": "States, situations, diseases, disorders",
        "process": "Procedures, methods, protocols, activities",
        "measurement": "Quantities, values, metrics, levels",
        "time": "Dates, periods, durations, schedules",
        "document": "Papers, reports, studies, publications"
    }

    MEDICAL = {
        "patient": "Person receiving medical care",
        "doctor": "Medical professional, physician",
        "condition": "Medical condition, disease, illness",
        "medication": "Pharmaceutical drug, treatment",
        "symptom": "Observable sign of disease or condition",
        "treatment": "Medical intervention, therapy",
        "medical_facility": "Hospital, clinic, medical center",
        "vital_sign": "Measurable bodily function indicator",
        "toxin": "Harmful substance, environmental toxin",
        "chemical": "Chemical compound, substance",
        "mineral": "Mineral, trace element, supplement",
        "vitamin": "Vitamin, nutrient",
        "enzyme": "Biological enzyme, protein",
        "hormone": "Hormone, endocrine substance",
        "detox_method": "Detoxification method, cleansing protocol",
        "supplement": "Dietary supplement, nutritional product",
        "heavy_metal": "Heavy metal, toxic metal",
        "pesticide": "Pesticide, agricultural chemical",
        "protocol": "Treatment protocol, therapeutic regimen",
        "organ": "Body organ, anatomical structure",
        "substance": "Chemical substance, compound"
    }

    TECHNICAL = {
        "technology": "Software, hardware, tools",
        "system": "Computer system, platform",
        "database": "Data storage system",
        "api": "Application programming interface",
        "server": "Computer server, host",
        "framework": "Software framework, library",
        "architecture_pattern": "Software design pattern",
        "platform": "Computing platform, environment"
    }

    LEGAL = {
        "court": "Legal court, tribunal",
        "judge": "Legal judge, justice",
        "lawyer": "Attorney, legal counsel",
        "statute": "Law, legal statute",
        "case": "Legal case, lawsuit",
        "contract": "Legal agreement, contract",
        "plaintiff": "Party bringing lawsuit",
        "defendant": "Party being sued"
    }

    BUSINESS = {
        "executive": "Business executive, leader",
        "company": "Business entity, corporation",
        "market": "Business market, sector",
        "financial_metric": "Revenue, profit, financial measure",
        "strategy": "Business strategy, plan",
        "investor": "Person or entity that invests",
        "product": "Business product, service",
        "competitor": "Business competitor, rival"
    }

    SCIENTIFIC = {
        "researcher": "Scientific researcher, scientist",
        "research_topic": "Area of scientific study",
        "publication": "Scientific paper, journal",
        "experiment": "Scientific experiment, study",
        "hypothesis": "Scientific hypothesis, theory",
        "methodology": "Research method, approach",
        "finding": "Research result, discovery",
        "institution": "Research institution, university"
    }


class DomainRelationTypes:
    """Predefined relation types for different domains."""

    GENERAL = {
        "works_for": "Employment relationship",
        "located_in": "Location relationship",
        "part_of": "Membership or component relationship",
        "related_to": "General association",
        "owns": "Ownership relationship",
        "manages": "Management relationship",
        "causes": "Causation relationship",
        "affects": "Influence or impact relationship",
        "contains": "Containment relationship",
        "produces": "Production or creation relationship",
        "uses": "Usage or utilization relationship",
        "interacts_with": "Interaction relationship",
        "depends_on": "Dependency relationship",
        "leads_to": "Consequence or result relationship",
        "prevents": "Prevention or blocking relationship",
        "supports": "Support or assistance relationship"
    }

    MEDICAL = {
        "treats": "Medical treatment relationship",
        "diagnosed_with": "Diagnosis relationship",
        "prescribed": "Prescription relationship",
        "causes": "Causation relationship",
        "manifests_as": "Symptom manifestation",
        "administered_at": "Location of medical care",
        "affects": "Disease or condition affecting an organ or body part",
        "damages": "Harmful effect on organ or tissue",
        "protects": "Protective effect against disease or damage",
        "detoxifies": "Removal or neutralization of toxins",
        "binds_to": "Chemical or molecular binding relationship",
        "accumulates_in": "Substance accumulation in organ or tissue",
        "depletes": "Reduction or depletion of substance",
        "supports": "Supportive or beneficial effect",
        "inhibits": "Inhibitory or blocking effect",
        "activates": "Activation or stimulation effect",
        "metabolizes": "Metabolic processing relationship",
        "excretes": "Elimination or excretion pathway",
        "absorbs": "Absorption or uptake relationship",
        "converts_to": "Chemical conversion or transformation",
        "interacts_with": "Interaction between substances or processes"
    }

    TECHNICAL = {
        "uses": "Technology usage relationship",
        "connects_to": "System connection",
        "depends_on": "Dependency relationship",
        "implements": "Implementation relationship",
        "deployed_on": "Deployment relationship",
        "integrates_with": "Integration relationship"
    }

    LEGAL = {
        "ruled_on": "Legal ruling relationship",
        "governed_by": "Legal authority relationship",
        "represents": "Legal representation",
        "sued_by": "Legal action relationship",
        "enforces": "Legal enforcement",
        "violates": "Legal violation"
    }

    BUSINESS = {
        "leads": "Leadership relationship",
        "competes_with": "Competition relationship",
        "invests_in": "Investment relationship",
        "partners_with": "Partnership relationship",
        "acquires": "Acquisition relationship",
        "serves": "Customer service relationship"
    }

    SCIENTIFIC = {
        "researches": "Research relationship",
        "publishes": "Publication relationship",
        "collaborates_with": "Research collaboration",
        "cites": "Citation relationship",
        "validates": "Validation relationship",
        "contradicts": "Contradiction relationship"
    }
