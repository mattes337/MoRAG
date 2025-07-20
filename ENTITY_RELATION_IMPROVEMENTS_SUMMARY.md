# Entity and Relation Processing Improvements

## Overview

This document summarizes the comprehensive improvements made to entity normalization and relation handling in the MoRAG knowledge graph ingestion system. These changes address the core issues of entity consistency, contextual information handling, and semantic relationship richness.

**Key Philosophy**: Instead of using hardcoded language patterns, the LLM is now responsible for intelligent entity normalization across ALL languages (German, English, Spanish, French, Italian, etc.).

## Key Improvements Implemented

### 1. LLM-Based Entity Normalization

**File**: `packages/morag-graph/src/morag_graph/utils/entity_normalizer.py`

#### Philosophy:
- **LLM-driven approach**: The LLM does the heavy lifting for normalization across any language
- **No hardcoded patterns**: Eliminates brittle language-specific regex patterns
- **Universal language support**: Works with German, English, Spanish, French, Italian, and any other language
- **Intelligent context understanding**: LLM understands linguistic nuances better than regex

#### Features:
- **Universal language normalization**: LLM handles singular forms in any language
  - German: "Hunde" → "Hund", "Schwermetalle" → "Schwermetall"
  - English: "cats" → "cat", "children" → "child"
  - Spanish: "médicos" → "médico", "niños" → "niño"
  - French: "médecins" → "médecin", "enfants" → "enfant"
- **Gender normalization**: LLM converts gendered forms to canonical forms
  - German: "Pilotinnen" → "Pilot", "Ärztin" → "Arzt"
  - Spanish: "médicas" → "médico", "profesoras" → "profesor"
  - French: "médecines" → "médecin", "professeures" → "professeur"
- **Conjugation handling**: LLM normalizes conjugated forms to base forms
  - German: "Belastungen" → "Belastung"
  - English: "running" → "run", "studied" → "study"
  - Spanish: "corriendo" → "correr", "estudiado" → "estudiar"
- **Universal abbreviation handling**: Supports abbreviations in multiple languages
  - "WHO", "Weltgesundheitsorganisation", "Organización Mundial de la Salud" → "WHO"
  - "DNA", "Desoxyribonukleinsäure", "ácido desoxirribonucleico" → "DNA"
- **Context removal**: Strips positional and contextual information in any language
  - "protein in cells" → "protein"
  - "Protein bei Patienten" → "Protein"
  - "proteína en células" → "proteína"

#### Integration:
- Enhanced prompts in `EntityExtractionAgent` emphasize LLM responsibility for normalization
- Comprehensive examples across multiple languages in prompts
- Fallback sync method for basic cleanup when LLM is unavailable
- Stores original names in attributes when normalization occurs

### 2. Semantic Relation Enhancement

**File**: `packages/morag-graph/src/morag_graph/utils/semantic_relation_enhancer.py`

#### Features:
- **Domain-aware relation detection**: Automatically detects medical, technical, business, and academic contexts
- **Pattern-based relation extraction**: Uses regex patterns to identify specific relationship types
- **Rich relation categories**:
  - **Causal**: causes, prevents, enables, triggers
  - **Temporal**: precedes, follows, occurs_during
  - **Hierarchical**: contains, manages, supervises
  - **Functional**: uses, operates, depends_on
  - **Medical**: treats, diagnoses, interacts_with
  - **Technical**: implements, connects_to, processes
  - **Knowledge**: teaches, explains, demonstrates
  - **Creation**: creates, develops, originates_from

#### Integration:
- Integrated into `RelationExtractionAgent` for automatic relation type enhancement
- Validates relation types to ensure meaningful semantic relationships
- Provides domain-specific relation suggestions

### 3. Multiple Relations Between Entity Pairs

**Enhanced in**: `packages/morag-graph/src/morag_graph/ai/relation_agent.py`

#### Features:
- **Preserves multiple relation types**: Same entity pair can have different relation types
  - Example: Doctor → Patient can have "treats", "diagnoses", "monitors"
- **Smart deduplication**: Only removes exact duplicates (same source, target, and type)
- **Context merging**: Combines contexts from duplicate relations
- **Enhanced prompts**: Explicitly encourages extraction of multiple meaningful relationships

#### Benefits:
- Richer knowledge representation
- Better capture of complex real-world relationships
- Contextual information stored in relations, not entities

### 4. Improved Entity Processing

**Enhanced in**: `packages/morag-graph/src/morag_graph/ai/entity_agent.py`

#### Features:
- **Automatic normalization**: All extracted entities are normalized to canonical forms
- **Original name preservation**: Stores original entity names in attributes when changed
- **Better deduplication**: Uses normalized forms for consistent entity identification
- **Enhanced prompts**: Updated to emphasize canonical form extraction

## Technical Implementation Details

### LLM-Based Entity Normalization Pipeline

1. **Input**: Raw entity name from text
2. **Abbreviation check**: Handle universal abbreviations (WHO, DNA, etc.)
3. **Basic context removal**: Strip obvious contextual markers
4. **LLM normalization**: Send to LLM with comprehensive normalization instructions
5. **LLM processing**: LLM applies language-specific knowledge for:
   - Singular form conversion
   - Gender normalization
   - Conjugation handling
   - Context removal
   - Proper capitalization
6. **Fallback**: Basic cleanup if LLM unavailable
7. **Output**: Canonical entity name

### Enhanced Entity Extraction Prompts

The LLM receives detailed instructions for normalization across languages:
- Comprehensive examples for German, English, Spanish, French
- Clear rules for singular forms, gender normalization, conjugation handling
- Context removal instructions with multilingual examples
- Abbreviation resolution guidelines
- Emphasis that the LLM must do this work, not rely on post-processing

### Relation Enhancement Pipeline

1. **Input**: Basic relation with source, target, context
2. **Domain detection**: Analyze context and entities for domain indicators
3. **Pattern matching**: Apply domain-specific and general relation patterns
4. **Type extraction**: Extract specific relation type from context
5. **Validation**: Ensure relation type is meaningful and well-formed
6. **Output**: Enhanced semantic relation type

### Deduplication Strategy

- **Entities**: Group by normalized name (case-insensitive), merge attributes
- **Relations**: Group by (source_id, target_id, type), preserve different types, merge duplicates

## Testing

**File**: `packages/morag-graph/tests/test_enhanced_entity_relation_processing.py`

Comprehensive test suite covering:
- German and English entity normalization
- Gender and plural form handling
- Abbreviation normalization
- Context removal
- Semantic relation enhancement
- Domain detection
- Multiple relation preservation
- Duplicate relation merging

## Benefits and Impact

### 1. Entity Consistency
- **Before**: "brain", "brains", "Brain", "BRAIN" created separate entities
- **After**: All variations normalize to "brain" (single entity)

### 2. Language Support
- **Before**: German entities created inconsistent forms
- **After**: "Pilotinnen", "Pilotin" both normalize to "Pilot"

### 3. Relation Richness
- **Before**: Mostly "mentions" and "relates_to" relations
- **After**: Specific semantic relations like "treats", "causes", "implements"

### 4. Multiple Relationships
- **Before**: Only one relation type per entity pair
- **After**: Multiple meaningful relations between same entities

### 5. Context Handling
- **Before**: Contextual information mixed into entity names
- **After**: Clean entity names with context stored in relations

## Usage Examples

### LLM-Based Entity Normalization
```python
from morag_graph.utils.entity_normalizer import EntityNormalizer

normalizer = EntityNormalizer(llm_client=your_llm_client)

# Async LLM-based normalization (recommended)
# German examples
normalized = await normalizer.normalize_entity_name("Pilotinnen", "de")  # → "Pilot"
normalized = await normalizer.normalize_entity_name("Schwermetalle", "de")  # → "Schwermetall"

# English examples
normalized = await normalizer.normalize_entity_name("children", "en")  # → "child"
normalized = await normalizer.normalize_entity_name("mice", "en")  # → "mouse"

# Spanish examples
normalized = await normalizer.normalize_entity_name("médicos", "es")  # → "médico"
normalized = await normalizer.normalize_entity_name("niños", "es")  # → "niño"

# French examples
normalized = await normalizer.normalize_entity_name("médecins", "fr")  # → "médecin"
normalized = await normalizer.normalize_entity_name("enfants", "fr")  # → "enfant"

# Context removal (any language)
normalized = await normalizer.normalize_entity_name("protein in cells")  # → "protein"
normalized = await normalizer.normalize_entity_name("Protein bei Patienten")  # → "Protein"
normalized = await normalizer.normalize_entity_name("proteína en células")  # → "proteína"

# Sync fallback (basic cleanup only)
normalized = normalizer.normalize_entity_name_sync("who")  # → "WHO"
normalized = normalizer.normalize_entity_name_sync("dna")  # → "DNA"
```

### Relation Enhancement
```python
from morag_graph.utils.semantic_relation_enhancer import SemanticRelationEnhancer

enhancer = SemanticRelationEnhancer()

# Medical context
relation_type = enhancer.enhance_relation_type(
    "doctor", "patient", 
    "The doctor treats the patient with medication", 
    "relates_to"
)
# Returns: "treats"

# Technical context
relation_type = enhancer.enhance_relation_type(
    "software", "database",
    "The software connects to the database",
    "relates_to" 
)
# Returns: "connects_to"
```

## Future Enhancements

1. **Machine Learning Integration**: Train models on domain-specific entity and relation patterns
2. **Multi-language Support**: Extend to French, Spanish, Italian, etc.
3. **Custom Domain Rules**: Allow users to define domain-specific normalization rules
4. **Confidence Scoring**: Add confidence scores for normalization decisions
5. **Relation Hierarchies**: Create hierarchical relation type taxonomies

## Conclusion

These improvements significantly enhance the quality and consistency of the knowledge graph by:
- Ensuring entities are always in canonical form regardless of how they appear in text
- Creating richer, more meaningful semantic relationships
- Supporting complex real-world scenarios with multiple relation types
- Properly separating entity identity from contextual information

The result is a more robust, semantically rich knowledge graph that better captures the nuances of real-world relationships and entities.
