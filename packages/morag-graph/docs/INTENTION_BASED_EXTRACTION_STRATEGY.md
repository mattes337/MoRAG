# Intention-Based Entity and Relation Type Abstraction Strategy

## Problem Statement

The current dynamic entity and relation extraction system generates too many specific types, leading to fragmented knowledge graphs. For example:
- Organization hierarchies create separate relation types like "IS_CEO", "IS_CTO", "IS_CFO" instead of a unified "IS_MEMBER" or "IS_EMPLOYEE"
- Medical content creates overly specific entity types like "BRAIN_REGION", "CELL_TYPE", "ANATOMICAL_LOCATION" instead of a unified "ANATOMICAL"

## Solution Overview

Implement an intention-based extraction strategy that:
1. **Generates document intention summaries** to understand the document's purpose
2. **Uses intention context** to guide LLM toward more abstract, domain-appropriate types
3. **Enforces type abstraction rules** while maintaining LLM-driven classification

## Implementation Strategy

### 1. Document Intention Analysis

**Process:**
- Generate a concise intention summary of the entire document before entity/relation extraction
- Use existing `summarize_document` functionality with intention-focused prompts
- Examples of intention summaries:
  - Medical content: "Heal the pineal gland for spiritual enlightenment"
  - Organizational docs: "Document explaining the structure of the organization/company"
  - Technical docs: "Guide for implementing software architecture patterns"

**Integration Point:**
- Add intention analysis step in `GraphProcessor.process_document()` before entity extraction
- Store intention summary in document metadata for consistent use across chunks

### 2. Enhanced LLM Prompts

**Entity Type Abstraction:**
- Instruct LLM to use broader, more abstract entity types
- Provide intention-specific guidance for type selection
- Enforce singular form for all entity types
- Examples:
  - `BRAIN_REGION`, `CELL_TYPE`, `ANATOMICAL_LOCATION` → `ANATOMICAL`
  - `SOFTWARE_LIBRARY`, `PROGRAMMING_LANGUAGE`, `FRAMEWORK` → `TECHNOLOGY`
  - `CEO`, `CTO`, `MANAGER` → `PERSON`

**Relation Type Abstraction:**
- Guide LLM toward coarser relationship categories
- Use intention summary to determine appropriate abstraction level
- Examples:
  - `IS_CEO`, `IS_CTO`, `IS_CFO` → `IS_MEMBER` or `IS_EMPLOYEE`
  - `CAUSES_INFLAMMATION`, `TRIGGERS_RESPONSE` → `AFFECTS`
  - `IMPLEMENTS_INTERFACE`, `EXTENDS_CLASS` → `USES`

### 3. Intention-Specific Type Mapping

**Domain-Aware Abstraction:**
- Medical/Health: Focus on anatomical structures, conditions, treatments
- Organizational: Emphasize roles, departments, hierarchies
- Technical: Concentrate on systems, processes, technologies
- Educational: Highlight concepts, methods, subjects

**Abstraction Rules:**
- Entity types should be broad semantic categories (max 10-15 types per domain)
- Relation types should capture fundamental relationship patterns (max 8-12 types per domain)
- Maintain semantic meaning while reducing granularity

## Implementation Plan

### Phase 1: Document Intention Analysis
1. Add `generate_document_intention()` method to `GraphProcessor`
2. Integrate intention generation into processing pipeline
3. Store intention in document metadata

### Phase 2: Enhanced Prompts
1. Modify entity extractor prompts to include intention context
2. Add abstraction guidelines based on intention
3. Update relation extractor with intention-aware type selection

### Phase 3: Type Validation and Normalization
1. Implement intention-based type validation
2. Add post-processing normalization based on intention
3. Create intention-specific type mapping rules

### Phase 4: Testing and Refinement
1. Test with various document types and intentions
2. Validate type reduction and semantic preservation
3. Refine abstraction rules based on results

## Expected Benefits

1. **Reduced Type Fragmentation:** Fewer, more meaningful entity and relation types
2. **Better Graph Connectivity:** More abstract types create stronger connections
3. **Domain Awareness:** Intention-driven abstraction respects document context
4. **Maintained Flexibility:** LLM still determines types, but with better guidance
5. **Improved Query Performance:** Fewer types make graph traversal more efficient

## Configuration Options

- `enable_intention_analysis`: Toggle intention-based processing
- `intention_max_length`: Maximum length for intention summaries
- `abstraction_level`: Control how abstract types should be (low/medium/high)
- `domain_specific_rules`: Enable domain-specific type mapping

## Backward Compatibility

- Existing extraction methods remain unchanged
- Intention analysis is additive, not replacing current functionality
- Configuration flags allow gradual adoption
- Legacy type systems continue to work alongside new approach
