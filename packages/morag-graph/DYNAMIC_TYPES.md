# Dynamic Entity and Relation Types

The MoRAG graph extraction system now supports **complete dynamic control** over entity and relation types, allowing you to optimize extraction quality for any domain.

## Overview

Both `EntityExtractor` and `RelationExtractor` accept optional type parameters that give you full control over what types are extracted:

- **Default behavior**: If no types are specified (`None`), uses predefined defaults
- **Custom types**: If types are provided, uses EXACTLY those types (no defaults included)
- **Empty types**: Even empty dictionaries `{}` are supported for maximum control
- **Backward compatible**: Existing code continues to work unchanged

## Usage Examples

### 1. Default Types (General Purpose)
```python
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.config import LLMConfig

config = LLMConfig(provider="openai", model="gpt-4")

# Uses predefined default types
entity_extractor = EntityExtractor(config)
relation_extractor = RelationExtractor(config)
```

### 2. Domain-Specific Types
```python
# Medical domain
medical_entities = {
    "DISEASE": "Medical condition or illness",
    "TREATMENT": "Medical intervention or therapy", 
    "SYMPTOM": "Observable sign of disease",
    "MEDICATION": "Pharmaceutical drug or medicine"
}

medical_relations = {
    "CAUSES": "Pathogen causes disease",
    "TREATS": "Treatment treats condition",
    "MANIFESTS_AS": "Disease manifests as symptom"
}

entity_extractor = EntityExtractor(config, entity_types=medical_entities)
relation_extractor = RelationExtractor(config, relation_types=medical_relations)
```

### 3. Minimal Types (Highly Focused)
```python
# Extract only specific relationships
minimal_relations = {
    "CAUSES": "Direct causal relationship"
}

relation_extractor = RelationExtractor(config, relation_types=minimal_relations)
```

### 4. Maximum Control (No Predefined Types)
```python
# Let the LLM determine types dynamically
entity_extractor = EntityExtractor(config, entity_types={})
relation_extractor = RelationExtractor(config, relation_types={})
```

## Default Types Reference

### Default Entity Types
```python
DEFAULT_ENTITY_TYPES = {
    "PERSON": "Individual person mentioned in the text",
    "ORGANIZATION": "Company, institution, or group", 
    "LOCATION": "Geographic location, place, or address",
    "EVENT": "Significant occurrence or happening",
    "CONCEPT": "Abstract idea, theory, or principle",
    "OBJECT": "Physical item, product, or artifact"
}
```

### Default Relation Types
```python
DEFAULT_RELATION_TYPES = {
    "RELATED_TO": "General relationship between entities",
    "PART_OF": "Entity is a component or subset of another",
    "CAUSES": "Entity directly causes or leads to another",
    "LOCATED_IN": "Entity is physically situated within another",
    "WORKS_FOR": "Person is employed by or affiliated with organization",
    "PARTICIPATES_IN": "Entity takes part in an event or activity"
}
```

## Benefits

### 🎯 **Domain Optimization**
Tailor extraction to your specific field (medical, legal, financial, etc.) by defining relevant types.

### 🔍 **Precision Control** 
Use minimal type sets to focus on specific relationships and reduce noise.

### 🚀 **Quality Improvement**
Domain-specific types lead to more accurate and relevant extractions.

### 🔄 **Backward Compatibility**
Existing code continues to work without changes.

### ⚡ **Flexibility**
Switch between general-purpose and specialized extraction as needed.

## Implementation Details

### Type Parameter Handling
```python
# In both extractors:
self.entity_types = entity_types if entity_types is not None else self.DEFAULT_ENTITY_TYPES
self.relation_types = relation_types if relation_types is not None else self.DEFAULT_RELATION_TYPES
```

This ensures:
- `None` → Uses defaults (backward compatibility)
- `{}` → Uses no types (maximum control)
- `{"TYPE": "desc"}` → Uses exactly those types

### System Prompt Generation
The system prompts are dynamically generated based on the specified types:

```python
# Entity types are injected into the prompt
entity_types_text = "\n".join([
    f"- {type_name}: {description}" 
    for type_name, description in self.entity_types.items()
])
```

## Testing

Comprehensive tests ensure the dynamic type system works correctly:

- `test_complete_dynamic_types.py` - Verifies complete control
- `test_empty_types.py` - Tests edge cases with empty types
- `complete_dynamic_types_demo.py` - Live demonstration

Run tests:
```bash
python test_complete_dynamic_types.py
python test_empty_types.py
python examples/complete_dynamic_types_demo.py
```

## Migration Guide

### Existing Code
No changes needed - existing code continues to work:
```python
# This still works exactly as before
extractor = EntityExtractor(config)
```

### New Features
To use dynamic types, simply add the type parameters:
```python
# Add custom types
extractor = EntityExtractor(config, entity_types=my_types)
```

## Best Practices

1. **Start with defaults** for general-purpose extraction
2. **Define domain types** for specialized use cases
3. **Use minimal types** when focusing on specific relationships
4. **Test with your data** to find optimal type definitions
5. **Document your types** with clear descriptions

---

*This feature provides the flexibility to optimize graph extraction for any domain while maintaining full backward compatibility.*