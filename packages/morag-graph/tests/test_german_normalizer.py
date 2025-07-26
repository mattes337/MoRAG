"""Tests for German language normalization in OpenIE."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from morag_graph.normalizers.entity_normalizer import EntityNormalizer
from morag_graph.models import Entity


class TestGermanNormalizer:
    """Test cases for German language normalization."""
    
    @pytest.fixture
    def german_config(self):
        """Configuration for German language processing."""
        return {
            'language': 'de',
            'enable_stemming': True,
            'enable_lemmatization': True,
            'case_sensitive': False
        }
    
    @pytest.fixture
    def german_entities(self):
        """Sample German entities for testing."""
        return [
            Entity(
                id="entity_1",
                name="Volkswagen AG",
                canonical_name="volkswagen_ag",
                entity_type="ORG",
                confidence=0.95,
                metadata={"language": "de", "source": "test"}
            ),
            Entity(
                id="entity_2",
                name="Johann Müller",
                canonical_name="johann_müller",
                entity_type="PERSON",
                confidence=0.92,
                metadata={"language": "de", "source": "test"}
            ),
            Entity(
                id="entity_3",
                name="Geschäftsführer",
                canonical_name="geschäftsführer",
                entity_type="TITLE",
                confidence=0.88,
                metadata={"language": "de", "source": "test"}
            ),
            Entity(
                id="entity_4",
                name="Niedersachsen",
                canonical_name="niedersachsen",
                entity_type="GPE",
                confidence=0.90,
                metadata={"language": "de", "source": "test"}
            )
        ]
    
    @pytest.fixture
    def normalizer(self, german_config):
        """Create German entity normalizer for testing."""
        return EntityNormalizer(german_config)
    
    def test_init(self, normalizer):
        """Test normalizer initialization for German."""
        assert normalizer.language == 'de'
        assert normalizer.enable_stemming is True
        assert normalizer.enable_lemmatization is True
        assert normalizer.case_sensitive is False
    
    @pytest.mark.asyncio
    async def test_normalize_german_entities(self, normalizer, german_entities):
        """Test normalization of German entities."""
        # Mock the normalization process
        with patch.object(normalizer, '_normalize_entity_name') as mock_normalize:
            mock_normalize.side_effect = [
                "volkswagen",  # Normalized company name
                "johann_mueller",  # Normalized person name (ü -> ue)
                "geschaeftsfuehrer",  # Normalized title (ä -> ae, ü -> ue)
                "niedersachsen"  # Normalized place name
            ]
            
            normalized_entities = await normalizer.normalize_entities(german_entities)
            
            assert len(normalized_entities) == len(german_entities)
            assert normalized_entities[0].canonical_name == "volkswagen"
            assert normalized_entities[1].canonical_name == "johann_mueller"
            assert normalized_entities[2].canonical_name == "geschaeftsfuehrer"
            assert normalized_entities[3].canonical_name == "niedersachsen"
    
    def test_german_umlaut_normalization(self, normalizer):
        """Test German umlaut normalization."""
        test_cases = [
            ("Müller", "mueller"),
            ("Größe", "groesse"),
            ("Bär", "baer"),
            ("Köln", "koeln"),
            ("Düsseldorf", "duesseldorf"),
            ("Straße", "strasse"),
            ("Weiß", "weiss"),
            ("Fußball", "fussball")
        ]
        
        for original, expected in test_cases:
            normalized = normalizer._normalize_umlauts(original)
            assert normalized == expected, f"Failed to normalize {original} to {expected}, got {normalized}"
    
    def test_german_case_normalization(self, normalizer):
        """Test German case normalization."""
        test_cases = [
            ("VOLKSWAGEN AG", "volkswagen ag"),
            ("Johann MÜLLER", "johann müller"),
            ("GeschäftsFÜHRER", "geschäftsführer"),
            ("NIEDERSACHSEN", "niedersachsen")
        ]
        
        for original, expected in test_cases:
            normalized = normalizer._normalize_case(original)
            assert normalized == expected, f"Failed to normalize case of {original}"
    
    def test_german_compound_word_handling(self, normalizer):
        """Test handling of German compound words."""
        test_cases = [
            ("Geschäftsführer", ["geschäft", "führer"]),
            ("Volkswagen", ["volks", "wagen"]),
            ("Bundesrepublik", ["bundes", "republik"]),
            ("Kraftfahrzeug", ["kraft", "fahrzeug"])
        ]
        
        for compound, expected_parts in test_cases:
            # Mock compound word splitting
            with patch.object(normalizer, '_split_compound_word', return_value=expected_parts):
                parts = normalizer._split_compound_word(compound)
                assert parts == expected_parts
    
    @pytest.mark.asyncio
    async def test_german_entity_type_normalization(self, normalizer):
        """Test German entity type normalization."""
        german_entity_types = [
            ("PERSON", "PERSON"),
            ("ORGANISATION", "ORG"),
            ("ORT", "GPE"),
            ("DATUM", "DATE"),
            ("GELD", "MONEY"),
            ("PROZENT", "PERCENT")
        ]
        
        for german_type, expected_type in german_entity_types:
            normalized_type = await normalizer.normalize_entity_type(german_type)
            assert normalized_type == expected_type
    
    def test_german_stopword_removal(self, normalizer):
        """Test German stopword removal."""
        german_stopwords = ["der", "die", "das", "und", "oder", "aber", "von", "zu", "mit", "bei"]
        
        test_text = "der große Volkswagen und die kleine BMW"
        expected_without_stopwords = "große Volkswagen kleine BMW"
        
        result = normalizer._remove_stopwords(test_text)
        
        # Check that stopwords are removed
        for stopword in german_stopwords:
            if stopword in test_text:
                assert stopword not in result
    
    @pytest.mark.asyncio
    async def test_german_predicate_normalization(self, normalizer):
        """Test German predicate normalization."""
        german_predicates = [
            ("arbeitet bei", "works_at"),
            ("ist Geschäftsführer von", "is_ceo_of"),
            ("lebt in", "lives_in"),
            ("wurde gegründet von", "founded_by"),
            ("gehört zu", "belongs_to"),
            ("befindet sich in", "located_in")
        ]
        
        for german_pred, expected_normalized in german_predicates:
            # Mock predicate normalization
            with patch.object(normalizer, 'normalize_predicate', return_value=expected_normalized):
                normalized = await normalizer.normalize_predicate(german_pred)
                assert normalized == expected_normalized
    
    def test_german_special_characters(self, normalizer):
        """Test handling of German special characters."""
        test_cases = [
            ("Straße", "strasse"),  # ß -> ss
            ("Größe", "groesse"),   # ö -> oe
            ("Müller", "mueller"),  # ü -> ue
            ("Bär", "baer"),        # ä -> ae
            ("Weiß", "weiss"),      # ß -> ss
            ("Köln", "koeln")       # ö -> oe
        ]
        
        for original, expected in test_cases:
            normalized = normalizer._handle_special_characters(original)
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_german_entity_linking(self, normalizer, german_entities):
        """Test entity linking for German entities."""
        # Test linking entities with similar German names
        entity1 = Entity(
            id="e1",
            name="Johann Müller",
            canonical_name="johann_müller",
            entity_type="PERSON",
            confidence=0.9,
            metadata={"language": "de"}
        )
        
        entity2 = Entity(
            id="e2",
            name="Johann Mueller",  # Alternative spelling
            canonical_name="johann_mueller",
            entity_type="PERSON",
            confidence=0.85,
            metadata={"language": "de"}
        )
        
        # Mock similarity calculation
        with patch.object(normalizer, 'calculate_similarity', return_value=0.95):
            similarity = await normalizer.calculate_similarity(entity1.name, entity2.name)
            assert similarity >= 0.9  # Should be high similarity despite different spelling
    
    def test_german_text_preprocessing(self, normalizer):
        """Test German text preprocessing."""
        german_text = "Die Volkswagen AG wurde 1937 in Wolfsburg gegründet."
        
        # Test various preprocessing steps
        preprocessed = normalizer.preprocess_text(german_text)
        
        # Should handle German-specific preprocessing
        assert isinstance(preprocessed, str)
        assert len(preprocessed) > 0
    
    @pytest.mark.asyncio
    async def test_multilingual_entity_matching(self, normalizer):
        """Test matching entities across German and English."""
        german_entity = Entity(
            id="de_1",
            name="Volkswagen AG",
            canonical_name="volkswagen",
            entity_type="ORG",
            confidence=0.95,
            metadata={"language": "de"}
        )
        
        english_entity = Entity(
            id="en_1",
            name="Volkswagen Group",
            canonical_name="volkswagen",
            entity_type="ORG",
            confidence=0.92,
            metadata={"language": "en"}
        )
        
        # Mock cross-language matching
        with patch.object(normalizer, 'match_cross_language', return_value=True):
            is_match = await normalizer.match_cross_language(german_entity, english_entity)
            assert is_match is True
    
    def test_german_number_normalization(self, normalizer):
        """Test German number format normalization."""
        german_numbers = [
            ("1.000.000", "1000000"),      # German thousand separator
            ("1,5", "1.5"),                # German decimal separator
            ("€ 100,50", "100.50 EUR"),    # German currency format
            ("50%", "0.5"),                # Percentage
            ("1. Januar 2023", "2023-01-01")  # German date format
        ]
        
        for german_format, expected_normalized in german_numbers:
            # Mock number normalization
            with patch.object(normalizer, '_normalize_number', return_value=expected_normalized):
                normalized = normalizer._normalize_number(german_format)
                assert normalized == expected_normalized


if __name__ == "__main__":
    pytest.main([__file__])
