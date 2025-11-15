import pytest
from app.ml.services.scoring_service import ScoringService, ErrorDetail

class TestScoringService:
    """Test cases for Scoring Service."""
    
    @pytest.fixture
    def scoring_service(self):
        """Create scoring service instance for testing."""
        return ScoringService()
    
    def test_calculate_gop_score_perfect_match(self, scoring_service):
        """Test scoring with perfect phoneme match."""
        expected = ["t", "e", "s", "t"]
        actual = ["t", "e", "s", "t"]
        confidence = [1.0, 1.0, 1.0, 1.0]
        
        result = scoring_service.calculate_gop_sync(expected, actual, confidence)
        
        assert result.overall_score == 100.0
        assert len(result.errors) == 0
        assert result.dimensions["accuracy"] == 100.0
    
    def test_calculate_gop_score_with_substitutions(self, scoring_service):
        """Test scoring with phoneme substitutions."""
        expected = ["t", "e", "s", "t"]
        actual = ["d", "e", "s", "d"]  # t -> d substitutions
        confidence = [0.8, 1.0, 1.0, 0.8]
        
        result = scoring_service.calculate_gop_sync(expected, actual, confidence)
        
        assert result.overall_score < 100.0
        assert len(result.errors) == 2
        assert all(error.type == "substitution" for error in result.errors)
        
        # Check specific errors
        t_d_errors = [e for e in result.errors if e.expected == "t" and e.actual == "d"]
        assert len(t_d_errors) == 2
    
    def test_calculate_gop_score_with_deletions(self, scoring_service):
        """Test scoring with phoneme deletions."""
        expected = ["t", "e", "s", "t"]
        actual = ["t", "s", "t"]  # 'e' deleted
        confidence = [1.0, 0.8, 0.8]
        
        result = scoring_service.calculate_gop_sync(expected, actual, confidence)
        
        assert result.overall_score < 100.0
        deletion_errors = [e for e in result.errors if e.type == "deletion"]
        assert len(deletion_errors) >= 1
    
    def test_phoneme_similarity_high_similarity(self, scoring_service):
        """Test phoneme similarity for similar phonemes."""
        # Alveolar stops - high similarity
        similarity = scoring_service._calculate_phoneme_similarity("t", "d")
        assert similarity == 0.8
        
        # Velar stops - high similarity  
        similarity = scoring_service._calculate_phoneme_similarity("k", "g")
        assert similarity == 0.8
    
    def test_phoneme_similarity_low_similarity(self, scoring_service):
        """Test phoneme similarity for different phonemes."""
        similarity = scoring_service._calculate_phoneme_similarity("t", "k")
        assert similarity == 0.3  # Default low similarity
    
    def test_phoneme_similarity_vowels(self, scoring_service):
        """Test vowel similarity."""
        # Similar vowels
        similarity = scoring_service._calculate_phoneme_similarity("i", "ɪ")
        assert similarity == 0.9
    
    def test_error_classification(self, scoring_service):
        """Test error type classification."""
        # Substitution
        error = scoring_service._classify_error("t", "d", 0, 80.0)
        assert error.type == "substitution"
        assert error.expected == "t"
        assert error.actual == "d"
        
        # Deletion
        error = scoring_service._classify_error("t", "", 0, 50.0)
        assert error.type == "deletion"
        assert error.expected == "t"
        assert error.actual == ""
        
        # Insertion
        error = scoring_service._classify_error("", "t", 0, 50.0)
        assert error.type == "insertion"
        assert error.expected == ""
        assert error.actual == "t"
    
    def test_calculate_overall_score(self, scoring_service):
        """Test overall score calculation."""
        # High scores, few errors
        phoneme_scores = [95.0, 90.0, 85.0, 92.0]
        errors = [ErrorDetail("substitution", "t", "d", 1, 0.8)]
        
        score = scoring_service._calculate_overall_score(phoneme_scores, errors)
        assert score == pytest.approx(90.5, rel=1e-1)  # 90.5 - 5 (1 error)
        
        # Many errors - should hit max penalty
        many_errors = [ErrorDetail("substitution", "a", "b", i, 0.8) for i in range(10)]
        score = scoring_service._calculate_overall_score(phoneme_scores, many_errors)
        assert score == pytest.approx(65.5, rel=1e-1)  # 90.5 - 25 (max penalty)
    
    def test_calculate_dimension_scores(self, scoring_service):
        """Test dimension score calculation."""
        phoneme_scores = [80.0, 85.0, 90.0, 75.0]
        errors = [ErrorDetail("substitution", "t", "d", 0, 0.8)]
        expected_phonemes = ["t", "e", "s", "t"]
        
        dimensions = scoring_service._calculate_dimension_scores(
            phoneme_scores, errors, expected_phonemes
        )
        
        assert "accuracy" in dimensions
        assert "fluency" in dimensions
        assert "prosody" in dimensions
        assert "stress" in dimensions
        
        # All scores should be in valid range
        for score in dimensions.values():
            assert 0.0 <= score <= 100.0
    
    def test_empty_inputs(self, scoring_service):
        """Test handling of empty inputs."""
        result = scoring_service.calculate_gop_sync([], [], [])
        
        assert result.overall_score == 50.0  # Default score
        assert len(result.errors) == 0
        assert result.confidence == 0.0
    
    def test_missing_confidence_scores(self, scoring_service):
        """Test scoring when confidence scores are not provided."""
        result = scoring_service.calculate_gop_sync(
            ["t", "e", "s", "t"],
            ["t", "e", "s", "t"],
            None  # No confidence scores
        )
        
        assert result.overall_score == 100.0
        assert result.confidence == 0.85  # Default confidence
    
    # Helper method for synchronous testing
    async def calculate_gop_sync(self, expected, actual, confidence):
        """Synchronous wrapper for testing."""
        return await self.calculate_gop_score(expected, actual, confidence)

# Add to ScoringService class for testing
def calculate_gop_sync(self, expected, actual, confidence):
    """Synchronous version for testing."""
    import asyncio
    return asyncio.run(self.calculate_gop_score(expected, actual, confidence))

# Monkey patch for testing
ScoringService.calculate_gop_sync = calculate_gop_sync
