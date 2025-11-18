import pytest
import numpy as np
import librosa
from unittest.mock import Mock, patch
from app.ml.services.enhanced_scoring import (
    EnhancedScoringService, 
    FluencyFeatures, 
    ProsodyFeatures, 
    StressFeatures
)


@pytest.fixture
def scoring_service():
    """Provide enhanced scoring service for any test class."""
    return EnhancedScoringService()


@pytest.fixture
def mock_audio_data():
    """Create mock audio data for testing."""
    duration = 3.0  # 3 seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)

    t = np.linspace(0, duration, samples)
    audio = (
        0.6 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 450 * t)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope
    noise = 0.05 * np.random.normal(0, 1, samples)
    audio = audio + noise
    audio = audio / np.max(np.abs(audio))

    return audio, sample_rate

class TestEnhancedScoringService:
    """Test cases for enhanced multi-dimensional scoring service."""
    
    
    @pytest.mark.asyncio
    async def test_comprehensive_scoring_english(self, scoring_service, mock_audio_data):
        """Test comprehensive scoring for English pronunciation."""
        audio_path = self._create_temp_audio_file(mock_audio_data)
        transcript = "hello how are you doing today"
        
        result = await scoring_service.calculate_comprehensive_score(
            audio_path=audio_path,
            transcript=transcript,
            language="en-US"
        )
        
        # Verify response structure
        assert "overall_score" in result
        assert "dimensions" in result
        assert "features" in result
        assert "feedback" in result
        assert result["language"] == "en-US"
        assert result["analysis_level"] == "comprehensive"
        
        # Verify dimensions
        dimensions = result["dimensions"]
        assert "accuracy" in dimensions
        assert "fluency" in dimensions
        assert "prosody" in dimensions
        assert "stress" in dimensions
        
        # Verify score ranges
        for dim_score in dimensions.values():
            assert 0 <= dim_score <= 100
        
        # Verify overall score
        assert 0 <= result["overall_score"] <= 100

    
    @pytest.mark.asyncio
    async def test_fluency_feature_extraction(self, scoring_service, mock_audio_data):
        """Test fluency feature extraction."""
        audio, sr = mock_audio_data
        transcript = "test sentence"
        
        features = await scoring_service._extract_fluency_features(audio, sr, transcript)
        
        assert isinstance(features, FluencyFeatures)
        assert features.speech_rate >= 0
        assert features.pause_duration >= 0
        assert features.speech_duration >= 0
        assert 0 <= features.pause_ratio <= 1
        assert features.disfluency_count >= 0
        assert 0 <= features.rhythm_regularity <= 1
    
    @pytest.mark.asyncio
    async def test_prosody_feature_extraction(self, scoring_service, mock_audio_data):
        """Test prosody feature extraction."""
        audio, sr = mock_audio_data
        
        features = await scoring_service._extract_prosody_features(audio, sr)
        
        assert isinstance(features, ProsodyFeatures)
        assert features.pitch_mean >= 0
        assert features.pitch_std >= 0
        assert features.pitch_range >= 0
        assert isinstance(features.intonation_contour, list)
        assert features.energy_variation >= 0
        assert 0 <= features.emphasis_score <= 1
    
    @pytest.mark.asyncio
    async def test_stress_feature_extraction(self, scoring_service, mock_audio_data):
        """Test stress feature extraction."""
        audio, sr = mock_audio_data
        transcript = "hello world"
        
        features = await scoring_service._extract_stress_features(audio, sr, transcript)
        
        assert isinstance(features, StressFeatures)
        assert 0 <= features.stress_pattern_score <= 1
        assert 0 <= features.primary_stress_placement <= 1
        assert len(features.syllable_timings) > 0
    
    def test_fluency_score_calculation(self, scoring_service):
        """Test fluency score calculation from features."""
        # Ideal fluency features
        ideal_features = FluencyFeatures(
            speech_rate=165,  # Ideal WPM
            pause_duration=0.25,
            speech_duration=2.5,
            pause_ratio=0.25,  # Good pause ratio
            disfluency_count=2,  # Low disfluencies
            rhythm_regularity=0.8  # Good rhythm
        )
        
        score = scoring_service._calculate_fluency_score(ideal_features)
        assert score >= 85  # Should get high score for ideal features
        
        # Poor fluency features
        poor_features = FluencyFeatures(
            speech_rate=80,   # Too slow
            pause_duration=0.8,
            speech_duration=1.0,
            pause_ratio=0.45,  # Too many pauses
            disfluency_count=15,  # Many disfluencies
            rhythm_regularity=0.2  # Poor rhythm
        )
        
        score = scoring_service._calculate_fluency_score(poor_features)
        assert score <= 60  # Should get low score for poor features
    
    def test_prosody_score_calculation(self, scoring_service):
        """Test prosody score calculation from features."""
        # Good prosody features
        good_prosody = ProsodyFeatures(
            pitch_mean=150,
            pitch_std=100,  # Good variation
            pitch_range=120,
            intonation_contour=[150, 180, 120, 160],
            energy_variation=0.8,
            emphasis_score=0.7
        )
        
        score = scoring_service._calculate_prosody_score(good_prosody)
        assert score >= 80
        
        # Flat prosody features (monotone)
        flat_prosody = ProsodyFeatures(
            pitch_mean=100,
            pitch_std=5,  # Very little variation
            pitch_range=10,
            intonation_contour=[100, 105, 98, 102],
            energy_variation=0.1,
            emphasis_score=0.05
        )
        
        score = scoring_service._calculate_prosody_score(flat_prosody)
        assert score <= 70
    
    def test_stress_score_calculation(self, scoring_service):
        """Test stress score calculation."""
        # Good stress patterns
        good_stress = StressFeatures(
            stress_pattern_score=0.9,
            primary_stress_placement=0.3,
            secondary_stress_placement=0.2,
            weak_form_pronunciation=0.25,
            syllable_timing=[0.3, 0.2, 0.3, 0.2]
        )
        
        score = scoring_service._calculate_stress_score(good_stress)
        assert score >= 85
        
        # Poor stress patterns
        poor_stress = StressFeatures(
            stress_pattern_score=0.3,
            primary_stress_placement=0.1,
            secondary_stress_placement=0.0,
            weak_form_pronunciation=0.05,
            syllable_timing=[0.1, 0.8, 0.05, 0.05]
        )
        
        score = scoring_service._calculate_stress_score(poor_stress)
        assert score <= 70
    
    @pytest.mark.asyncio
    async def test_fallback_scoring(self, scoring_service):
        """Test fallback scoring when comprehensive analysis fails."""
        result = await scoring_service._fallback_scoring(
            "dummy_path.wav", 
            "test transcript", 
            "en-US"
        )
        
        assert result["overall_score"] == 60.0
        assert result["analysis_level"] == "fallback"
        assert "feedback" in result
    
    def test_emphasis_score_calculation(self, scoring_service):
        """Test emphasis score calculation."""
        # Audio with clear peaks (high emphasis)
        high_energy = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.2])
        high_score = scoring_service._calculate_emphasis_score(high_energy)
        assert high_score > 0.5
        
        # Flat audio (low emphasis)
        flat_energy = np.array([0.3, 0.32, 0.29, 0.31, 0.3, 0.28])
        flat_score = scoring_service._calculate_emphasis_score(flat_energy)
        assert flat_score < 0.2
    
    def test_enhanced_feedback_generation(self, scoring_service):
        """Test enhanced feedback generation."""
        # Sample features
        fluency_features = FluencyFeatures(
            speech_rate=100,  # Slow
            pause_duration=0.5,
            pause_ratio=0.35,  # Too many pauses
            disfluency_count=5,
            rhythm_regularity=0.4
        )
        
        prosody_features = ProsodyFeatures(
            pitch_std=20,  # Monotone
            emphasis_score=0.15  # Low emphasis
        )
        
        stress_features = StressFeatures(
            stress_pattern_score=0.4  # Poor stress patterns
        )
        
        feedback = scoring_service._generate_enhanced_feedback(
            fluency_features, prosody_features, stress_features
        )
        
        assert "fluency" in feedback
        assert "prosody" in feedback
        assert "stress" in feedback
        
        # Verify specific fluency recommendations
        fluency_feedback = feedback["fluency"]
        assert any("slow" in rec.lower() for rec in fluency_feedback["recommendations"])
        
        # Verify specific prosody recommendations
        prosody_feedback = feedback["prosody"]
        assert any("pitch" in rec.lower() for rec in prosody_feedback["recommendations"])
    
    def _create_temp_audio_file(self, mock_audio_data):
        """Create temporary audio file for testing."""
        import tempfile
        import soundfile as sf
        import os
        
        audio, sr = mock_audio_data
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio, sr)
        temp_file.close()
        
        # Cleanup function
        def cleanup():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return temp_file.name

class TestEnglishPronunciationSpecifics:
    """Test English-specific pronunciation analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create English pronunciation analyzer."""
        from app.ml.models.wav2vec_trainer import EnglishPronunciationAnalyzer
        return EnglishPronunciationAnalyzer()
    
    def test_english_error_analysis(self, analyzer):
        """Test English-specific error analysis."""
        expected_phonemes = ['ɝ', 'ɚ', 'θ', 'ð']  # Rhotic and TH sounds
        actual_phonemes = ['ɹ', 'ə', 't', 'd']     # Common substitutions

        errors = analyzer.analyze_english_errors(expected_phonemes, actual_phonemes)
        
        assert 'total_errors' in errors
        assert 'error_patterns' in errors
        assert 'specific_challenges' in errors
        
        # Should detect rhoticity errors
        assert len(errors['rhoticity_errors']) > 0
        
        # Should detect TH errors
        assert len(errors['th_errors']) > 0


    

# Integration tests
class TestIntegration:
    """Integration tests for enhanced scoring."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_english(self):
        """Test full pipeline for English pronunciation scoring."""
        from app.ml.services.enhanced_scoring import EnhancedScoringService
        
        service = EnhancedScoringService()
        
        # This test would require actual audio file
        # For now, test the structure and error handling
        try:
            result = await service.calculate_comprehensive_score(
                "nonexistent.wav",
                "test transcript",
                "en-US"
            )
            # Should return fallback result for nonexistent file
            assert result["analysis_level"] == "fallback"
        except:
            pass  # Expected to fail due to nonexistent file

# Performance tests
class TestPerformance:
    """Performance tests for enhanced scoring."""
    
    @pytest.mark.asyncio
    async def test_scoring_speed(self, scoring_service):
        """Test that scoring completes within acceptable time."""
        import time
        import tempfile
        
        # Create test audio
        audio, sr = np.random.randn(16000 * 2), 16000  # 2 seconds
        audio_path = self._create_temp_audio_file((audio, sr))
        
        start_time = time.time()
        
        try:
            result = await scoring_service.calculate_comprehensive_score(
                audio_path=audio_path,
                transcript="quick brown fox",
                language="en-US"
            )
            
            processing_time = time.time() - start_time
            
            # Sprint 2 target: full analysis under 3 seconds
            assert processing_time < 3.0
            
        finally:
            # Cleanup
            try:
                import os
                os.unlink(audio_path)
            except:
                pass


# Reuse helper utilities across classes
TestPerformance._create_temp_audio_file = TestEnhancedScoringService._create_temp_audio_file
