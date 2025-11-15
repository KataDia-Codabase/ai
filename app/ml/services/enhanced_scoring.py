"""
Enhanced multi-dimensional scoring service for English pronunciation.
Implements accuracy, fluency, prosody, and stress analysis.
"""

import torch
import numpy as np
import librosa
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from app.ml.services.stt_service import STTService
from app.ml.services.scoring_service import ScoringResult, ErrorDetail
from app.ml.models.wav2vec_trainer import EnglishPronunciationAnalyzer
import structlog
from pathlib import Path

logger = structlog.get_logger()

@dataclass
class FluencyFeatures:
    """Fluency analysis features."""
    speech_rate: float            # words per minute
    pause_duration: float         # average pause duration
    speech_duration: float        # total speaking duration
    pause_ratio: float           # pause_time / total_time
    disfluency_count: int         # number of disfluencies
    rhythm_regularity: float      # timing consistency score

@dataclass 
class ProsodyFeatures:
    """Prosody analysis features."""
    pitch_mean: float            # average pitch
    pitch_std: float             # pitch variation
    pitch_range: float           # overall pitch range
    intonation_contour: List[float]  # pitch over time
    energy_variation: float      # energy level changes
    emphasis_score: float        # stress/emphasis accuracy

@dataclass
class StressFeatures:
    """Word stress analysis features."""
    stress_pattern_score: float   # stress accuracy score
    primary_stress_placement: float
    secondary_stress_placement: float
    weak_form_pronunciation: float
    syllable_timing: List[float]

class EnhancedScoringService:
    """Enhanced multi-dimensional scoring for English pronunciation."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.analyzer = EnglishPronunciationAnalyzer()
        
        # Model placeholders - would be loaded from trained models
        self.fluency_model = None  # Load trained fluency model
        self.prosody_model = None  # Load trained prosody model
        self.stress_model = None   # Load trained stress model
    
    async def calculate_comprehensive_score(
        self,
        audio_path: str,
        transcript: str,
        language: str = "en-US"
    ) -> Dict:
        """
        Calculate comprehensive pronunciation score across multiple dimensions.
        
        Args:
            audio_path: Path to audio file
            transcript: Reference transcript
            language: Language code (focus on en-US)
            
        Returns:
            Comprehensive scoring results with all dimensions
        """
        try:
            logger.info(f"Starting comprehensive scoring for {language}")
            
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            y = librosa.util.normalize(y)
            
            # Extract features
            fluency_features = await self._extract_fluency_features(y, sr, transcript)
            prosody_features = await self._extract_prosody_features(y, sr)
            stress_features = await self._extract_stress_features(y, sr, transcript)
            
            # Calculate scores for each dimension
            accuracy_score = await self._calculate_accuracy_score(audio_path, transcript)
            fluency_score = await self._calculate_fluency_score(fluency_features)
            prosody_score = await self._calculate_prosody_score(prosody_features)
            stress_score = await self._calculate_stress_score(stress_features)
            
            # Combine scores with weights
            weights = {"accuracy": 0.4, "fluency": 0.2, "prosody": 0.2, "stress": 0.2}
            overall_score = (
                accuracy_score * weights["accuracy"] +
                fluency_score * weights["fluency"] +
                prosody_score * weights["prosody"] +
                stress_score * weights["stress"]
            )
            
            # Generate detailed feedback
            feedback = await self._generate_enhanced_feedback(
                fluency_features, prosody_features, stress_features
            )
            
            return {
                "overall_score": overall_score,
                "dimensions": {
                    "accuracy": accuracy_score,
                    "fluency": fluency_score,
                    "prosody": prosody_score,
                    "stress": stress_score
                },
                "features": {
                    "fluency": fluency_features,
                    "prosody": prosody_features,
                    "stress": stress_features
                },
                "feedback": feedback,
                "language": language,
                "analysis_level": "comprehensive"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive scoring failed: {e}")
            # Return fallback scoring
            return await self._fallback_scoring(audio_path, transcript, language)
    
    async def _extract_fluency_features(self, y: np.ndarray, sr: int, transcript: str) -> FluencyFeatures:
        """Extract fluency-related features from audio."""
        try:
            # Detect speech/pause segments
            intervals = librosa.effects.split(y, top_db=25)
            
            speech_segments = []
            pause_segments = []
            
            # Identify speech and pause intervals
            for i in range(len(intervals) - 1):
                start, end = intervals[i]
                next_start, _ = intervals[i + 1]
                
                # Current segment is speech
                speech_duration = (end - start) / sr
                speech_segments.append(speech_duration)
                
                # Gap to next segment is pause
                pause_duration = (next_start - end) / sr
                if pause_duration > 0.1:  # Minimum pause threshold
                    pause_segments.append(pause_duration)
            
            # Calculate fluency metrics
            total_speech_time = sum(speech_segments)
            total_pause_time = sum(pause_segments)
            total_duration = total_speech_time + total_pause_time
            
            # Speech rate (words per minute)
            word_count = len(transcript.split())
            speech_rate = (word_count / total_speech_time) * 60 if total_speech_time > 0 else 0
            
            # Pause characteristics
            avg_pause_duration = np.mean(pause_segments) if pause_segments else 0
            pause_ratio = total_pause_time / total_duration if total_duration > 0 else 0
            
            # Rhythm regularity (consistency of speech segments)
            rhythm_regularity = 1.0 - (np.std(speech_segments) / np.mean(speech_segments)) if speech_segments else 0
            
            # Count disfluencies (filled pauses, repetitions, etc.)
            # This is a simplified detection - would use ML classifier in production
            energy = librosa.feature.rms(y=y)[0]
            energy_changes = np.diff(energy)
            disfluency_count = len(np.where(np.abs(energy_changes) > np.percentile(np.abs(energy_changes), 90))[0])
            disfluency_count = int(disfluency_count * 0.1)  # Scale down
            
            return FluencyFeatures(
                speech_rate=float(speech_rate),
                pause_duration=float(avg_pause_duration),
                speech_duration=float(total_speech_time),
                pause_ratio=float(pause_ratio),
                disfluency_count=int(disfluency_count),
                rhythm_regularity=float(rhythm_regularity)
            )
            
        except Exception as e:
            logger.error(f"Fluency feature extraction failed: {e}")
            return FluencyFeatures(0, 0, 0, 0, 0, 0)
    
    async def _extract_prosody_features(self, y: np.ndarray, sr: int) -> ProsodyFeatures:
        """Extract prosody-related features from audio."""
        try:
            # Pitch extraction
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get fundamental frequency (F0)
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                f0_values.append(pitch)
            
            f0_values = np.array([p for p in f0_values if p > 0])  # Remove zeros
            
            if len(f0_values) > 0:
                pitch_mean = np.mean(f0_values)
                pitch_std = np.std(f0_values)
                pitch_range = np.max(f0_values) - np.min(f0_values)
                intonation_contour = f0_values.tolist()
            else:
                pitch_mean = 0
                pitch_std = 0
                pitch_range = 0
                intonation_contour = []
            
            # Energy variation (dynamic range)
            energy = librosa.feature.rms(y=y)[0]
            energy_variation = np.std(energy)
            
            # Emphasis detection (stress/emphasis through intensity)
            # Look for energy peaks relative to surrounding
            emphasis_score = self._calculate_emphasis_score(energy)
            
            return ProsodyFeatures(
                pitch_mean=float(pitch_mean),
                pitch_std=float(pitch_std),
                pitch_range=float(pitch_range),
                intonation_contour=intonation_contour,
                energy_variation=float(energy_variation),
                emphasis_score=float(emphasis_score)
            )
            
        except Exception as e:
            logger.error(f"Prosody feature extraction failed: {e}")
            return ProsodyFeatures(0, 0, 0, [], 0, 0)
    
    async def _extract_stress_features(self, y: np.ndarray, sr: int, transcript: str) -> StressFeatures:
        """Extract word stress features."""
        try:
            words = transcript.split()
            syllable_timings = []
            
            # Get word boundaries (simplified - would use ASR timing in production)
            word_boundaries = np.linspace(0, len(y), len(words) + 1)
            
            # Analyze each word for stress patterns
            stressed_words = 0
            weak_form_words = 0
            
            for i, (word, (start, end)) in enumerate(zip(words, zip(word_boundaries[:-1], word_boundaries[1:]))):
                word_segment = y[int(start):int(end)]
                
                # Energy analysis for stress detection
                energy = librosa.feature.rms(y=word_segment)[0]
                relative_energy = np.mean(energy) / np.mean(librosa.feature.rms(y=y)[0])
                
                syllable_timings.append(float(end - start))
                
                # Simple stress detection based on energy and length
                if len(word) > 3 and relative_energy > 1.2:  # Likely stressed
                    stressed_words += 1
                elif len(word) <= 3 and relative_energy < 0.8:  # Likely weak form
                    weak_form_words += 1
            
            # Calculate stress pattern score
            expected_stress_positions = len([w for w in words if len(w.split()) > 1])  # Simplified
            actual_stress_ratio = stressed_words / max(len(words), 1)
            stress_pattern_score = 1.0 - abs(actual_stress_ratio - 0.3)  # Expected ~30% stressed
            
            return StressFeatures(
                stress_pattern_score=float(stress_pattern_score),
                primary_stress_placement=float(stressed_words / max(len(words), 1)),
                secondary_stress_placement=0.0,  # Would need more sophisticated analysis
                weak_form_pronunciation=float(weak_form_words / max(len(words), 1)),
                syllable_timing=syllable_timings
            )
            
        except Exception as e:
            logger.error(f"Stress feature extraction failed: {e}")
            return StressFeatures(0, 0, 0, 0, [])
    
    def _calculate_emphasis_score(self, energy: np.ndarray) -> float:
        """Calculate emphasis score based on energy peaks."""
        try:
            # Find significant energy peaks
            energy_smooth = np.convolve(energy, np.ones(10)/10, mode='same')
            peaks = []
            
            for i in range(1, len(energy_smooth) - 1):
                if (energy_smooth[i] > energy_smooth[i-1] and 
                    energy_smooth[i] > energy_smooth[i+1] and
                    energy_smooth[i] > np.mean(energy_smooth) + np.std(energy_smooth)):
                    peaks.append(i)
            
            # Emphasis score based on frequency and prominence of peaks
            emphasis_score = min(len(peaks) / (len(energy) / 100), 1.0)  # Normalize
            return float(emphasis_score)
            
        except Exception:
            return 0.0
    
    async def _calculate_accuracy_score(self, audio_path: str, transcript: str) -> float:
        """Calculate pronunciation accuracy using fine-tuned Wav2Vec2."""
        # This would use the fine-tuned Wav2Vec2 model
        # For now, return placeholder that will be implemented after training
        return 75.0  # Placeholder
    
    async def _calculate_fluency_score(self, features: FluencyFeatures) -> float:
        """Calculate fluency score from features."""
        score = 50.0  # Base score
        
        # Speech rate scoring (ideal: 150-180 WPM for English)
        if 150 <= features.speech_rate <= 180:
            score += 20
        elif 120 <= features.speech_rate <= 200:
            score += 15
        elif 100 <= features.speech_rate <= 220:
            score += 10
        
        # Pause ratio scoring (ideal: 20-30% pauses)
        if 0.2 <= features.pause_ratio <= 0.3:
            score += 15
        elif 0.15 <= features.pause_ratio <= 0.35:
            score += 10
        
        # Rhythm regularity
        score += features.rhythm_regularity * 10
        
        # Disfluency penalty
        score -= min(features.disfluency_count * 2, 15)
        
        return max(0, min(100, score))
    
    async def _calculate_prosody_score(self, features: ProsodyFeatures) -> float:
        """Calculate prosody score from features."""
        score = 50.0  # Base score
        
        # Pitch variation (good variation shows expressive speech)
        if 50 <= features.pitch_std <= 150:
            score += 20
        elif 30 <= features.pitch_std <= 200:
            score += 15
        
        # Pitch range
        if features.pitch_range > 50:
            score += 15
        elif features.pitch_range > 30:
            score += 10
        
        # Energy variation (shows expressiveness)
        if features.emphasis_score > 0.3:
            score += 10
        elif features.emphasis_score > 0.2:
            score += 5
        
        return max(0, min(100, score))
    
    async def _calculate_stress_score(self, features: StressFeatures) -> float:
        """Calculate stress placement score."""
        score = features.stress_pattern_score * 80  # Base on pattern accuracy
        
        # Bonus for good weak form pronunciation
        if 0.2 <= features.weak_form_pronunciation <= 0.4:
            score += 10
        
        # Regular syllable timing bonus
        if len(features.syllable_timing) > 1:
            timing_std = np.std(features.syllable_timing)
            timing_mean = np.mean(features.syllable_timing)
            if timing_std / timing_mean < 0.3:
                score += 10
        
        return max(0, min(100, score))
    
    async def _generate_enhanced_feedback(
        self, 
        fluency: FluencyFeatures, 
        prosody: ProsodyFeatures, 
        stress: StressFeatures
    ) -> Dict:
        """Generate detailed feedback for each dimension."""
        
        feedback = {
            "fluency": {
                "analysis": "",
                "recommendations": [],
                "specific_areas": []
            },
            "prosody": {
                "analysis": "",
                "recommendations": [],
                "specific_areas": []
            },
            "stress": {
                "analysis": "",
                "recommendations": [],
                "specific_areas": []
            }
        }
        
        # Fluency feedback
        if fluency.speech_rate < 120:
            feedback["fluency"]["specific_areas"].append("Speech rate too slow")
            feedback["fluency"]["recommendations"].append("Try to speak at 150-180 words per minute")
        elif fluency.speech_rate > 200:
            feedback["fluency"]["specific_areas"].append("Speech rate too fast")
            feedback["fluency"]["recommendations"].append("Slow down to improve clarity")
        
        if fluency.pause_ratio > 0.4:
            feedback["fluency"]["specific_areas"].append("Too many pauses")
            feedback["fluency"]["recommendations"].append("Practice continuous speech")
        elif fluency.pause_ratio < 0.1:
            feedback["fluency"]["specific_areas"].append("Not enough pauses")
            feedback["fluency"]["recommendations"].append("Add natural pauses for clarity")
        
        # Prosody feedback
        if prosody.pitch_std < 30:
            feedback["prosody"]["specific_areas"].append("Monotonous intonation")
            feedback["prosody"]["recommendations"].append("Vary your pitch for better expression")
        
        if prosody.emphasis_score < 0.2:
            feedback["prosody"]["specific_areas"].append("Need more emphasis on important words")
            feedback["prosody"]["recommendations"].append("Practice stress patterns in sentences")
        
        # Stress feedback
        if stress.stress_pattern_score < 0.6:
            feedback["stress"]["specific_areas"].append("Word stress patterns need improvement")
            feedback["stress"]["recommendations"].append("Focus on English stress rules (nouns vs verbs)")
        
        return feedback
    
    async def _fallback_scoring(self, audio_path: str, transcript: str, language: str) -> Dict:
        """Fallback scoring if comprehensive analysis fails."""
        logger.warning("Using fallback scoring due to analysis failure")
        return {
            "overall_score": 60.0,
            "dimensions": {
                "accuracy": 60.0,
                "fluency": 60.0,
                "prosody": 60.0,
                "stress": 60.0
            },
            "features": {},
            "feedback": "Scoring analysis encountered issues. Please try again.",
            "language": language,
            "analysis_level": "fallback"
        }
