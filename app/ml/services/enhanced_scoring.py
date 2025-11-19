"""Enhanced multi-dimensional scoring service for English pronunciation."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import structlog
import torch
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.core.config import settings
from app.ml.services.audio_processing import AudioPreprocessor, ProcessedAudio
from app.ml.services.error_detection import ErrorDetectionService, WordError
from app.ml.services.phoneme_service import AlignmentResult
from app.ml.services.scoring_service import ErrorDetail
from app.ml.models.wav2vec_trainer import EnglishPronunciationAnalyzer
from app.ml.services.optimized_model_loader import (
    get_optimized_model_and_processor,
    initialize_model_loader
)

logger = structlog.get_logger()

LANGUAGE_MODEL_PATHS = {
    "en-US": settings.WAV2VEC_ENGLISH_MODEL_PATH,
}

PHONEME_MAPS = {
    "en-US": {
        'a': 'ə', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɪ',
        'f': 'f', 'g': 'g', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'əʊ',
        'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
        'u': 'ʊ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z'
    },
    "id-ID": {
        'a': 'a', 'b': 'b', 'c': 'tʃ', 'd': 'd', 'e': 'e',
        'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'dʒ',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o',
        'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
        'u': 'u', 'v': 'f', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z',
        'ŋ': 'ŋ', 'ñ': 'ɲ'
    }
}

@dataclass
class FluencyFeatures:
    """Fluency analysis features."""

    speech_rate: float = 0.0            # words per minute
    pause_duration: float = 0.0         # average pause duration
    speech_duration: float = 0.0        # total speaking duration
    pause_ratio: float = 0.0            # pause_time / total_time
    disfluency_count: int = 0          # number of disfluencies
    rhythm_regularity: float = 0.0     # timing consistency score

@dataclass 
class ProsodyFeatures:
    """Prosody analysis features."""
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    intonation_contour: List[float] = field(default_factory=list)
    energy_variation: float = 0.0
    emphasis_score: float = 0.0

@dataclass
class StressFeatures:
    """Word stress analysis features."""
    stress_pattern_score: float = 0.0
    primary_stress_placement: float = 0.0
    secondary_stress_placement: float = 0.0
    weak_form_pronunciation: float = 0.0
    syllable_timing: List[float] = field(default_factory=list)

    @property
    def syllable_timings(self) -> List[float]:
        """Backward-compatible alias used by newer helpers."""
        return self.syllable_timing

    @syllable_timings.setter
    def syllable_timings(self, value: List[float]) -> None:
        self.syllable_timing = value

class EnhancedScoringService:
    """Enhanced multi-dimensional scoring for English pronunciation."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.analyzer = EnglishPronunciationAnalyzer()
        self.audio_processor = AudioPreprocessor(target_sample_rate=self.sample_rate)
        self.error_detector = ErrorDetectionService()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language_model_paths = LANGUAGE_MODEL_PATHS
        self.accuracy_models: Dict[str, Wav2Vec2ForCTC] = {}
        self.accuracy_processors: Dict[str, Wav2Vec2Processor] = {}
        self._accuracy_ready: Dict[str, bool] = {}
        
        try:
            initialize_model_loader()
            logger.info("Optimized model loader initialized")
        except Exception as e:
            logger.warning("Failed to initialize optimized model loader", error=str(e))

    def supports_language(self, language: str) -> bool:
        """Return True if comprehensive scoring is available for language."""
        return language in self.language_model_paths
    
    async def calculate_comprehensive_score(
        self,
        audio_path: str,
        transcript: str,
        language: str = "en-US",
        alignment: Optional[AlignmentResult] = None,
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
            if language not in self.language_model_paths:
                logger.warning("Enhanced scoring not available for language", language=language)
                return await self._fallback_scoring(audio_path, transcript, language)

            logger.info("Starting comprehensive scoring", language=language)

            processed_audio = await self.audio_processor.process(audio_path)
            y = processed_audio.waveform
            sr = processed_audio.sample_rate

            # Extract fluency/prosody/stress features from optimized audio.
            fluency_features = await self._extract_fluency_features(y, sr, transcript)
            prosody_features = await self._extract_prosody_features(y, sr)
            stress_features = await self._extract_stress_features(y, sr, transcript)

            accuracy_score, recognized_text, wer_score = await self._calculate_accuracy_score(
                waveform=y,
                transcript=transcript,
                language=language,
            )

            fluency_score = self._calculate_fluency_score(fluency_features)
            prosody_score = self._calculate_prosody_score(prosody_features)
            stress_score = self._calculate_stress_score(stress_features)

            weights = {"accuracy": 0.4, "fluency": 0.2, "prosody": 0.2, "stress": 0.2}
            overall_score = (
                accuracy_score * weights["accuracy"]
                + fluency_score * weights["fluency"]
                + prosody_score * weights["prosody"]
                + stress_score * weights["stress"]
            )

            phonemes = [p.phoneme for p in alignment.phonemes] if alignment else None
            expected_phonemes = self._approximate_phonemes(transcript, language)
            actual_phonemes = self._approximate_phonemes(recognized_text, language)
            detection = self.error_detector.analyze(
                reference_text=transcript,
                hypothesis_text=recognized_text,
                expected_phonemes=expected_phonemes or phonemes,
                actual_phonemes=actual_phonemes or phonemes,
                language=language,
            )
            error_details = [
                ErrorDetail(
                    type=err.type,
                    expected=err.expected,
                    actual=err.actual,
                    position=err.position,
                    confidence=err.confidence,
                )
                for err in detection.word_errors
            ]

            confidence = max(0.0, min(1.0, 1.0 - wer_score))
            feedback = self._generate_enhanced_feedback(
                fluency_features, prosody_features, stress_features
            )

            return {
                "overall_score": overall_score,
                "dimensions": {
                    "accuracy": accuracy_score,
                    "fluency": fluency_score,
                    "prosody": prosody_score,
                    "stress": stress_score,
                },
                "features": {
                    "fluency": fluency_features,
                    "prosody": prosody_features,
                    "stress": stress_features,
                },
                "feedback": feedback,
                "language": language,
                "analysis_level": "comprehensive",
                "errors": error_details,
                "confidence": confidence,
                "recognized_transcript": recognized_text,
                "wer": wer_score,
                "audio_profile": {
                    "duration": processed_audio.duration,
                    "sample_rate": processed_audio.sample_rate,
                    "processing": processed_audio.metadata,
                },
                "error_summary": detection.summary,
                "phoneme_insights": detection.phoneme_insights,
            }
            
        except Exception as e:
            logger.error("Comprehensive scoring failed", error=str(e))
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
            logger.error("Fluency feature extraction failed", error=str(e))
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
            logger.error("Prosody feature extraction failed", error=str(e))
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
            logger.error("Stress feature extraction failed", error=str(e))
            return StressFeatures(0, 0, 0, 0, [])
    
    def _calculate_emphasis_score(self, energy: np.ndarray) -> float:
        """Calculate emphasis score based on energy peaks."""
        try:
            if len(energy) < 3:
                return 0.0

            # Smooth to reduce spurious spikes but keep short utterance traits
            window = min(10, max(3, len(energy) // 5))
            kernel = np.ones(window) / window
            energy_smooth = np.convolve(energy, kernel, mode='same')

            mean_energy = np.mean(energy_smooth)
            std_energy = np.std(energy_smooth) + 1e-6
            dynamic_range = np.max(energy_smooth) - np.min(energy_smooth)

            peaks = [
                i for i in range(1, len(energy_smooth) - 1)
                if energy_smooth[i] > energy_smooth[i - 1]
                and energy_smooth[i] > energy_smooth[i + 1]
                and energy_smooth[i] > mean_energy + 0.5 * std_energy
            ]

            prominence = dynamic_range / (mean_energy + std_energy)
            if prominence < 0.4:
                return float(min(0.2, 0.5 * prominence))

            if not peaks:
                return float(min(0.15, 0.3 * prominence))

            peak_density = len(peaks) / max(len(energy_smooth), 1)
            emphasis_score = min(1.0, 0.5 * prominence + 1.2 * peak_density)
            return float(max(0.0, emphasis_score))
            
        except Exception:
            return 0.0
    
    async def _calculate_accuracy_score(
        self,
        waveform: np.ndarray,
        transcript: str,
        language: str,
    ) -> Tuple[float, str, float]:
        """Calculate pronunciation accuracy using the fine-tuned Wav2Vec2 model."""

        normalized_transcript = transcript.lower().strip()

        if not await self._load_accuracy_model(language):
            logger.warning(
                "Wav2Vec2 model not available; using heuristic accuracy",
                language=language,
            )
            return 72.0, normalized_transcript, 0.28

        processor = self.accuracy_processors.get(language)
        model = self.accuracy_models.get(language)

        async def _infer_text() -> str:
            def _run() -> str:
                inputs = processor(
                    waveform,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(self.device)
                attention_mask = (
                    inputs.attention_mask.to(self.device)
                    if hasattr(inputs, "attention_mask")
                    else None
                )

                with torch.no_grad():
                    logits = model(
                        input_values,
                        attention_mask=attention_mask,
                    ).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(predicted_ids)[0]
                return text.strip().lower()

            return await asyncio.to_thread(_run)

        recognized_text = await _infer_text()

        if not normalized_transcript:
            return 50.0, recognized_text, 1.0

        wer_score = wer(normalized_transcript, recognized_text)
        wer_score = float(max(0.0, min(1.0, wer_score)))
        accuracy_score = max(0.0, min(100.0, 100.0 * (1.0 - wer_score)))

        return accuracy_score, recognized_text, wer_score

    async def _load_accuracy_model(self, language: str) -> bool:
        if self._accuracy_ready.get(language):
            return True

        model_path_str = self.language_model_paths.get(language)
        if not model_path_str:
            logger.warning("No model path configured for language", language=language)
            self._accuracy_ready[language] = False
            return False

        try:
            # Use optimized model loader to automatically load optimized version if available
            model, processor, metadata = await get_optimized_model_and_processor(language)
            
            if model is None or processor is None:
                logger.error(
                    "Failed to load model and processor",
                    language=language,
                )
                self._accuracy_ready[language] = False
                return False
            
            self.accuracy_processors[language] = processor
            self.accuracy_models[language] = model
            self._accuracy_ready[language] = True
            
            # Log which model was loaded (optimized or original)
            model_type = metadata.get('model_type', 'unknown')
            optimization_level = metadata.get('optimization_level', 0)
            logger.info(
                "Loaded Wav2Vec2 model",
                language=language,
                model_type=model_type,
                optimization_level=optimization_level,
                device=self.device,
            )
            
            # Log performance metrics if available
            perf_metrics = metadata.get('performance_metrics', {})
            if perf_metrics:
                logger.info(
                    "Model optimization metrics",
                    metrics=perf_metrics
                )
            
            return True
            
        except Exception as exc:
            logger.error(
                "Failed to load Wav2Vec2 model",
                language=language,
                error=str(exc),
            )
            self._accuracy_ready[language] = False
            return False
    
    def _calculate_fluency_score(self, features: FluencyFeatures) -> float:
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
    
    def _calculate_prosody_score(self, features: ProsodyFeatures) -> float:
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
    
    def _calculate_stress_score(self, features: StressFeatures) -> float:
        """Calculate stress placement score."""
        score = features.stress_pattern_score * 80  # Base on pattern accuracy
        
        # Bonus for good weak form pronunciation
        if 0.2 <= features.weak_form_pronunciation <= 0.4:
            score += 10
        
        # Regular syllable timing bonus
        if len(features.syllable_timings) > 1:
            timing_std = np.std(features.syllable_timings)
            timing_mean = np.mean(features.syllable_timings)
            if timing_std / timing_mean < 0.3:
                score += 10
        
        return max(0, min(100, score))
    
    def _generate_enhanced_feedback(
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
            feedback["fluency"]["recommendations"].append(
                "Your speech rate is slow; try to speak at 150-180 words per minute"
            )
        elif fluency.speech_rate > 200:
            feedback["fluency"]["specific_areas"].append("Speech rate too fast")
            feedback["fluency"]["recommendations"].append("Speech feels too fast; slow down to improve clarity")
        
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
    
    def _approximate_phonemes(self, text: str, language: str) -> List[str]:
        """Create a lightweight phoneme approximation for error analysis."""
        mapping = PHONEME_MAPS.get(language, PHONEME_MAPS["en-US"])
        phonemes: List[str] = []
        for char in text.lower():
            if char.isalpha() or char in ("ŋ", "ñ"):
                phonemes.append(mapping.get(char, char))
        return phonemes

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
            "analysis_level": "fallback",
            "errors": [],
            "confidence": 0.5,
            "recognized_transcript": transcript.lower().strip(),
            "wer": 0.4,
            "audio_profile": None,
            "error_summary": {"total": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        }
