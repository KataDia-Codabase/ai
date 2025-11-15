import librosa
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import structlog
# praatio 5.x uses 'textgrid' module instead of 'tgio'
from praatio import textgrid
from dataclasses import dataclass
from app.core.config import settings
import asyncio

logger = structlog.get_logger()

@dataclass
class PhonemeResult:
    phoneme: str
    start_time: float
    end_time: float
    confidence: float

@dataclass
class AlignmentResult:
    phonemes: List[PhonemeResult]
    words: List[Dict]
    alignment_score: float

class PhonemeService:
    """Service for phoneme extraction and alignment using Montreal Forced Aligner."""
    
    def __init__(self):
        self.mfa_setup = False
        self.indonesian_dict = None
        self.english_dict = None
        self._setup_mfa()
    
    def _setup_mfa(self):
        """Setup Montreal Forced Aligner configuration."""
        try:
            # This is a placeholder for MFA setup
            # In actual implementation, we would setup MFA models and dictionaries
            logger.info("Setting up Montreal Forced Aligner")
            
            # Check if dictionaries exist
            indonesian_dict_path = Path(settings.MFA_MODEL_PATH) / "indonesian_dictionary.mfa"
            english_dict_path = Path(settings.MFA_MODEL_PATH) / "english_dictionary.mfa"
            
            if indonesian_dict_path.exists():
                self.indonesian_dict = str(indonesian_dict_path)
                logger.info("Indonesian dictionary found")
            
            if english_dict_path.exists():
                self.english_dict = str(english_dict_path)
                logger.info("English dictionary found")
            
            self.mfa_setup = True
            
        except Exception as e:
            logger.error("Failed to setup MFA", error=str(e))
            # Continue without MFA, use placeholder implementation
    
    async def extract_phonemes(
        self, 
        audio_path: str, 
        transcript: str, 
        language: str
    ) -> AlignmentResult:
        """
        Extract phonemes with timing from audio and transcript.
        
        Args:
            audio_path: Path to audio file
            transcript: Ground truth transcript
            language: Language code (id-ID or en-US)
            
        Returns:
            AlignmentResult with phonemes, words, and confidence scores
        """
        try:
            if self.mfa_setup:
                return await self._extract_with_mfa(audio_path, transcript, language)
            else:
                # Fallback to placeholder implementation
                return await self._extract_placeholder(audio_path, transcript, language)
                
        except Exception as e:
            logger.error(
                "Phoneme extraction failed",
                audio_path=audio_path,
                transcript=transcript[:50],
                language=language,
                error=str(e)
            )
            # Return placeholder result on error
            return await self._extract_placeholder(audio_path, transcript, language)
    
    async def _extract_with_mfa(
        self, 
        audio_path: str, 
        transcript: str, 
        language: str
    ) -> AlignmentResult:
        """Extract phonemes using Montreal Forced Aligner."""
        # This is a placeholder for MFA integration
        # In actual implementation, we would call MFA CLI or Python API
        logger.info("Extracting phonemes with MFA")
        
        # TODO: Implement actual MFA integration
        # 1. Prepare audio and transcript files
        # 2. Call MFA align command
        # 3. Parse TextGrid output
        # 4. Extract phonemes with timing
        
        # For now, return placeholder
        return await self._extract_placeholder(audio_path, transcript, language)
    
    async def _extract_placeholder(
        self, 
        audio_path: str, 
        transcript: str, 
        language: str
    ) -> AlignmentResult:
        """Placeholder phoneme extraction for development."""
        try:
            # Load audio to get duration
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # Simple word segmentation (placeholder)
            words = transcript.split()
            word_durations = duration / len(words)
            
            # Generate placeholder phonemes based on words
            phonemes = []
            current_time = 0.0
            
            for i, word in enumerate(words):
                # Simple phoneme estimation (very basic)
                phoneme_str = self._word_to_phonemes_placeholder(word, language)
                phoneme_duration = word_durations / max(len(phoneme_str), 1)
                
                j = 0
                while j < len(phoneme_str) - 1:
                    phoneme = PhonemeResult(
                        phoneme=phoneme_str[j] + phoneme_str[j+1],  # Consonant pairs
                        start_time=current_time,
                        end_time=current_time + phoneme_duration * 2,
                        confidence=0.85
                    )
                    phonemes.append(phoneme)
                    current_time += phoneme_duration * 2
                    j += 2
                
                # Handle last phoneme if odd
                if j < len(phoneme_str):
                    phoneme = PhonemeResult(
                        phoneme=phoneme_str[j],
                        start_time=current_time,
                        end_time=current_time + phoneme_duration,
                        confidence=0.85
                    )
                    phonemes.append(phoneme)
                    current_time += phoneme_duration
            
            # Create word timing info
            word_results = []
            word_start = 0.0
            for word in words:
                word_results.append({
                    "word": word,
                    "start_time": word_start,
                    "end_time": word_start + word_durations,
                    "confidence": 0.90
                })
                word_start += word_durations
            
            return AlignmentResult(
                phonemes=phonemes,
                words=word_results,
                alignment_score=0.85
            )
            
        except Exception as e:
            logger.error("Placeholder phoneme extraction failed", error=str(e))
            # Return empty result
            return AlignmentResult(
                phonemes=[],
                words=[],
                alignment_score=0.0
            )
    
    def _word_to_phonemes_placeholder(self, word: str, language: str) -> str:
        """Very basic phoneme conversion for development."""
        # This is a very crude placeholder - replace with actual phoneme dictionaries
        if language == "id-ID":
            # Indonesian phoneme approximation
            phoneme_map = {
                'a': 'a', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o',
                'b': 'b', 'c': 'tʃ', 'd': 'd', 'f': 'f', 'g': 'g',
                'h': 'h', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm',
                'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
                't': 't', 'v': 'f', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z'
            }
        else:
            # English phoneme approximation
            phoneme_map = {
                'a': 'ə', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɪ',
                'f': 'f', 'g': 'g', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
                'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'əʊ',
                'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
                'u': 'ʊ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z'
            }
        
        phonemes = []
        for char in word.lower():
            if char.isalpha():
                phonemes.append(phoneme_map.get(char, char))
        
        return ''.join(phonemes)
    
    def cleanup_temp_files(self, audio_path: str):
        """Clean up temporary files created during processing."""
        try:
            # TODO: Implement cleanup of MFA temporary files
            pass
        except Exception as e:
            logger.error("Failed to cleanup temp files", error=str(e))
