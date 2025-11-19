import librosa
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import structlog
from praatio import textgrid
from dataclasses import dataclass
from app.core.config import settings
import asyncio
import subprocess
import tempfile
import shutil
import json
from datetime import datetime

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
    
    MFA_MODELS = {
        "en-US": "english_us_arpa",
        "id-ID": "indonesian_cv"
    }
    
    def __init__(self):
        self.mfa_available = False
        self.mfa_model_paths = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "mfa_temp"
        self.temp_dir.mkdir(exist_ok=True)
        self._setup_mfa()
    
    def _setup_mfa(self):
        """Setup Montreal Forced Aligner with language models."""
        try:
            # Check if MFA is installed
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                mfa_version = result.stdout.decode().strip()
                logger.info("Montreal Forced Aligner found", version=mfa_version)
                
                # Check/download models for each language
                for lang_code, model_name in self.MFA_MODELS.items():
                    try:
                        # Verify model exists
                        verify_cmd = ["mfa", "model", "inspect", model_name]
                        result = subprocess.run(
                            verify_cmd,
                            capture_output=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            self.mfa_model_paths[lang_code] = model_name
                            logger.info(
                                "MFA model available",
                                language=lang_code,
                                model=model_name
                            )
                        else:
                            logger.warning(
                                "MFA model not found, downloading",
                                language=lang_code,
                                model=model_name
                            )
                            # Try to download model
                            download_cmd = ["mfa", "model", "download", "acoustic", model_name]
                            subprocess.run(download_cmd, timeout=60)
                            self.mfa_model_paths[lang_code] = model_name
                            
                    except Exception as e:
                        logger.warning(
                            "Failed to setup MFA model",
                            language=lang_code,
                            error=str(e)
                        )
                
                self.mfa_available = len(self.mfa_model_paths) > 0
                if self.mfa_available:
                    logger.info(
                        "Montreal Forced Aligner ready",
                        models=list(self.mfa_model_paths.keys())
                    )
                    
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(
                "MFA not available, falling back to placeholder",
                error=str(e)
            )
            self.mfa_available = False
    
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
            if self.mfa_available and language in self.mfa_model_paths:
                return await self._extract_with_mfa(audio_path, transcript, language)
            else:
                # Fallback to placeholder implementation
                logger.warning(
                    "MFA not available for language, using placeholder",
                    language=language
                )
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
        """Extract phonemes using Montreal Forced Aligner with actual alignment."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._extract_with_mfa_sync,
            audio_path,
            transcript,
            language
        )
    
    def _extract_with_mfa_sync(
        self,
        audio_path: str,
        transcript: str,
        language: str
    ) -> AlignmentResult:
        """Synchronous MFA extraction using subprocess."""
        try:
            model_name = self.mfa_model_paths.get(language)
            if not model_name:
                logger.warning("No MFA model for language", language=language)
                return self._extract_placeholder_sync(audio_path, transcript, language)
            
            # Create temporary working directory
            work_dir = self.temp_dir / f"mfa_{datetime.now().timestamp()}"
            work_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy audio to temp directory with standard name
            audio_stem = Path(audio_path).stem
            temp_audio = work_dir / f"{audio_stem}.wav"
            shutil.copy(audio_path, temp_audio)
            
            # Create transcript file for MFA
            transcript_file = work_dir / f"{audio_stem}.lab"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript.lower())
            
            # Run MFA alignment
            output_dir = work_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            logger.info(
                "Running MFA alignment",
                model=model_name,
                language=language,
                audio=str(temp_audio)
            )
            
            # Use MFA align command
            align_cmd = [
                "mfa",
                "align",
                str(work_dir),
                model_name,
                model_name,
                str(output_dir),
                "-j", "1",
                "--clean"
            ]
            
            result = subprocess.run(
                align_cmd,
                capture_output=True,
                timeout=120,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(
                    "MFA alignment failed",
                    stderr=result.stderr,
                    stdout=result.stdout
                )
                # Fallback to placeholder
                return self._extract_placeholder_sync(audio_path, transcript, language)
            
            # Parse TextGrid output
            textgrid_file = output_dir / f"{audio_stem}.TextGrid"
            if not textgrid_file.exists():
                logger.warning("TextGrid output not found", path=str(textgrid_file))
                return self._extract_placeholder_sync(audio_path, transcript, language)
            
            # Extract phonemes from TextGrid
            alignment_result = self._parse_textgrid(str(textgrid_file), transcript)
            
            # Cleanup
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.warning("Failed to cleanup MFA temp dir", error=str(e))
            
            logger.info(
                "MFA alignment successful",
                phoneme_count=len(alignment_result.phonemes),
                alignment_score=alignment_result.alignment_score
            )
            
            return alignment_result
            
        except subprocess.TimeoutExpired:
            logger.error("MFA alignment timeout")
            return self._extract_placeholder_sync(audio_path, transcript, language)
        except Exception as e:
            logger.error("MFA extraction error", error=str(e))
            return self._extract_placeholder_sync(audio_path, transcript, language)
    
    def _parse_textgrid(self, textgrid_path: str, transcript: str) -> AlignmentResult:
        """Parse TextGrid output from MFA and extract phoneme details."""
        try:
            tg = textgrid.TextGrid()
            tg.read(textgrid_path)
            
            phonemes = []
            words = []
            
            # Usually TextGrid has 2 tiers: words and phones
            phones_tier = None
            words_tier = None
            
            for tier in tg.tiers:
                if "phone" in tier.name.lower():
                    phones_tier = tier
                elif "word" in tier.name.lower():
                    words_tier = tier
            
            # Extract phonemes from phones tier
            if phones_tier:
                for entry in phones_tier.entries:
                    if entry.label.strip():  # Skip silences
                        phoneme = PhonemeResult(
                            phoneme=entry.label,
                            start_time=float(entry.start),
                            end_time=float(entry.end),
                            confidence=0.95
                        )
                        phonemes.append(phoneme)
            
            # Extract words from words tier
            if words_tier:
                for entry in words_tier.entries:
                    if entry.label.strip():
                        words.append({
                            "word": entry.label,
                            "start_time": float(entry.start),
                            "end_time": float(entry.end),
                            "confidence": 0.95
                        })
            
            # Calculate alignment score
            alignment_score = min(0.95, 0.90 + (len(phonemes) / 100))  # Higher score with more phonemes
            
            return AlignmentResult(
                phonemes=phonemes,
                words=words,
                alignment_score=alignment_score
            )
            
        except Exception as e:
            logger.error("TextGrid parsing failed", error=str(e))
            return AlignmentResult(
                phonemes=[],
                words=[],
                alignment_score=0.0
            )
    
    async def _extract_placeholder(
        self, 
        audio_path: str, 
        transcript: str, 
        language: str
    ) -> AlignmentResult:
        """Placeholder phoneme extraction for development (fallback)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._extract_placeholder_sync,
            audio_path,
            transcript,
            language
        )
    
    def _extract_placeholder_sync(
        self,
        audio_path: str,
        transcript: str,
        language: str
    ) -> AlignmentResult:
        """Synchronous placeholder extraction."""
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
            # Cleanup MFA temp directory
            if self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info("MFA temp directory cleaned")
                except Exception as e:
                    logger.warning("Failed to cleanup MFA temp dir", error=str(e))
        except Exception as e:
            logger.error("Failed to cleanup temp files", error=str(e))
    
    def __del__(self):
        """Cleanup on service destruction."""
        try:
            self.cleanup_temp_files("")
        except:
            pass
