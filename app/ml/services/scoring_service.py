import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class ErrorDetail:
    type: str  # substitution, deletion, insertion
    expected: str
    actual: str
    position: int
    confidence: float
    phoneme_index: Optional[int] = None

@dataclass
class ScoringResult:
    overall_score: float
    dimensions: Dict[str, float]  # accuracy, fluency, prosody, stress
    errors: List[ErrorDetail]
    confidence: float
    details: Dict

class ScoringService:
    """Service for pronunciation scoring using Goodness of Pronunciation (GOP) algorithm."""
    
    def __init__(self):
        self.gop_thresholds = {
            "vowel": 0.7,
            "consonant": 0.6,
            "cluster": 0.5,
            "overall": 0.65
        }
        self.dimension_weights = {
            "accuracy": 0.4,
            "fluency": 0.2,
            "prosody": 0.2,
            "stress": 0.2
        }
    
    async def calculate_gop_score(
        self, 
        expected_phonemes: List[str],
        actual_phonemes: List[str],
        confidence_scores: List[float] = None
    ) -> ScoringResult:
        """
        Calculate Goodness of Pronunciation score.
        
        Args:
            expected_phonemes: Expected phonemes from reference
            actual_phonemes: Detected phonemes from user audio
            confidence_scores: Confidence scores for each phoneme
            
        Returns:
            ScoringResult with overall score and detailed breakdown
        """
        try:
            logger.info(
                "Calculating GOP score",
                expected_count=len(expected_phonemes),
                actual_count=len(actual_phonemes)
            )
            
            # Ensure confidence scores are provided
            if confidence_scores is None:
                confidence_scores = [0.85] * len(expected_phonemes)
            
            # Calculate phoneme-by-phoneme GOP scores
            phoneme_scores = []
            errors = []
            
            for i, (exp_phone, act_phone) in enumerate(zip(expected_phonemes, actual_phonemes)):
                # Skip if out of bounds
                if i >= len(confidence_scores):
                    break
                
                # Calculate GOP score for this phoneme
                gop_score = self._calculate_phoneme_gop(
                    exp_phone, act_phone, confidence_scores[i]
                )
                phoneme_scores.append(gop_score)
                
                # Detect errors
                if exp_phone != act_phone:
                    error = self._classify_error(exp_phone, act_phone, i, gop_score)
                    errors.append(error)
            
            # Handle substitutions/deletions/insertions
            alignment_errors = self._detect_alignment_errors(
                expected_phonemes, actual_phonemes, phoneme_scores
            )
            errors.extend(alignment_errors)
            
            # Calculate overall and dimension scores
            overall_score = self._calculate_overall_score(phoneme_scores, errors)
            dimension_scores = self._calculate_dimension_scores(
                phoneme_scores, errors, expected_phonemes
            )
            
            # Calculate confidence
            confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return ScoringResult(
                overall_score=overall_score,
                dimensions=dimension_scores,
                errors=errors,
                confidence=confidence,
                details={
                    "phoneme_scores": phoneme_scores,
                    "expected_phonemes": expected_phonemes,
                    "actual_phonemes": actual_phonemes,
                    "gop_thresholds": self.gop_thresholds
                }
            )
            
        except Exception as e:
            logger.error("GOP scoring failed", error=str(e))
            # Return default result on error
            return ScoringResult(
                overall_score=50.0,
                dimensions={"accuracy": 50.0, "fluency": 50.0, "prosody": 50.0, "stress": 50.0},
                errors=[],
                confidence=0.0,
                details={}
            )
    
    def _calculate_phoneme_gop(
        self, 
        expected: str, 
        actual: str, 
        confidence: float
    ) -> float:
        """Calculate GOP score for individual phoneme."""
        if expected == actual:
            # Perfect match
            return 100.0
        else:
            # Mismatch - calculate similarity score
            similarity = self._calculate_phoneme_similarity(expected, actual)
            return similarity * confidence * 80.0  # Max 80% for mismatches
    
    def _calculate_phoneme_similarity(self, expected: str, actual: str) -> float:
        """Calculate phoneme similarity based on articulatory features."""
        # Very basic similarity calculation - would be more sophisticated in production
        similar_pairs = {
            ('t', 'd'): 0.8,  # Alveolar stops
            ('k', 'g'): 0.8,  # Velar stops
            ('p', 'b'): 0.8,  # Bilabial stops
            ('s', 'z'): 0.7,  # Alveolar fricatives
            ('f', 'v'): 0.7,  # Labiodental fricatives
            ('i', 'ɪ'): 0.9,  # Vowel similarity
            ('e', 'ɛ'): 0.9,  # Vowel similarity
            ('a', 'ə'): 0.8,  # Vowel similarity
        }
        
        # Check if we have predefined similarity
        key = (expected, actual)
        reverse_key = (actual, expected)
        
        if key in similar_pairs:
            return similar_pairs[key]
        elif reverse_key in similar_pairs:
            return similar_pairs[reverse_key]
        else:
            # No predefined similarity, return low value
            return 0.3  # Default low similarity
    
    def _classify_error(
        self, 
        expected: str, 
        actual: str, 
        position: int,
        gop_score: float
    ) -> ErrorDetail:
        """Classify type of pronunciation error."""
        if expected == actual:
            error_type = "correct"
        elif actual == "":
            error_type = "deletion"
        elif expected == "":
            error_type = "insertion"
        else:
            error_type = "substitution"
        
        return ErrorDetail(
            type=error_type,
            expected=expected,
            actual=actual,
            position=position,
            confidence=min(gop_score / 100.0, 1.0)
        )
    
    def _detect_alignment_errors(
        self, 
        expected: List[str], 
        actual: List[str], 
        scores: List[float]
    ) -> List[ErrorDetail]:
        """Detect alignment errors (insertions, deletions)."""
        errors = []
        
        # Simple length-based error detection
        if len(expected) != len(actual):
            if len(actual) > len(expected):
                # Potential insertions
                for i in range(len(expected), len(actual)):
                    if i < len(actual):
                        errors.append(ErrorDetail(
                            type="insertion",
                            expected="",
                            actual=actual[i],
                            position=i,
                            confidence=0.7
                        ))
            else:
                # Potential deletions
                for i in range(len(actual), len(expected)):
                    if i < len(expected):
                        errors.append(ErrorDetail(
                            type="deletion",
                            expected=expected[i],
                            actual="",
                            position=i,
                            confidence=0.7
                        ))
        
        return errors
    
    def _calculate_overall_score(
        self, 
        phoneme_scores: List[float], 
        errors: List[ErrorDetail]
    ) -> float:
        """Calculate overall pronunciation score."""
        if not phoneme_scores:
            return 50.0
        
        # Base score from phoneme accuracy
        base_score = np.mean(phoneme_scores)
        
        # Penalty for errors
        error_penalty = len(errors) * 5.0  # 5 points per error
        max_penalty = 25.0  # Maximum penalty of 25 points
        
        final_score = base_score - min(error_penalty, max_penalty)
        return max(0.0, min(100.0, final_score))
    
    def _calculate_dimension_scores(
        self, 
        phoneme_scores: List[float], 
        errors: List[ErrorDetail], 
        expected_phonemes: List[str]
    ) -> Dict[str, float]:
        """Calculate dimension-specific scores."""
        # Accuracy dimension - based on phoneme scores
        accuracy = np.mean(phoneme_scores) if phoneme_scores else 50.0
        
        # Fluency dimension - placeholder (will be enhanced in Sprint 2)
        fluency = max(50.0, accuracy - 10.0) if len(errors) > 2 else accuracy
        
        # Prosody dimension - placeholder (will be enhanced in Sprint 2)
        prosody = accuracy if accuracy > 75.0 else accuracy + 5.0
        
        # Stress dimension - placeholder (will be enhanced in Sprint 2)
        stress = accuracy - len([e for e in errors if e.type == "substitution"]) * 2.0
        
        # Ensure scores are in [0, 100] range
        dimensions = {
            "accuracy": max(0.0, min(100.0, accuracy)),
            "fluency": max(0.0, min(100.0, fluency)),
            "prosody": max(0.0, min(100.0, prosody)),
            "stress": max(0.0, min(100.0, stress))
        }
        
        return dimensions
