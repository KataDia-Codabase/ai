"""
CEFR assessment service for English pronunciation levels (A1-C2).
Implements comprehensive CEFR level assessment based on multiple dimensions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()

class CEFRLevel(Enum):
    """CEFR proficiency levels."""
    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient

@dataclass
class CEFRCriteria:
    """CEFR criteria for pronunciation assessment."""
    accuracy_range: Tuple[float, float]
    fluency_range: Tuple[float, float]
    prosody_range: Tuple[float, float]
    stress_range: Tuple[float, float]
    characteristics: List[str]
    common_issues: List[str]
    target_speech_rate: Tuple[float, float]  # WPM

# English-specific CEFR criteria
ENGLISH_CEFR_CRITERIA = {
    CEFRLevel.A1: CEFRCriteria(
        accuracy_range=(40, 60),
        fluency_range=(30, 50),
        prosody_range=(20, 40),
        stress_range=(20, 45),
        characteristics=[
            "Very limited pronunciation control",
            "Speech often unintelligible",
            "Strong L1 interference",
            "Basic sound substitution patterns"
        ],
        common_issues=[
            "Consonant mispronunciation",
            "Vowel reduction issues",
            "Incorrect stress placement",
            "Poor word segmentation"
        ],
        target_speech_rate=(80, 120)
    ),
    
    CEFRLevel.A2: CEFRCriteria(
        accuracy_range=(55, 70),
        fluency_range=(45, 65),
        prosody_range=(35, 55),
        stress_range=(40, 60),
        characteristics=[
            "Basic pronunciation control",
            "Generally intelligible with effort",
            "Still significant L1 influence",
            "Simple sentence patterns manageable"
        ],
        common_issues=[
            "Persistent sound errors",
            "Irregular speech rhythm",
            "Limited intonation patterns",
            "Stress on many words incorrect"
        ],
        target_speech_rate=(90, 130)
    ),
    
    CEFRLevel.B1: CEFRCriteria(
        accuracy_range=(65, 80),
        fluency_range=(60, 75),
        prosody_range=(50, 70),
        stress_range=(55, 75),
        characteristics=[
            "Clear pronunciation of common words",
            "Generally intelligible",
            "Some automaticity in speech",
            "Can handle familiar expressions well"
        ],
        common_issues=[
            "Problematic consonant clusters",
            "Incorrect word stress in complex words",
            "Limited intonation variation",
            "Hesitations and restarts common"
        ],
        target_speech_rate=(120, 160)
    ),
    
    CEFRLevel.B2: CEFRCriteria(
        accuracy_range=(75, 90),
        fluency_range=(70, 85),
        prosody_range=(65, 80),
        stress_range=(70, 88),
        characteristics=[
            "Clear and mostly accurate pronunciation",
            "Highly intelligible with native-like clarity",
            "Good control of rhythm and flow",
            "Comfortable with most sounds and patterns"
        ],
        common_issues=[
            "Minor inaccuracies in complex sounds",
            "Occasional stress mistakes",
            "Limited prosodic variation in formal speech",
            "Some interference with rare sounds"
        ],
        target_speech_rate=(140, 180)
    ),
    
    CEFRLevel.C1: CEFRCriteria(
        accuracy_range=(85, 95),
        fluency_range=(80, 92),
        prosody_range=(75, 90),
        stress_range=(80, 95),
        characteristics=[
            "High-level pronunciation accuracy",
            "Native-like clarity and precision",
            "Excellent control of English prosody",
            "Rare pronunciation errors"
        ],
        common_issues=[
            "Subtle sound nuances",
            "Complex stress patterns in specialized vocabulary",
            "Prosoda in technical discourse",
            "Minor regional accent elements"
        ],
        target_speech_rate=(150, 200)
    ),
    
    CEFRLevel.C2: CEFRCriteria(
        accuracy_range=(92, 100),
        fluency_range=(90, 100),
        prosody_range=(85, 100),
        stress_range=(92, 100),
        characteristics=[
            "Native-like pronunciation mastery",
            "Complete phonological control",
            "Full prosodic native-like delivery",
            "Consistent accuracy across all contexts"
        ],
        common_issues=[
            "Negligible pronunciation issues",
            "Professional-level precision needed",
            "Subtle stylistic variations",
            "Very minor accent traces"
        ],
        target_speech_rate=(160, 220)
    )
}

@dataclass
class CEFRAssessmentResult:
    """Result of CEFR assessment."""
    cefr_level: CEFRLevel
    confidence: float  # 0-1
    dimension_scores: Dict[str, float]
    level_confidence: Dict[CEFRLevel, float]
    strengths: List[str]
    weaknesses: List[str]
    next_level_requirements: Dict[str, float]
    progression_tips: List[str]

class EnglishCEFRAssessment:
    """CEFR assessment system for English pronunciation."""
    
    def __init__(self):
        self.criteria = ENGLISH_CEFR_CRITERIA
        self.english_priority_sounds = [
            'rhotic_R', 'TH_sounds', 'vowel_reductions', 
            'consonant_clusters', 'weak_forms', 'schwa_sounds'
        ]
    
    async def assess_cefr_level(
        self,
        comprehensive_score: Dict,
        user_history: Optional[List[Dict]] = None,
        target_language: str = "en-US"
    ) -> CEFRAssessmentResult:
        """
        Assess CEFR level based on comprehensive pronunciation scoring.
        
        Args:
            comprehensive_score: Results from enhanced scoring
            user_history: Previous assessment results
            target_language: Language (should be en-US for English)
            
        Returns:
            CEFRAssessmentResult with detailed assessment
        """
        try:
            logger.info("Starting CEFR level assessment for English")
            
            # Extract dimension scores
            dimensions = comprehensive_score.get('dimensions', {})
            features = comprehensive_score.get('features', {})
            
            # Calculate base level assessment
            level_probabilities = self._calculate_level_probabilities(dimensions, features)
            primary_level = max(level_probabilities.items(), key=lambda x: x[1])[0]
            confidence = level_probabilities[primary_level]
            
            # Analyze strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                dimensions, features, primary_level
            )
            
            # Determine requirements for next level
            next_level_requirements = self._calculate_next_level_reqs(
                primary_level, dimensions, features
            )
            
            # Generate progression tips
            progression_tips = self._generate_progression_tips(
                primary_level, weaknesses, features
            )
            
            return CEFRAssessmentResult(
                cefr_level=primary_level,
                confidence=float(confidence),
                dimension_scores=dimensions,
                level_confidence={k: float(v) for k, v in level_probabilities.items()},
                strengths=strengths,
                weaknesses=weaknesses,
                next_level_requirements=next_level_requirements,
                progression_tips=progression_tips
            )
            
        except Exception as e:
            logger.error(f"CEFR assessment failed: {e}")
            # Return default assessment
            return self._get_default_assessment(dimensions)
    
    def _calculate_level_probabilities(
        self, 
        dimensions: Dict[str, float], 
        features: Dict[str, Any] = None
    ) -> Dict[CEFRLevel, float]:
        """Calculate probability scores for each CEFR level."""
        
        level_scores = {}
        
        for level, criteria in self.criteria.items():
            score = 0.0
            weights = {
                'accuracy': 0.35,
                'fluency': 0.25, 
                'prosody': 0.20,
                'stress': 0.20
            }
            
            # Calculate dimension match scores
            for dimension, range_tuple in [
                ('accuracy', criteria.accuracy_range),
                ('fluency', criteria.fluency_range),
                ('prosody', criteria.prosody_range),
                ('stress', criteria.stress_range)
            ]:
                dimension_score = dimensions.get(dimension, 0)
                
                # Calculate how well score fits the criteria range
                if range_tuple[0] <= dimension_score <= range_tuple[1]:
                    # Perfect match
                    match_score = 1.0
                else:
                    # Calculate distance from range
                    if dimension_score < range_tuple[0]:
                        match_score = max(0, 1.0 - (range_tuple[0] - dimension_score) / 50)
                    else:
                        match_score = max(0, 1.0 - (dimension_score - range_tuple[1]) / 50)
                
                score += match_score * weights[dimension]
            
            # Add speech rate bonus if available
            if features and 'fluency' in features:
                speech_rate = features['fluency'].get('speech_rate', 0)
                target_min, target_max = criteria.target_speech_rate
                
                if target_min <= speech_rate <= target_max:
                    score += 0.05
                elif rating_difference := min(abs(speech_rate - target_min), abs(speech_rate - target_max)):
                    score -= max(0, 0.02 - rating_difference / 100)
            
            level_scores[level] = score
        
        # Normalize probabilities
        total_score = sum(level_scores.values())
        if total_score > 0:
            level_scores = {level: score/total_score for level, score in level_scores.items()}
        
        return level_scores
    
    def _analyze_strengths_weaknesses(
        self, 
        dimensions: Dict[str, float],
        features: Dict[str, Any],
        level: CEFRLevel
    ) -> Tuple[List[str], List[str]]:
        """Analyze specific strengths and weaknesses based on CEFR level."""
        
        strengths = []
        weaknesses = []
        
        # Analyze dimensional performance
        for dimension, score in dimensions.items():
            criteria_range = getattr(self.criteria[level], f"{dimension}_range")
            
            if score >= criteria_range[1]:
                strengths.append(f"Excellent {dimension} control")
            elif score >= criteria_range[0]:
                strengths.append(f"Good {dimension} performance")
            else:
                weaknesses.append(f"Improvement needed in {dimension}")
        
        # Add English-specific analysis
        if features:
            fluency_features = features.get('fluency', {})
            
            # Speech rate analysis
            speech_rate = fluency_features.get('speech_rate', 0)
            target_min, target_max = self.criteria[level].target_speech_rate
            
            if target_min <= speech_rate <= target_max:
                strengths.append("Appropriate speech rate")
            elif speech_rate < target_min:
                weaknesses.append("Speech rate too slow")
            else:
                weaknesses.append("Speech rate too fast")
            
            # Fluency analysis
            pause_ratio = fluency_features.get('pause_ratio', 0)
            if pause_ratio > 0.4:
                weaknesses.append("Excessive pausing affects fluency")
            elif pause_ratio < 0.1:
                weaknesses.append("Need more natural pausing")
            
            # Disfluency analysis
            if fluency_features.get('disfluency_count', 0) > 5:
                weaknesses.append("High number of speech disruptions")
        
        return strengths, weaknesses
    
    def _calculate_next_level_reqs(
        self, 
        current_level: CEFRLevel,
        dimensions: Dict[str, float],
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate requirements to advance to next CEFR level."""
        
        # Determine next level
        level_order = [CEFRLevel.A1, CEFRLevel.A2, CEFRLevel.B1, CEFRLevel.B2, CEFRLevel.C1, CEFRLevel.C2]
        current_index = level_order.index(current_level)
        
        if current_index >= len(level_order) - 1:
            return {}  # Already at highest level
        
        next_level = level_order[current_index + 1]
        next_criteria = self.criteria[next_level]
        
        requirements = {}
        for dimension in ['accuracy', 'fluency', 'prosody', 'stress']:
            current_score = dimensions.get(dimension, 0)
            target_min = next_criteria.accuracy_range[0]  # Use accuracy as base
            requirements[dimension] = max(0, target_min - current_score)
        
        return requirements
    
    def _generate_progression_tips(
        self, 
        level: CEFRLevel, 
        weaknesses: List[str],
        features: Dict[str, Any]
    ) -> List[str]:
        """Generate specific tips for progression to next level."""
        
        tips = []
        
        # Level-specific progression tips
        level_tips = {
            CEFRLevel.A1: [
                "Focus on mastering basic English sounds",
                "Practice individual phonemes before words",
                "Use audio models for imitation",
                "Practice with very short phrases"
            ],
            CEFRLevel.A2: [
                "Practice reading aloud slowly and clearly",
                "Work on common English word stress patterns",
                "Use minimal pairs for problem sounds",
                "Practice connected speech in simple sentences"
            ],
            CEFRLevel.B1: [
                "Practice shadowing native speakers",
                "Work on rhythm and flow in longer sentences",
                "Master English weak forms and reductions",
                "Practice intonation in questions and statements"
            ],
            CEFRLevel.B2: [
                "Focus on complex consonant clusters",
                "Practice stress in compound words",
                "Work on expressive delivery",
                "Practice with authentic materials"
            ],
            CEFRLevel.C1: [
                "Fine-tune subtle sound distinctions",
                "Master prosody in different contexts",
                "Practice professional pronunciation",
                "Work on reducing any remaining accent features"
            ]
        }
        
        tips.extend(level_tips.get(level, []))
        
        # Add weakness-specific tips
        for weakness in weaknesses[:2]:  # Limit to avoid overwhelm
            if "accuracy" in weakness:
                tips.append("Practice with phonetic exercises and audio dictionaries")
            elif "fluency" in weakness:
                tips.append("Use shadowing techniques with audio clips")
            elif "prosody" in weakness:
                tips.append("Practice sentence melody with nursery rhymes and songs")
            elif "stress" in weakness:
                tips.append("Learn English stress rules with word families")
        
        return tips[:5]  # Return top 5 tips
    
    def _get_default_assessment(self, dimensions: Dict[str, float]) -> CEFRAssessmentResult:
        """Get default assessment if analysis fails."""
        return CEFRAssessmentResult(
            cefr_level=CEFRLevel.B1,
            confidence=0.5,
            dimension_scores=dimensions,
            level_confidence={level: 0.16 for level in CEFRLevel},
            strengths=["Clear effort in pronunciation"],
            weaknesses=["Need more practice in multiple areas"],
            next_level_requirements={dim: 15.0 for dim in ['accuracy', 'fluency', 'prosody', 'stress']},
            progression_tips=["Continue regular practice", "Focus on specific improvement areas"]
        )
    
    def get_level_description(self, level: CEFRLevel) -> Dict:
        """Get detailed description of CEFR level for English pronunciation."""
        
        descriptions = {
            CEFRLevel.A1: {
                "title": "Beginner - Breakthrough",
                "overview": "Can pronounce very basic phrases and words with significant effort and L1 influence.",
                "capabilities": [
                    "Can pronounce isolated words when repeated slowly",
                    "Limited to very common English sounds",
                    "Strong accent from native language",
                    "Often requires repetition for understanding"
                ],
                "examples": ["hello", "goodbye", "yes", "no", "thank you"],
                "next_focus": [
                    "Mastering basic vowel sounds",
                    "Simple consonant articulation",
                    "Basic word stress patterns"
                ]
            },
            
            CEFRLevel.A2: {
                "title": "Elementary - Waystage", 
                "overview": "Can pronounce simple phrases and everyday expressions, still with noticeable foreign accent.",
                "capabilities": [
                    "Can read simple text aloud with some clarity",
                    "Basic sentence pronunciation recognizable",
                    "Can handle familiar expressions with repetition",
                    "Mispronunciations still frequent"
                ],
                "examples": ["how are you", "my name is", "where is", "I need"],
                "next_focus": [
                    "Consonant clusters",
                    "Vowel length distinctions",
                    "Sentence rhythm patterns"
                ]
            },
            
            CEFRLevel.B1: {
                "title": "Intermediate - Threshold",
                "overview": "Clear pronunciation of most everyday English, with some control over phonological patterns.",
                "capabilities": [
                    "Generally intelligible conversation",
                    "Good pronunciation of frequent words",
                    "Some automaticity in common expressions",
                    "Can read aloud with confidence"
                ],
                "examples": ["I would like to", "Can you help me", "That sounds interesting"],
                "next_focus": [
                    "Weak forms and reductions",
                    "Connected speech processes",
                    "Expressive intonation"
                ]
            },
            
            CEFRLevel.B2: {
                "title": "Upper Intermediate - Vantage",
                "overview": "Clear and accurate pronunciation with few errors apparent to listener.",
                "capabilities": [
                    "High intelligibility across topics",
                    "Good control of English rhythm features",
                    "Comfortable with most sound patterns",
                    "Can handle complex vocabulary pronunciation"
                ],
                "examples": ["I'd appreciate it if you could", "The situation is complicated"],
                "next_focus": [
                    "Subtle sound distinctions",
                    "Prosodic variation in formal speech",
                    "Technical vocabulary pronunciation"
                ]
            },
            
            CEFRLevel.C1: {
                "title": "Advanced - Effective Operational Proficiency",
                "overview": "Accurate and natural pronunciation comparable to educated native speakers.",
                "capabilities": [
                    "Native-like clarity and precision",
                    "Full control over phonological features",
                    "Appropriate prosody for different contexts",
                    "Minimal pronunciation restrictions"
                ],
                "examples": ["Professional presentations", "Academic lectures", "Business negotiations"],
                "next_focus": [
                    "Regional accent features",
                    "Style-specific pronunciation",
                    "Professional level refinement"
                ]
            },
            
            CEFRLevel.C2: {
                "title": "Proficient - Mastery",
                "overview": "Mastery of English pronunciation comparable to highly educated native speakers.",
                "capabilities": [
                    "Complete phonological control",
                    "Consistent accuracy in all contexts",
                    "Full prosodic native-like delivery",
                    "Professional-level pronunciation clarity"
                ],
                "examples": ["Any context with complete accuracy"],
                "next_focus": ["Pronunciation refinement complete"]
            }
        }
        
        return descriptions.get(level, {
            "title": "Level Description",
            "overview": "CEFR level assessment for English pronunciation",
            "capabilities": [],
            "examples": [],
            "next_focus": []
        })
