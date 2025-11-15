"""
Gemini 2.0 Flash integration for personalized pronunlciation feedback.
Generates context-aware, detailed feedback for English pronunciation.
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import structlog
from app.core.config import settings
import json
import asyncio

logger = structlog.get_logger()

class GeminiFeedbackService:
    """Service for generating personalized feedback using Gemini 2.0 Flash."""
    
    def __init__(self):
        self.model = None
        self._setup_gemini()
        
        # System prompts for different feedback types
        self.system_prompts = {
            "pronunciation_scoring": self._get_pronunciation_prompt(),
            "fluency_improvement": self._get_fluency_prompt(),
            "prosody_development": self._get_prosody_prompt(),
            "stress_patterns": self._get_stress_prompt()
        }
    
    def _setup_gemini(self):
        """Initialize Gemini 2.0 Flash model."""
        try:
            if settings.OPENAI_API_KEY:  # Reuse existing API key field for Gemini
                genai.configure(api_key=settings.OPENAI_API_KEY)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("Gemini 2.0 Flash model initialized successfully")
            else:
                logger.warning("Gemini API key not provided, using template feedback")
        except Exception as e:
            logger.error(f"Failed to setup Gemini: {e}")
            self.model = None
    
    async def generate_feedback(
        self, 
        comprehensive_result: Dict,
        language: str = "en-US",
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate personalized pronunciation feedback using Gemini 2.0 Flash.
        
        Args:
            comprehensive_result: Results from enhanced pronunciation scoring
            language: Language code (should be en-US for this implementation)
            user_context: Optional user context (skill level, previous errors, etc.)
            
        Returns:
            Dict with detailed feedback in multiple categories
        """
        try:
            if not self.model:
                return self._generate_template_feedback(comprehensive_result)
            
            # Prepare feedback request
            feedback_request = self._prepare_feedback_request(
                comprehensive_result, language, user_context
            )
            
            # Generate feedback using Gemini
            gemini_response = await self._call_gemini(feedback_request)
            
            # Parse and structure feedback
            structured_feedback = self._parse_gemini_response(gemini_response)
            
            # Add English-specific insights
            structured_feedback = self._enhance_with_english_insights(
                structured_feedback, comprehensive_result
            )
            
            logger.info("Gemini feedback generated successfully")
            return structured_feedback
            
        except Exception as e:
            logger.error(f"Gemini feedback generation failed: {e}")
            return self._generate_template_feedback(comprehensive_result)
    
    def _prepare_feedback_request(
        self, 
        comprehensive_result: Dict, 
        language: str,
        user_context: Optional[Dict]
    ) -> str:
        """Prepare detailed prompt for Gemini."""
        
        # Extract key information
        overall_score = comprehensive_result.get('overall_score', 0)
        dimensions = comprehensive_result.get('dimensions', {})
        features = comprehensive_result.get('features', {})
        
        # Build prompt sections
        prompt_parts = [
            f"Generate detailed pronunciation feedback for a {language} learner.",
            "",
            f"Overall Score: {overall_score}/100",
            "",
            "Dimensional Scores:",
            f"- Accuracy: {dimensions.get('accuracy', 'N/A')}/100",
            f"- Fluency: {dimensions.get('fluency', 'N/A')}/100", 
            f"- Prosody: {dimensions.get('prosody', 'N/A')}/100",
            f"- Stress: {dimensions.get('stress', 'N/A')}/100",
            ""
        ]
        
        # Add feature details if available
        if features:
            prompt_parts.extend([
                "Detailed Analysis:",
                "",
                "Fluency Features:",
            ])
            
            # Safely extract fluency features (could be object or dict)
            fluency_data = features.get('fluency', {})
            if hasattr(fluency_data, '__dict__'):
                fluency_dict = fluency_data.__dict__
            elif isinstance(fluency_data, dict):
                fluency_dict = fluency_data
            else:
                fluency_dict = {}
            
            prompt_parts.extend([
                f"- Speech Rate: {fluency_dict.get('speech_rate', 'N/A')} WPM",
                f"- Pause Ratio: {fluency_dict.get('pause_ratio', 'N/A')}",
                f"- Disfluencies: {fluency_dict.get('disfluency_count', 'N/A')}",
                "",
                "Prosody Features:",
            ])
            
            # Safely extract prosody features
            prosody_data = features.get('prosody', {})
            if hasattr(prosody_data, '__dict__'):
                prosody_dict = prosody_data.__dict__
            elif isinstance(prosody_data, dict):
                prosody_dict = prosody_data
            else:
                prosody_dict = {}
            
            prompt_parts.extend([
                f"- Pitch Variation: {prosody_dict.get('pitch_std', 'N/A')} Hz",
                f"- Pitch Range: {prosody_dict.get('pitch_range', 'N/A')} Hz",
                f"- Emphasis Score: {prosody_dict.get('emphasis_score', 'N/A')}",
                "",
                "Stress Features:",
            ])
            
            # Safely extract stress features
            stress_data = features.get('stress', {})
            if hasattr(stress_data, '__dict__'):
                stress_dict = stress_data.__dict__
            elif isinstance(stress_data, dict):
                stress_dict = stress_data
            else:
                stress_dict = {}
            
            prompt_parts.extend([
                f"- Stress Pattern Accuracy: {stress_dict.get('strength_pattern_score', 'N/A')}",
                f"- Weak Form Usage: {stress_dict.get('weak_form_pronunciation', 'N/A')}",
                ""
            ])
        
        # Add detailed feedback from model if available
        if comprehensive_result.get('feedback'):
            feedback_by_dimension = comprehensive_result['feedback']
            prompt_parts.append("Specific Issues Found:")
            
            for dimension, feedback_data in feedback_by_dimension.items():
                if feedback_data.get('specific_areas'):
                    prompt_parts.append(f"\n{dimension.title()}:")
                    for issue in feedback_data['specific_areas']:
                        prompt_parts.append(f"- {issue}")
                    for recommendation in feedback_data['recommendations']:
                        prompt_parts.append(f"Recommendation: {recommendation}")
            prompt_parts.append("")
        
        # Add user context if available
        if user_context:
            prompt_parts.extend([
                "User Context:",
                f"- Skill Level: {user_context.get('skill_level', 'Unknown')}",
                f"- Native Language: {user_context.get('native_language', 'Unknown')}",
                f"- Persistent Issues: {user_context.get('persistent_issues', 'None')}",
                ""
            ])
        
        # Instructions for response format
        prompt_parts.extend([
            "Please provide feedback in the following JSON format:",
            "{",
            '  "overall_assessment": "Brief summary of performance",',
            '  "strengths": ["List of 2-3 strengths"],',
            '  "areas_to_focus": ["List of 2-3 specific areas"],',
            '  "detailed_feedback": {',
            '    "accuracy": {"analysis": "...", "recommendations": ["..."]},',
            '    "fluency": {"analysis": "...", "recommendations": ["..."]},',
            '    "prosody": {"analysis": "...", "recommendations": ["..."]},',
            '    "stress": {"analysis": "...", "recommendations": ["..."]}',
            '  },',
            '  "practice_suggestions": [',
            '    "Specific exercises or practice methods",',
            '    "Resources or materials to study"',
            '  ],',
            '  "next_steps": [',
            '    "Specific goals for next practice session",',
            '    "Areas to monitor closely"',
            '  ]',
            '}',
            "",
            "Make the feedback encouraging, specific, and actionable. Focus on English-specific pronunciation challenges. Avoid generic responses."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini 2.0 Flash API."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    def _parse_gemini_response(self, response: str) -> Dict:
        """Parse Gemini response into structured feedback."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            else:
                # Try to find JSON in the response
                if "{" in response:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_text = response[json_start:json_end]
                else:
                    json_text = response
            
            # Parse JSON
            feedback_data = json.loads(json_text)
            
            # Ensure all required fields are present
            required_fields = [
                "overall_assessment", "strengths", "areas_to_focus",
                "practice_suggestions", "next_steps"
            ]
            
            for field in required_fields:
                if field not in feedback_data:
                    feedback_data[field] = self._get_default_field_value(field)
            
            return feedback_data
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            # Extract text-based feedback if JSON parsing fails
            return self._extract_text_feedback(response)
    
    def _extract_text_feedback(self, response: str) -> Dict:
        """Extract feedback from text response when JSON parsing fails."""
        return {
            "overall_assessment": response[:200] + "..." if len(response) > 200 else response,
            "strengths": ["Pronunciation shows effort", "Good attempt at complex sounds"],
            "areas_to_focus": ["Practice word stress patterns", "Work on fluency"],
            "detailed_feedback": {
                "accuracy": {"analysis": "Focus on sound accuracy", "recommendations": ["Practice individual sounds"]},
                "fluency": {"analysis": "Work on speech flow", "recommendations": ["Practice with rhythm"]},
                "prosody": {"analysis": "Develop expressive speech", "recommendations": ["Vary pitch patterns"]},
                "stress": {"analysis": "Improve stress placement", "recommendations": ["Learn English stress rules"]}
            },
            "practice_suggestions": [
                "Practice with English pronunciation resources",
                "Focus on specific error patterns"
            ],
            "next_steps": [
                "Continue practicing daily",
                "Record and compare with native speakers"
            ]
        }
    
    def _get_default_field_value(self, field: str) -> any:
        """Get default values for feedback fields."""
        defaults = {
            "overall_assessment": "Good effort with room for improvement.",
            "strengths": ["Clear pronunciation", "Good effort"],
            "areas_to_focus": ["Continue practicing", "Focus on specific challenges"],
            "practice_suggestions": ["Regular practice recommended", "Use pronunciation resources"],
            "next_steps": ["Continue daily practice", "Focus on specific areas"]
        }
        return defaults.get(field, ["Default feedback"])
    
    def _enhance_with_english_insights(
        self, 
        feedback: Dict, 
        comprehensive_result: Dict
    ) -> Dict:
        """Add English-specific pronunciation insights."""
        
        # Add English-specific challenges
        english_insights = []
        
        dimensions = comprehensive_result.get('dimensions', {})
        
        # Identify specific English challenges
        if dimensions.get('stress', 0) < 70:
            english_insights.append({
                "challenge": "English word stress",
                "explanation": "English has variable stress patterns that differ from many other languages",
                "practice": "Practice noun-verb stress pairs (e.g., RECORD vs. reCORD)"
            })
        
        if dimensions.get('prosody', 0) < 70:
            english_insights.append({
                "challenge": "English intonation",
                "explanation": "English uses pitch variation to convey meaning and emotion",
                "practice": "Practice question intonation (rising pitch) vs statements (falling pitch)"
            })
        
        # Add to feedback
        if english_insights:
            feedback["english_specific_insights"] = english_insights
            feedback["practice_suggestions"].extend([
                insight["practice"] for insight in english_insights[:2]
            ])
        
        return feedback
    
    def _generate_template_feedback(self, comprehensive_result: Dict) -> Dict:
        """Generate template-based feedback when Gemini is unavailable."""
        overall_score = comprehensive_result.get('overall_score', 0)
        dimensions = comprehensive_result.get('dimensions', {})
        
        # Determine assessment level
        if overall_score >= 85:
            assessment = "Excellent pronunciation with strong clarity and naturalness."
            strengths = ["Clear articulation", "Good fluency", "Expressive delivery"]
        elif overall_score >= 70:
            assessment = "Good pronunciation with room for refinement in some areas."
            strengths = ["Generally clear speech", "Good structure"]
        elif overall_score >= 55:
            assessment = "Developing pronunciation with specific areas for improvement."
            strengths = ["Clear effort", "Progress evident"]
        else:
            assessment = "Pronunciation needs focused practice in multiple areas."
            strengths = ["Willingness to practice", "Foundation established"]
        
        # Identify areas for focus
        focus_areas = []
        if dimensions.get('accuracy', 0) < 70:
            focus_areas.append("Sound accuracy and phoneme production")
        if dimensions.get('fluency', 0) < 70:
            focus_areas.append("Speech flow and pacing")
        if dimensions.get('prosody', 0) < 70:
            focus_areas.append("Intonation and expressive delivery")
        if dimensions.get('stress', 0) < 70:
            focus_areas.append("Word stress patterns")
        
        return {
            "overall_assessment": assessment,
            "strengths": strengths,
            "areas_to_focus": focus_areas,
            "detailed_feedback": {
                "accuracy": {
                    "analysis": "Practice individual English sounds",
                    "recommendations": ["Focus on problematic phonemes", "Use minimal pairs practice"]
                },
                "fluency": {
                    "analysis": "Work on connected speech",
                    "recommendations": ["Practice reading aloud", "Use shadowing exercises"]
                },
                "prosody": {
                    "analysis": "Develop expressive patterns",
                    "recommendations": ["Practice melody of sentences", "Listen to native speakers"]
                },
                "stress": {
                    "analysis": "Improve stress placement",
                    "recommendations": ["Learn English stress rules", "Practice with stress patterns"]
                }
            },
            "practice_suggestions": [
                "Daily pronunciation practice with native speaker models",
                "Record and compare your pronunciation",
                "Focus on specific English consonant clusters and vowel reductions"
            ],
            "next_steps": [
                "Practice for 15-20 minutes daily",
                "Focus on identified weak areas",
                "Use pronunciation apps and resources"
            ],
            "note": "Feedback generated from templates - AI feedback temporarily unavailable"
        }
    
    def _get_system_prompt(self): 
        """Get system prompt for Gemini."""
        return """You are an expert English pronunciation coach and phonetician. 
        Your role is to provide personalized, encouraging, and actionable feedback 
        on English pronunciation. Consider the learner's potential background as a 
        non-native speaker of English. Be specific about English phonological 
        challenges. Always be encouraging and provide concrete improvement steps."""
    
    def _get_pronunciation_prompt(self):
        """Get specialized prompt for pronunciation scoring feedback."""
        return self._get_system_prompt() + """
        
        Focus on:
        - Sound accuracy (vowels, consonants)
        - Phonetic precision
        - Common L1 interference patterns
        - Specific English problematic sounds (TH, R, vowel reductions)
        """
    
    def _get_fluency_prompt(self):
        """Get specialized prompt for fluency feedback."""
        return self._get_system_prompt() + """
        
        Focus on:
        - Speech rate (ideal: 150-180 WPM)
        - Pause patterns and timing
        - Rhythm and flow
        - Connected speech processes
        - Appropriate pausing for meaning
        """
    
    def _get_prosody_prompt(self):
        """Get specialized prompt for prosody feedback."""
        return self._get_system_prompt() + """
        
        Focus on:
        - Pitch variation and intonation contours
        - Sentence melody (questions vs statements)
        - Emotional expression through voice
        - Appropriate stress and emphasis
        - Communicative effectiveness
        """
    
    def _get_stress_prompt(self):
        """Get specialized prompt for stress pattern feedback.""" 
        return self._get_system_prompt() + """
        
        Focus on:
        - Primary word stress placement
        - Sentence-level stress patterns
        - Weak forms and reductions
        - Content vs function word stress
        - English stress rules (nouns vs verbs, compounds)
        """
