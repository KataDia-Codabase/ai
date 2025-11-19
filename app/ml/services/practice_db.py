"""
Practice Database Service untuk tracking practice history dan recommendations.
"""

from typing import Dict, List, Optional
from datetime import datetime
import structlog
from app.ml.services.adaptive_learning import PracticeRecommendation, DifficultyLevel

logger = structlog.get_logger()


class PracticeDatabase:
    """In-memory practice database untuk tracking user practice history."""
    
    def __init__(self):
        # Structure: {user_id: {"recommendations": [...], "reviews": [...]}}
        self.user_practice_history = {}
        self.practice_items_cache = {}
        self._initialize_practice_items()
    
    def _initialize_practice_items(self):
        """Initialize practice items database."""
        self.practice_items_cache = {
            "beginner": {
                "accuracy": [
                    {"id": "acc_beg_1", "word": "hello", "difficulty": "beginner"},
                    {"id": "acc_beg_2", "word": "water", "difficulty": "beginner"},
                    {"id": "acc_beg_3", "word": "phone", "difficulty": "beginner"},
                ],
                "fluency": [
                    {"id": "flu_beg_1", "text": "I am fine", "difficulty": "beginner"},
                    {"id": "flu_beg_2", "text": "Nice to meet you", "difficulty": "beginner"},
                ],
                "prosody": [
                    {"id": "pro_beg_1", "text": "Really?", "difficulty": "beginner"},
                    {"id": "pro_beg_2", "text": "How are you?", "difficulty": "beginner"},
                ],
                "stress": [
                    {"id": "str_beg_1", "words": ["RECORD", "reCORD"], "difficulty": "beginner"},
                    {"id": "str_beg_2", "words": ["PRESENT", "preSENT"], "difficulty": "beginner"},
                ]
            },
            "elementary": {
                "accuracy": [
                    {"id": "acc_ele_1", "word": "unfortunately", "difficulty": "elementary"},
                    {"id": "acc_ele_2", "word": "vegetable", "difficulty": "elementary"},
                ],
                "fluency": [
                    {"id": "flu_ele_1", "text": "The weather is nice today", "difficulty": "elementary"},
                    {"id": "flu_ele_2", "text": "I like to read books", "difficulty": "elementary"},
                ],
                "prosody": [
                    {"id": "pro_ele_1", "text": "Do you like it?", "difficulty": "elementary"},
                    {"id": "pro_ele_2", "text": "That's surprising!", "difficulty": "elementary"},
                ],
                "stress": [
                    {"id": "str_ele_1", "words": ["EXPORT", "exPORT"], "difficulty": "elementary"},
                    {"id": "str_ele_2", "words": ["OBJECT", "obJECT"], "difficulty": "elementary"},
                ]
            },
            "intermediate": {
                "accuracy": [
                    {"id": "acc_int_1", "word": "pronunciation", "difficulty": "intermediate"},
                    {"id": "acc_int_2", "word": "specifically", "difficulty": "intermediate"},
                ],
                "fluency": [
                    {"id": "flu_int_1", "text": "The quick brown fox jumps over the lazy dog", "difficulty": "intermediate"},
                    {"id": "flu_int_2", "text": "I believe that learning languages is important", "difficulty": "intermediate"},
                ],
                "prosody": [
                    {"id": "pro_int_1", "text": "You're coming to the party, aren't you?", "difficulty": "intermediate"},
                    {"id": "pro_int_2", "text": "This is surprising, but true", "difficulty": "intermediate"},
                ],
                "stress": [
                    {"id": "str_int_1", "words": ["CONFLICT", "conFLICT"], "difficulty": "intermediate"},
                    {"id": "str_int_2", "words": ["INCREASE", "INcrease"], "difficulty": "intermediate"},
                ]
            },
            "upper_intermediate": {
                "accuracy": [
                    {"id": "acc_upi_1", "word": "miscellaneous", "difficulty": "upper_intermediate"},
                    {"id": "acc_upi_2", "word": "entrepreneur", "difficulty": "upper_intermediate"},
                ],
                "fluency": [
                    {"id": "flu_upi_1", "text": "The implications of technological advancement are profound and multifaceted", "difficulty": "upper_intermediate"},
                    {"id": "flu_upi_2", "text": "Nevertheless, we must consider the potential ramifications carefully", "difficulty": "upper_intermediate"},
                ],
                "prosody": [
                    {"id": "pro_upi_1", "text": "Although it seems unlikely, it's actually quite possible", "difficulty": "upper_intermediate"},
                    {"id": "pro_upi_2", "text": "However, upon further reflection, we must reconsider", "difficulty": "upper_intermediate"},
                ],
                "stress": [
                    {"id": "str_upi_1", "words": ["PHOTOGRAPHY", "photoGRAPHY"], "difficulty": "upper_intermediate"},
                    {"id": "str_upi_2", "words": ["CONTROVERSY", "CONtroversy"], "difficulty": "upper_intermediate"},
                ]
            },
            "advanced": {
                "accuracy": [
                    {"id": "acc_adv_1", "word": "serendipitously", "difficulty": "advanced"},
                    {"id": "acc_adv_2", "word": "ostentatious", "difficulty": "advanced"},
                ],
                "fluency": [
                    {"id": "flu_adv_1", "text": "The paradigmatic shift in contemporary discourse necessitates a reevaluation of fundamental assumptions", "difficulty": "advanced"},
                    {"id": "flu_adv_2", "text": "Notwithstanding the aforementioned considerations, one must acknowledge the verisimilitude of such propositions", "difficulty": "advanced"},
                ],
                "prosody": [
                    {"id": "pro_adv_1", "text": "Paradoxically, the more we know, the less certain we become", "difficulty": "advanced"},
                    {"id": "pro_adv_2", "text": "Nevertheless, we persist in our endeavors", "difficulty": "advanced"},
                ],
                "stress": [
                    {"id": "str_adv_1", "words": ["CHARACTERISTIC", "characterISTic"], "difficulty": "advanced"},
                    {"id": "str_adv_2", "words": ["INFRASTRUCTURE", "INfrastructure"], "difficulty": "advanced"},
                ]
            },
            "proficiency": {
                "accuracy": [
                    {"id": "acc_pro_1", "word": "defenestration", "difficulty": "proficiency"},
                    {"id": "acc_pro_2", "word": "onomatopoeia", "difficulty": "proficiency"},
                ],
                "fluency": [
                    {"id": "flu_pro_1", "text": "The multifarious ramifications of such perspicacious observations warrant exhaustive elucidation", "difficulty": "proficiency"},
                    {"id": "flu_pro_2", "text": "In light of such considerable empirical evidence, we may justifiably posit alternative hypotheses", "difficulty": "proficiency"},
                ],
                "prosody": [
                    {"id": "pro_pro_1", "text": "Nevertheless, upon meticulous scrutiny, one discerns subtle nuances", "difficulty": "proficiency"},
                    {"id": "pro_pro_2", "text": "Notwithstanding such perspicuous arguments, divergent interpretations persist", "difficulty": "proficiency"},
                ],
                "stress": [
                    {"id": "str_pro_1", "words": ["UNIVERSITY", "uniVERsity"], "difficulty": "proficiency"},
                    {"id": "str_pro_2", "words": ["DICTIONARY", "dicTIONary"], "difficulty": "proficiency"},
                ]
            }
        }
    
    async def save_recommendation(
        self,
        user_id: str,
        recommendation: PracticeRecommendation
    ) -> bool:
        """Save practice recommendation untuk user."""
        try:
            if user_id not in self.user_practice_history:
                self.user_practice_history[user_id] = {
                    "recommendations": [],
                    "reviews": [],
                    "last_updated": datetime.now()
                }
            
            self.user_practice_history[user_id]["recommendations"].append({
                "timestamp": datetime.now(),
                "difficulty": recommendation.difficulty_level.value,
                "focus_areas": recommendation.focus_areas,
                "session_duration": recommendation.session_duration_minutes,
                "next_review": recommendation.next_review_date,
                "rationale": recommendation.rationale
            })
            
            self.user_practice_history[user_id]["last_updated"] = datetime.now()
            
            logger.info(
                "Saved practice recommendation",
                user_id=user_id,
                difficulty=recommendation.difficulty_level.value
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
            return False
    
    async def get_practice_items(
        self,
        difficulty: str,
        focus_areas: List[str]
    ) -> List[Dict]:
        """Get practice items berdasarkan difficulty dan focus areas."""
        try:
            items = []
            difficulty_key = difficulty.replace(" ", "_")
            
            if difficulty_key not in self.practice_items_cache:
                difficulty_key = "intermediate"  # Default
            
            cache_items = self.practice_items_cache[difficulty_key]
            
            for focus_area in focus_areas:
                if focus_area in cache_items:
                    items.extend(cache_items[focus_area][:2])  # Top 2 items per area
                elif focus_area.startswith("error_"):
                    # Return accuracy items untuk error patterns
                    items.extend(cache_items.get("accuracy", [])[:1])
            
            # Return at least 3 items
            if not items:
                items = cache_items.get("accuracy", [])[:3]
            
            logger.info(
                "Retrieved practice items",
                difficulty=difficulty,
                focus_areas=focus_areas,
                item_count=len(items)
            )
            
            return items[:5]  # Max 5 items
            
        except Exception as e:
            logger.error(f"Failed to get practice items: {e}")
            return []
    
    async def get_user_history(self, user_id: str) -> Dict:
        """Get user practice history."""
        if user_id not in self.user_practice_history:
            return {
                "recommendations": [],
                "reviews": [],
                "total_sessions": 0
            }
        
        history = self.user_practice_history[user_id]
        
        return {
            "recommendations": history.get("recommendations", []),
            "reviews": history.get("reviews", []),
            "total_sessions": len(history.get("recommendations", [])),
            "last_updated": history.get("last_updated")
        }
    
    async def record_review(
        self,
        item_id: str,
        quality_score: float,
        user_id: str
    ) -> bool:
        """Record review session untuk spaced repetition tracking."""
        try:
            if user_id not in self.user_practice_history:
                self.user_practice_history[user_id] = {
                    "recommendations": [],
                    "reviews": [],
                    "last_updated": datetime.now()
                }
            
            self.user_practice_history[user_id]["reviews"].append({
                "item_id": item_id,
                "quality": quality_score,
                "timestamp": datetime.now()
            })
            
            logger.info(
                "Recorded review",
                user_id=user_id,
                item_id=item_id,
                quality=quality_score
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record review: {e}")
            return False


# Global instance
practice_db = PracticeDatabase()
