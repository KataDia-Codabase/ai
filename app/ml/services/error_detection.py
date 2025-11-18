from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import structlog

from app.ml.models.wav2vec_trainer import (
    EnglishPronunciationAnalyzer,
    IndonesianPronunciationAnalyzer,
)

logger = structlog.get_logger()


@dataclass
class WordError:
    type: str
    expected: str
    actual: str
    position: int
    confidence: float


@dataclass
class ErrorDetectionResult:
    word_errors: List[WordError]
    summary: Dict[str, int]
    phoneme_insights: Optional[Dict] = None


class ErrorDetectionService:
    """Detect substitutions/deletions/insertions between reference and hypothesis."""

    def __init__(self) -> None:
        self.english_analyzer = EnglishPronunciationAnalyzer()
        self.indonesian_analyzer = IndonesianPronunciationAnalyzer()

    def analyze(
        self,
        reference_text: str,
        hypothesis_text: str,
        expected_phonemes: Optional[List[str]] = None,
        actual_phonemes: Optional[List[str]] = None,
        language: str = "en-US",
    ) -> ErrorDetectionResult:
        ref_words = reference_text.lower().split()
        hyp_words = hypothesis_text.lower().split()
        matcher = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)

        errors: List[WordError] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            if tag == "replace":
                span = max(i2 - i1, j2 - j1)
                for offset in range(span):
                    expected = ref_words[i1 + offset] if i1 + offset < i2 else ""
                    actual = hyp_words[j1 + offset] if j1 + offset < j2 else ""
                    errors.append(
                        WordError(
                            type="substitution",
                            expected=expected,
                            actual=actual,
                            position=i1 + offset,
                            confidence=0.6,
                        )
                    )
            elif tag == "delete":
                for idx in range(i1, i2):
                    errors.append(
                        WordError(
                            type="deletion",
                            expected=ref_words[idx],
                            actual="",
                            position=idx,
                            confidence=0.55,
                        )
                    )
            elif tag == "insert":
                for idx in range(j1, j2):
                    errors.append(
                        WordError(
                            type="insertion",
                            expected="",
                            actual=hyp_words[idx],
                            position=i1,
                            confidence=0.5,
                        )
                    )

        summary = {
            "total": len(errors),
            "substitution": len([e for e in errors if e.type == "substitution"]),
            "deletion": len([e for e in errors if e.type == "deletion"]),
            "insertion": len([e for e in errors if e.type == "insertion"]),
        }

        phoneme_insights = None
        if expected_phonemes and actual_phonemes:
            try:
                if language == "id-ID":
                    phoneme_insights = self.indonesian_analyzer.analyze_indonesian_errors(
                        expected_phonemes, actual_phonemes
                    )
                else:
                    phoneme_insights = self.english_analyzer.analyze_english_errors(
                        expected_phonemes, actual_phonemes
                    )
            except Exception as exc:  # pragma: no cover - analyzer failures are non-critical
                logger.warning("Phoneme analysis failed", error=str(exc))

        return ErrorDetectionResult(
            word_errors=errors,
            summary=summary,
            phoneme_insights=phoneme_insights,
        )
