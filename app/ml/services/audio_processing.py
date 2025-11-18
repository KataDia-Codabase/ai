import asyncio
from dataclasses import dataclass, field
from typing import Dict

import librosa
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class ProcessedAudio:
    """Container for normalized/optimized audio ready for inference."""

    waveform: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, bool] = field(default_factory=dict)


class AudioPreprocessor:
    """Lightweight audio preprocessing optimized for mobile/origin constraints."""

    def __init__(self, target_sample_rate: int = 16000, mobile_duration_limit: float = 12.0):
        self.target_sample_rate = target_sample_rate
        self.mobile_duration_limit = mobile_duration_limit

    async def process(self, audio_path: str) -> ProcessedAudio:
        """Load, normalize, and trim audio for downstream models."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_sync, audio_path)

    def _process_sync(self, audio_path: str) -> ProcessedAudio:
        y, sr = librosa.load(audio_path, sr=None)
        metadata: Dict[str, bool] = {}

        # Resample to the model's preferred sample rate.
        if sr != self.target_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
            metadata["resampled"] = True

        # Remove leading/trailing silence for faster inference.
        y, _ = librosa.effects.trim(y, top_db=25)
        metadata["trimmed_silence"] = True

        # Normalize amplitude and apply light pre-emphasis to highlight consonants.
        y = librosa.util.normalize(y)
        y = librosa.effects.preemphasis(y)
        metadata["normalized"] = True

        # Limit clip length for mobile latency expectations.
        duration = len(y) / sr if sr else 0
        if duration > self.mobile_duration_limit:
            max_samples = int(self.mobile_duration_limit * sr)
            y = y[:max_samples]
            duration = len(y) / sr
            metadata["trimmed_for_mobile"] = True

        # Soft clipping to avoid saturation when re-encoding.
        y = np.clip(y, -1.0, 1.0)
        metadata["clipped"] = True

        return ProcessedAudio(
            waveform=y.astype(np.float32),
            sample_rate=sr,
            duration=duration,
            metadata=metadata,
        )
