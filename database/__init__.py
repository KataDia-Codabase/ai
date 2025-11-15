"""
Package initialization for database module
"""

from database.connection import get_db, init_db, check_db_connection, get_db_stats, DatabaseSession
from database.models import (
    User,
    AudioRecording,
    Transcription,
    PronunciationScore,
    CEFRAssessment,
    Feedback,
    LearningSession,
    UserProgress,
    APILog,
    SystemMetric
)

__all__ = [
    'get_db',
    'init_db',
    'check_db_connection',
    'get_db_stats',
    'DatabaseSession',
    'User',
    'AudioRecording',
    'Transcription',
    'PronunciationScore',
    'CEFRAssessment',
    'Feedback',
    'LearningSession',
    'UserProgress',
    'APILog',
    'SystemMetric'
]
