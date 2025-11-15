"""
SQLAlchemy ORM Models for KataDia AI
"""

from sqlalchemy import Column, Integer, String, Float, Text, JSON, ForeignKey, TIMESTAMP, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100))
    email = Column(String(255), index=True)
    native_language = Column(String(50))
    target_language = Column(String(50), default='english')
    cefr_level = Column(String(5))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    audio_recordings = relationship("AudioRecording", back_populates="user", cascade="all, delete-orphan")
    learning_sessions = relationship("LearningSession", back_populates="user", cascade="all, delete-orphan")
    cefr_assessments = relationship("CEFRAssessment", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")


class AudioRecording(Base):
    __tablename__ = "audio_recordings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recording_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    file_path = Column(String(500))
    file_size = Column(Integer)
    duration = Column(Float)
    format = Column(String(20))
    sample_rate = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audio_recordings")
    transcriptions = relationship("Transcription", back_populates="recording", cascade="all, delete-orphan")
    pronunciation_scores = relationship("PronunciationScore", back_populates="recording", cascade="all, delete-orphan")


class Transcription(Base):
    __tablename__ = "transcriptions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transcription_id = Column(String(255), unique=True, nullable=False, index=True)
    recording_id = Column(String(255), ForeignKey('audio_recordings.recording_id', ondelete='CASCADE'), nullable=False, index=True)
    text = Column(Text)
    language = Column(String(50))
    confidence = Column(Float)
    word_timestamps = Column(JSON)
    processing_time = Column(Float)
    stt_provider = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    recording = relationship("AudioRecording", back_populates="transcriptions")
    pronunciation_scores = relationship("PronunciationScore", back_populates="transcription")


class PronunciationScore(Base):
    __tablename__ = "pronunciation_scores"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    score_id = Column(String(255), unique=True, nullable=False, index=True)
    recording_id = Column(String(255), ForeignKey('audio_recordings.recording_id', ondelete='CASCADE'), nullable=False, index=True)
    transcription_id = Column(String(255), ForeignKey('transcriptions.transcription_id', ondelete='SET NULL'), index=True)
    overall_score = Column(Float)
    accuracy_score = Column(Float)
    fluency_score = Column(Float)
    prosody_score = Column(Float)
    stress_score = Column(Float)
    phoneme_accuracy = Column(JSON)
    word_scores = Column(JSON)
    mispronounced_words = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    recording = relationship("AudioRecording", back_populates="pronunciation_scores")
    transcription = relationship("Transcription", back_populates="pronunciation_scores")
    cefr_assessments = relationship("CEFRAssessment", back_populates="score")
    feedback = relationship("Feedback", back_populates="score", cascade="all, delete-orphan")


class CEFRAssessment(Base):
    __tablename__ = "cefr_assessments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    assessment_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    score_id = Column(String(255), ForeignKey('pronunciation_scores.score_id', ondelete='SET NULL'), index=True)
    cefr_level = Column(String(5), index=True)
    confidence = Column(Float)
    pronunciation_level = Column(String(5))
    fluency_level = Column(String(5))
    vocabulary_level = Column(String(5))
    grammar_level = Column(String(5))
    recommendations = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cefr_assessments")
    score = relationship("PronunciationScore", back_populates="cefr_assessments")


class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(255), unique=True, nullable=False, index=True)
    score_id = Column(String(255), ForeignKey('pronunciation_scores.score_id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    feedback_text = Column(Text)
    feedback_audio = Column(Text)
    suggestions = Column(JSON)
    strengths = Column(JSON)
    areas_for_improvement = Column(JSON)
    language = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    score = relationship("PronunciationScore", back_populates="feedback")
    user = relationship("User", back_populates="feedback")


class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    started_at = Column(TIMESTAMP, server_default=func.now())
    ended_at = Column(TIMESTAMP, nullable=True)
    duration = Column(Integer)
    recordings_count = Column(Integer, default=0)
    average_score = Column(Float)
    progress_percentage = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="learning_sessions")


class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    metric_date = Column(TIMESTAMP, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())


class APILog(Base):
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    endpoint = Column(String(255), index=True)
    method = Column(String(10))
    user_id = Column(String(255), index=True)
    status_code = Column(Integer)
    response_time = Column(Float)
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)


class SystemMetric(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), index=True)
    metric_value = Column(Float)
    metric_unit = Column(String(50))
    recorded_at = Column(TIMESTAMP, server_default=func.now(), index=True)
