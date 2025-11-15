-- KataDia AI - Database Schema
-- Database: katadia_ml
-- Description: MySQL database schema for KataDia pronunciation learning platform

-- Create database
CREATE DATABASE IF NOT EXISTS katadia_ml CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE katadia_ml;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100),
    email VARCHAR(255),
    native_language VARCHAR(50),
    target_language VARCHAR(50) DEFAULT 'english',
    cefr_level VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Audio recordings table
CREATE TABLE IF NOT EXISTS audio_recordings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    recording_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    file_size INT,
    duration FLOAT,
    format VARCHAR(20),
    sample_rate INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_recording_id (recording_id),
    INDEX idx_user_id (user_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Transcriptions table
CREATE TABLE IF NOT EXISTS transcriptions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transcription_id VARCHAR(255) UNIQUE NOT NULL,
    recording_id VARCHAR(255) NOT NULL,
    text TEXT,
    language VARCHAR(50),
    confidence FLOAT,
    word_timestamps JSON,
    processing_time FLOAT,
    stt_provider VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_transcription_id (transcription_id),
    INDEX idx_recording_id (recording_id),
    FOREIGN KEY (recording_id) REFERENCES audio_recordings(recording_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Pronunciation scores table
CREATE TABLE IF NOT EXISTS pronunciation_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    score_id VARCHAR(255) UNIQUE NOT NULL,
    recording_id VARCHAR(255) NOT NULL,
    transcription_id VARCHAR(255),
    overall_score FLOAT,
    accuracy_score FLOAT,
    fluency_score FLOAT,
    prosody_score FLOAT,
    stress_score FLOAT,
    phoneme_accuracy JSON,
    word_scores JSON,
    mispronounced_words JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_score_id (score_id),
    INDEX idx_recording_id (recording_id),
    INDEX idx_transcription_id (transcription_id),
    FOREIGN KEY (recording_id) REFERENCES audio_recordings(recording_id) ON DELETE CASCADE,
    FOREIGN KEY (transcription_id) REFERENCES transcriptions(transcription_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- CEFR assessments table
CREATE TABLE IF NOT EXISTS cefr_assessments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    score_id VARCHAR(255),
    cefr_level VARCHAR(5),
    confidence FLOAT,
    pronunciation_level VARCHAR(5),
    fluency_level VARCHAR(5),
    vocabulary_level VARCHAR(5),
    grammar_level VARCHAR(5),
    recommendations JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_assessment_id (assessment_id),
    INDEX idx_user_id (user_id),
    INDEX idx_score_id (score_id),
    INDEX idx_cefr_level (cefr_level),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (score_id) REFERENCES pronunciation_scores(score_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feedback_id VARCHAR(255) UNIQUE NOT NULL,
    score_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    feedback_text TEXT,
    feedback_audio TEXT,
    suggestions JSON,
    strengths JSON,
    areas_for_improvement JSON,
    language VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feedback_id (feedback_id),
    INDEX idx_score_id (score_id),
    INDEX idx_user_id (user_id),
    FOREIGN KEY (score_id) REFERENCES pronunciation_scores(score_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Learning sessions table
CREATE TABLE IF NOT EXISTS learning_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    duration INT,
    recordings_count INT DEFAULT 0,
    average_score FLOAT,
    progress_percentage FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_user_id (user_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- User progress tracking table
CREATE TABLE IF NOT EXISTS user_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    metric_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_metric_date (metric_date),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- API request logs table
CREATE TABLE IF NOT EXISTS api_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    user_id VARCHAR(255),
    status_code INT,
    response_time FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_request_id (request_id),
    INDEX idx_endpoint (endpoint),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System health metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    metric_unit VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_name (metric_name),
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create views for common queries

-- User statistics view
CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    u.user_id,
    u.username,
    u.cefr_level,
    COUNT(DISTINCT ar.id) as total_recordings,
    AVG(ps.overall_score) as average_score,
    MAX(ps.created_at) as last_activity,
    COUNT(DISTINCT ls.id) as total_sessions
FROM users u
LEFT JOIN audio_recordings ar ON u.user_id = ar.user_id
LEFT JOIN pronunciation_scores ps ON ar.recording_id = ps.recording_id
LEFT JOIN learning_sessions ls ON u.user_id = ls.user_id
GROUP BY u.user_id, u.username, u.cefr_level;

-- Daily activity summary view
CREATE OR REPLACE VIEW daily_activity AS
SELECT 
    DATE(created_at) as activity_date,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(*) as total_recordings,
    AVG(overall_score) as avg_score
FROM pronunciation_scores
GROUP BY DATE(created_at)
ORDER BY activity_date DESC;

-- Insert sample data for testing
INSERT INTO users (user_id, username, email, native_language, target_language, cefr_level) VALUES
('user_test_001', 'test_user', 'test@katadia.ai', 'indonesian', 'english', 'A2')
ON DUPLICATE KEY UPDATE username=VALUES(username);

-- Show table status
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    DATA_LENGTH,
    INDEX_LENGTH,
    CREATE_TIME
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'katadia_ml'
ORDER BY TABLE_NAME;
