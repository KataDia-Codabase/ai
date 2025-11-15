# KataDia AI - Database Setup Guide

## Prerequisites

### 1. Install MySQL

**Option A: Windows Installer (Recommended)**
```
1. Download MySQL Community Server: https://dev.mysql.com/downloads/mysql/
2. Run installer and select "Server only" or "Full"
3. Set root password (use same as in .env file)
4. Choose port 3306
5. Complete installation
```

**Option B: Using Chocolatey**
```powershell
choco install mysql
```

**Option C: Using Docker**
```bash
docker run -d --name katadia-mysql \
  -e MYSQL_ROOT_PASSWORD=password \
  -e MYSQL_DATABASE=katadia_ml \
  -p 3306:3306 \
  mysql:8.0
```

### 2. Verify MySQL Installation

```powershell
# Check if MySQL is running
Get-Service MySQL* | Select-Object Status, Name, DisplayName

# Or check with netstat
netstat -an | findstr :3306
```

### 3. Connect to MySQL

```powershell
# Using MySQL command line
mysql -u root -p

# Enter password when prompted
```

## Database Setup

### Method 1: Using SQL Script (Recommended)

```powershell
# Navigate to project directory
cd c:\Code\lomba\ai-KataDia\ai-ml-service

# Run schema creation script
mysql -u root -p < database/schema.sql

# Verify tables created
mysql -u root -p -e "USE katadia_ml; SHOW TABLES;"
```

### Method 2: Using Python Script

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install SQLAlchemy if not already installed
pip install sqlalchemy pymysql

# Run initialization script
python database/init_db.py create

# Test connection
python database/init_db.py test
```

### Method 3: From MySQL Command Line

```sql
-- Login to MySQL
mysql -u root -p

-- Create database
CREATE DATABASE IF NOT EXISTS katadia_ml CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Show databases
SHOW DATABASES;

-- Use database
USE katadia_ml;

-- Run schema file
SOURCE C:/Code/lomba/ai-KataDia/ai-ml-service/database/schema.sql;

-- Verify tables
SHOW TABLES;

-- Check table structure
DESCRIBE users;
```

## Environment Configuration

Update your `.env` file:

```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=katadia_ml
MYSQL_URI=mysql+pymysql://root:your_password_here@localhost:3306/katadia_ml
```

## Database Management Commands

### Create Tables
```powershell
python database/init_db.py create
```

### Reset Database (Drop and Recreate)
```powershell
python database/init_db.py reset
```

### Test Connection
```powershell
python database/init_db.py test
```

### Drop All Tables (Use with Caution!)
```powershell
python database/init_db.py drop
```

## Verify Database Setup

### Check Tables
```sql
USE katadia_ml;
SHOW TABLES;
```

Expected tables:
- users
- audio_recordings
- transcriptions
- pronunciation_scores
- cefr_assessments
- feedback
- learning_sessions
- user_progress
- api_logs
- system_metrics

### Check Sample Data
```sql
SELECT * FROM users;
```

Should show test user: `test_user` with user_id `user_test_001`

### Check Table Status
```sql
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    DATA_LENGTH,
    INDEX_LENGTH,
    CREATE_TIME
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'katadia_ml'
ORDER BY TABLE_NAME;
```

## Database Backup

### Create Backup
```powershell
mysqldump -u root -p katadia_ml > backup_$(Get-Date -Format "yyyyMMdd_HHmmss").sql
```

### Restore from Backup
```powershell
mysql -u root -p katadia_ml < backup_20251115_143000.sql
```

## Troubleshooting

### MySQL Service Not Running
```powershell
# Start MySQL service
net start MySQL80

# Or using Services GUI
services.msc
# Find MySQL80 and start it
```

### Cannot Connect to MySQL
```powershell
# Check if MySQL is listening
netstat -an | findstr :3306

# Check MySQL error log
# Location: C:\ProgramData\MySQL\MySQL Server 8.0\Data\*.err
```

### Access Denied Error
```sql
-- Reset root password
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
FLUSH PRIVILEGES;
```

### Table Already Exists
```sql
-- Drop database and recreate
DROP DATABASE IF EXISTS katadia_ml;
CREATE DATABASE katadia_ml CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## Database Schema Overview

```
katadia_ml/
├── users                    # User accounts and profiles
├── audio_recordings         # Audio file metadata
├── transcriptions          # STT results
├── pronunciation_scores    # Scoring results
├── cefr_assessments       # CEFR level assessments
├── feedback               # AI-generated feedback
├── learning_sessions      # User learning sessions
├── user_progress         # Progress tracking
├── api_logs             # API request logs
└── system_metrics       # System health metrics
```

## Next Steps

1. ✅ Install MySQL
2. ✅ Create database using schema.sql
3. ✅ Update .env with correct credentials
4. ✅ Test connection with init_db.py
5. ✅ Verify tables are created
6. ✅ Ready for application integration!

## Integration with FastAPI

Update `app/main.py` to initialize database on startup:

```python
from database.connection import init_db, check_db_connection

@app.on_event("startup")
async def startup_event():
    logger.info("KataDia AI ML Service starting up...")
    
    # Initialize database
    if check_db_connection():
        init_db()
        logger.info("Database initialized successfully")
    else:
        logger.error("Database connection failed - service may not work properly")
```

## Support

For issues or questions:
- Check MySQL error logs
- Verify credentials in .env
- Ensure MySQL service is running
- Check firewall settings for port 3306
