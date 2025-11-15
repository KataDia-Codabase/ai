"""
Database initialization and management scripts
"""

import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.connection import engine, init_db, check_db_connection, get_db_stats
from database.models import Base
from app.core.logging import get_logger

logger = get_logger()


def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False


def drop_tables():
    """Drop all database tables (use with caution!)"""
    try:
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped")
        return True
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        return False


def reset_database():
    """Drop and recreate all tables"""
    logger.warning("Resetting database (drop and recreate all tables)...")
    if drop_tables():
        return create_tables()
    return False


def test_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    if check_db_connection():
        logger.info("Database connection successful")
        stats = get_db_stats()
        logger.info(f"Connection pool stats: {stats}")
        return True
    else:
        logger.error("Database connection failed")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management for KataDia AI')
    parser.add_argument('action', choices=['init', 'create', 'drop', 'reset', 'test'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'init' or args.action == 'create':
        create_tables()
    elif args.action == 'drop':
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == 'yes':
            drop_tables()
        else:
            logger.info("Operation cancelled")
    elif args.action == 'reset':
        confirm = input("Are you sure you want to reset the database? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
        else:
            logger.info("Operation cancelled")
    elif args.action == 'test':
        test_connection()


if __name__ == "__main__":
    main()
