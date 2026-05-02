"""
Database initialization script.
Run this once to create all tables and indexes.

Usage:
    python -m db.init_db
"""

import logging
import sys

from db.database import db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Initialize database."""
    try:
        logger.info("Starting database initialization...")

        # Check connectivity
        if not db_manager.health_check():
            logger.error("Database health check failed. Ensure DATABASE_URL is correct.")
            sys.exit(1)

        # Create tables
        db_manager.create_tables()

        logger.info("✓ Database initialization completed successfully")
        logger.info("Tables created:")
        logger.info("  - analyses")
        logger.info("  - damage_metrics")
        logger.info("  - graph_features")
        logger.info("  - post_processing_stats")
        logger.info("  - api_audit_logs")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
