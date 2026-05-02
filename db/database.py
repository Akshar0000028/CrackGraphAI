"""
Database connection and session management.
Production-ready with connection pooling and proper lifecycle management.
"""

import logging
import os
from typing import Generator

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from db.models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/crackgraphai"
        )
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.pool_pre_ping = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"


class DatabaseManager:
    """Manages database connection and session lifecycle."""

    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._initialize()

    def _initialize(self):
        """Initialize database engine and session factory."""
        config = DatabaseConfig()

        logger.info(f"Initializing database: {config.database_url.split('@')[1] if '@' in config.database_url else 'SQLite'}")

        # Create engine with connection pooling
        self._engine = create_engine(
            config.database_url,
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=config.pool_pre_ping,
            echo=config.echo,
            connect_args={
                "connect_timeout": 10,
                "application_name": "crackgraphai",
            } if "postgresql" in config.database_url else {},
        )

        # Set up event listeners for connection pool
        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Configure connection on creation."""
            if "postgresql" in config.database_url:
                dbapi_conn.set_isolation_level(0)  # autocommit mode

        @event.listens_for(self._engine, "pool_connect")
        def receive_pool_connect(dbapi_conn, connection_record):
            """Log pool connections in debug mode."""
            if config.echo:
                logger.debug("New connection acquired from pool")

        # Create session factory
        self._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine,
        )

        logger.info("Database initialized successfully")

    def create_tables(self):
        """Create all tables in the database."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self._engine)
        logger.warning("Database tables dropped")

    def get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            self._initialize()
        return self._session_factory()

    def get_engine(self):
        """Get the database engine."""
        if self._engine is None:
            self._initialize()
        return self._engine

    def close(self):
        """Close all connections in the pool."""
        if self._engine is not None:
            self._engine.dispose()
            logger.info("Database connections closed")

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection for FastAPI endpoints."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()
