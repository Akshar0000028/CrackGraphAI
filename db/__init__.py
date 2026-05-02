"""
Database module for CrackGraphAI.
Provides ORM models, connection management, and repository layer.
"""

from db.database import DatabaseManager, db_manager, get_db
from db.models import Base, Analysis, DamageMetrics, GraphFeatures, PostProcessingStats, APIAuditLog
from db.repository import (
    AnalysisRepository,
    DamageMetricsRepository,
    GraphFeaturesRepository,
    PostProcessingStatsRepository,
    APIAuditLogRepository,
)
from db.service import AnalysisService, APIAuditService

__all__ = [
    "DatabaseManager",
    "db_manager",
    "get_db",
    "Base",
    "Analysis",
    "DamageMetrics",
    "GraphFeatures",
    "PostProcessingStats",
    "APIAuditLog",
    "AnalysisRepository",
    "DamageMetricsRepository",
    "GraphFeaturesRepository",
    "PostProcessingStatsRepository",
    "APIAuditLogRepository",
    "AnalysisService",
    "APIAuditService",
]
