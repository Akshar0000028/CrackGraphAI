"""
Database models for CrackGraphAI analysis results.
Production-ready SQLAlchemy ORM models with proper indexing and relationships.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    Index,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Analysis(Base):
    """Main analysis record for each crack detection prediction."""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    api_key_hash = Column(String(64), nullable=True, index=True)  # Hash of API key for audit
    input_filename = Column(String(255), nullable=True)
    input_file_size = Column(Integer, nullable=True)  # bytes
    input_file_hash = Column(String(64), nullable=True, index=True)  # SHA256 for deduplication

    # Core SI Score
    si_score = Column(Float, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)  # Low, Moderate, High, Critical, Failure Imminent

    # Processing metadata
    latency_seconds = Column(Float, nullable=False)
    from_cache = Column(Boolean, default=False)
    model_version = Column(String(50), nullable=True)

    # Image references (store paths instead of base64 blobs)
    segmentation_mask_path = Column(String(255), nullable=True)
    raw_mask_path = Column(String(255), nullable=True)
    skeleton_path = Column(String(255), nullable=True)
    keypoints_overlay_path = Column(String(255), nullable=True)
    probability_map_path = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    damage_metrics = relationship("DamageMetrics", back_populates="analysis", cascade="all, delete-orphan")
    graph_features = relationship("GraphFeatures", back_populates="analysis", cascade="all, delete-orphan")
    post_processing_stats = relationship("PostProcessingStats", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_request_id_created", "request_id", "created_at"),
        Index("idx_si_score_created", "si_score", "created_at"),
        Index("idx_risk_level_created", "risk_level", "created_at"),
    )


class DamageMetrics(Base):
    """Normalized damage scores for each analysis."""

    __tablename__ = "damage_metrics"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)

    density_damage = Column(Float, nullable=False)  # 0-1: crack coverage
    network_damage = Column(Float, nullable=False)  # 0-1: junction severity
    complexity_damage = Column(Float, nullable=False)  # 0-1: branching complexity
    width_damage = Column(Float, nullable=False)  # 0-1: crack thickness
    total_damage = Column(Float, nullable=False)  # 0-1: weighted combination

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    analysis = relationship("Analysis", back_populates="damage_metrics")


class GraphFeatures(Base):
    """Structural topology features extracted from crack graph."""

    __tablename__ = "graph_features"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)

    total_crack_length = Column(Float, nullable=False)  # pixels
    num_branches = Column(Integer, nullable=False)
    num_endpoints = Column(Integer, nullable=False)
    num_junctions = Column(Integer, nullable=False)
    longest_path = Column(Float, nullable=False)  # pixels
    graph_diameter = Column(Float, nullable=False)  # pixels
    mean_node_degree = Column(Float, nullable=False)
    connectivity_score = Column(Float, nullable=False)  # 0-1: how interconnected

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    analysis = relationship("Analysis", back_populates="graph_features")


class PostProcessingStats(Base):
    """Post-processing filtering statistics."""

    __tablename__ = "post_processing_stats"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)

    raw_pixels = Column(Integer, nullable=False)
    filtered_pixels = Column(Integer, nullable=False)
    final_pixels = Column(Integer, nullable=False)
    filtering_applied = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    analysis = relationship("Analysis", back_populates="post_processing_stats")


class APIAuditLog(Base):
    """Audit log for all API requests."""

    __tablename__ = "api_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), nullable=True, index=True)
    api_key_hash = Column(String(64), nullable=True, index=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False, index=True)
    response_time_ms = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_endpoint_created", "endpoint", "created_at"),
        Index("idx_status_created", "status_code", "created_at"),
    )
