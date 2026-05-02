"""
Repository layer for database operations.
Provides clean abstraction for CRUD operations on analysis results.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from db.models import (
    Analysis,
    DamageMetrics,
    GraphFeatures,
    PostProcessingStats,
    APIAuditLog,
)

logger = logging.getLogger(__name__)


class AnalysisRepository:
    """Repository for Analysis records."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        request_id: str,
        si_score: float,
        risk_level: str,
        latency_seconds: float,
        input_filename: Optional[str] = None,
        input_file_size: Optional[int] = None,
        input_file_hash: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        from_cache: bool = False,
        model_version: Optional[str] = None,
        segmentation_mask_path: Optional[str] = None,
        raw_mask_path: Optional[str] = None,
        skeleton_path: Optional[str] = None,
        keypoints_overlay_path: Optional[str] = None,
        probability_map_path: Optional[str] = None,
    ) -> Analysis:
        """Create a new analysis record."""
        analysis = Analysis(
            request_id=request_id,
            si_score=si_score,
            risk_level=risk_level,
            latency_seconds=latency_seconds,
            input_filename=input_filename,
            input_file_size=input_file_size,
            input_file_hash=input_file_hash,
            api_key_hash=api_key_hash,
            from_cache=from_cache,
            model_version=model_version,
            segmentation_mask_path=segmentation_mask_path,
            raw_mask_path=raw_mask_path,
            skeleton_path=skeleton_path,
            keypoints_overlay_path=keypoints_overlay_path,
            probability_map_path=probability_map_path,
        )
        self.session.add(analysis)
        self.session.commit()
        logger.info(f"Created analysis record: {request_id}")
        return analysis

    def get_by_request_id(self, request_id: str) -> Optional[Analysis]:
        """Get analysis by request ID."""
        return self.session.query(Analysis).filter(Analysis.request_id == request_id).first()

    def get_by_id(self, analysis_id: int) -> Optional[Analysis]:
        """Get analysis by database ID."""
        return self.session.query(Analysis).filter(Analysis.id == analysis_id).first()

    def get_by_file_hash(self, file_hash: str) -> Optional[Analysis]:
        """Get most recent analysis for a file (for deduplication)."""
        return (
            self.session.query(Analysis)
            .filter(Analysis.input_file_hash == file_hash)
            .order_by(desc(Analysis.created_at))
            .first()
        )

    def list_recent(self, limit: int = 100, offset: int = 0) -> List[Analysis]:
        """Get recent analyses."""
        return (
            self.session.query(Analysis)
            .order_by(desc(Analysis.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    def list_by_risk_level(self, risk_level: str, limit: int = 100) -> List[Analysis]:
        """Get analyses by risk level."""
        return (
            self.session.query(Analysis)
            .filter(Analysis.risk_level == risk_level)
            .order_by(desc(Analysis.created_at))
            .limit(limit)
            .all()
        )

    def list_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[Analysis]:
        """Get analyses within a date range."""
        return (
            self.session.query(Analysis)
            .filter(and_(Analysis.created_at >= start_date, Analysis.created_at <= end_date))
            .order_by(desc(Analysis.created_at))
            .limit(limit)
            .all()
        )

    def get_statistics(self) -> Dict:
        """Get aggregate statistics."""
        total = self.session.query(Analysis).count()
        avg_si = self.session.query(Analysis.si_score).count()

        if total == 0:
            return {
                "total_analyses": 0,
                "avg_si_score": 0.0,
                "risk_distribution": {},
                "avg_latency": 0.0,
            }

        from sqlalchemy import func

        avg_si = self.session.query(func.avg(Analysis.si_score)).scalar() or 0.0
        avg_latency = self.session.query(func.avg(Analysis.latency_seconds)).scalar() or 0.0

        risk_dist = (
            self.session.query(Analysis.risk_level, func.count(Analysis.id))
            .group_by(Analysis.risk_level)
            .all()
        )

        return {
            "total_analyses": total,
            "avg_si_score": round(float(avg_si), 4),
            "risk_distribution": {risk: count for risk, count in risk_dist},
            "avg_latency": round(float(avg_latency), 3),
        }

    def delete_older_than(self, days: int) -> int:
        """Delete analyses older than N days (for cleanup)."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        count = self.session.query(Analysis).filter(Analysis.created_at < cutoff_date).delete()
        self.session.commit()
        logger.info(f"Deleted {count} analyses older than {days} days")
        return count


class DamageMetricsRepository:
    """Repository for DamageMetrics records."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        analysis_id: int,
        density_damage: float,
        network_damage: float,
        complexity_damage: float,
        width_damage: float,
        total_damage: float,
    ) -> DamageMetrics:
        """Create damage metrics for an analysis."""
        metrics = DamageMetrics(
            analysis_id=analysis_id,
            density_damage=density_damage,
            network_damage=network_damage,
            complexity_damage=complexity_damage,
            width_damage=width_damage,
            total_damage=total_damage,
        )
        self.session.add(metrics)
        self.session.commit()
        return metrics

    def get_by_analysis_id(self, analysis_id: int) -> Optional[DamageMetrics]:
        """Get damage metrics for an analysis."""
        return self.session.query(DamageMetrics).filter(DamageMetrics.analysis_id == analysis_id).first()


class GraphFeaturesRepository:
    """Repository for GraphFeatures records."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        analysis_id: int,
        total_crack_length: float,
        num_branches: int,
        num_endpoints: int,
        num_junctions: int,
        longest_path: float,
        graph_diameter: float,
        mean_node_degree: float,
        connectivity_score: float,
    ) -> GraphFeatures:
        """Create graph features for an analysis."""
        features = GraphFeatures(
            analysis_id=analysis_id,
            total_crack_length=total_crack_length,
            num_branches=num_branches,
            num_endpoints=num_endpoints,
            num_junctions=num_junctions,
            longest_path=longest_path,
            graph_diameter=graph_diameter,
            mean_node_degree=mean_node_degree,
            connectivity_score=connectivity_score,
        )
        self.session.add(features)
        self.session.commit()
        return features

    def get_by_analysis_id(self, analysis_id: int) -> Optional[GraphFeatures]:
        """Get graph features for an analysis."""
        return self.session.query(GraphFeatures).filter(GraphFeatures.analysis_id == analysis_id).first()


class PostProcessingStatsRepository:
    """Repository for PostProcessingStats records."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        analysis_id: int,
        raw_pixels: int,
        filtered_pixels: int,
        final_pixels: int,
        filtering_applied: bool,
    ) -> PostProcessingStats:
        """Create post-processing stats for an analysis."""
        stats = PostProcessingStats(
            analysis_id=analysis_id,
            raw_pixels=raw_pixels,
            filtered_pixels=filtered_pixels,
            final_pixels=final_pixels,
            filtering_applied=filtering_applied,
        )
        self.session.add(stats)
        self.session.commit()
        return stats

    def get_by_analysis_id(self, analysis_id: int) -> Optional[PostProcessingStats]:
        """Get post-processing stats for an analysis."""
        return self.session.query(PostProcessingStats).filter(PostProcessingStats.analysis_id == analysis_id).first()


class APIAuditLogRepository:
    """Repository for API audit logs."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_id: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> APIAuditLog:
        """Create an audit log entry."""
        log = APIAuditLog(
            request_id=request_id,
            api_key_hash=api_key_hash,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error_message=error_message,
            ip_address=ip_address,
        )
        self.session.add(log)
        self.session.commit()
        return log

    def list_recent(self, limit: int = 100) -> List[APIAuditLog]:
        """Get recent audit logs."""
        return (
            self.session.query(APIAuditLog)
            .order_by(desc(APIAuditLog.created_at))
            .limit(limit)
            .all()
        )

    def get_error_rate(self, minutes: int = 5) -> float:
        """Get error rate in the last N minutes."""
        from sqlalchemy import func

        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        total = self.session.query(APIAuditLog).filter(APIAuditLog.created_at >= cutoff).count()
        errors = (
            self.session.query(APIAuditLog)
            .filter(and_(APIAuditLog.created_at >= cutoff, APIAuditLog.status_code >= 400))
            .count()
        )

        if total == 0:
            return 0.0
        return (errors / total) * 100
