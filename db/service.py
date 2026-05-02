"""
Service layer for database operations.
Handles business logic for storing and retrieving analysis results.
"""

import hashlib
import logging
from typing import Dict, Optional

from sqlalchemy.orm import Session

from db.repository import (
    AnalysisRepository,
    DamageMetricsRepository,
    GraphFeaturesRepository,
    PostProcessingStatsRepository,
    APIAuditLogRepository,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for managing analysis records and related data."""

    def __init__(self, session: Session):
        self.session = session
        self.analysis_repo = AnalysisRepository(session)
        self.damage_repo = DamageMetricsRepository(session)
        self.features_repo = GraphFeaturesRepository(session)
        self.stats_repo = PostProcessingStatsRepository(session)

    def save_analysis_result(
        self,
        request_id: str,
        result: Dict,
        input_filename: Optional[str] = None,
        input_file_size: Optional[int] = None,
        input_file_bytes: Optional[bytes] = None,
        api_key: Optional[str] = None,
        model_version: Optional[str] = None,
        image_output_dir: Optional[str] = None,
    ) -> int:
        """
        Save complete analysis result to database.

        Args:
            request_id: Unique request identifier
            result: API response dict with all analysis data
            input_filename: Original filename
            input_file_size: File size in bytes
            input_file_bytes: Raw file bytes (for hashing)
            api_key: API key used (for audit)
            model_version: Model version string
            image_output_dir: Directory where images are saved

        Returns:
            analysis_id (database primary key)
        """
        try:
            # Hash API key for audit trail
            api_key_hash = None
            if api_key:
                api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Hash input file for deduplication
            input_file_hash = None
            if input_file_bytes:
                input_file_hash = hashlib.sha256(input_file_bytes).hexdigest()

            # Build image paths
            segmentation_mask_path = None
            raw_mask_path = None
            skeleton_path = None
            keypoints_overlay_path = None
            probability_map_path = None

            if image_output_dir:
                segmentation_mask_path = f"{image_output_dir}/{request_id}_segmentation.png"
                raw_mask_path = f"{image_output_dir}/{request_id}_raw_mask.png"
                skeleton_path = f"{image_output_dir}/{request_id}_skeleton.png"
                keypoints_overlay_path = f"{image_output_dir}/{request_id}_keypoints.png"
                probability_map_path = f"{image_output_dir}/{request_id}_probability.png"

            # Create main analysis record
            analysis = self.analysis_repo.create(
                request_id=request_id,
                si_score=result.get("si_score", 0.0),
                risk_level=result.get("risk_level", "Unknown"),
                latency_seconds=result.get("latency_seconds", 0.0),
                input_filename=input_filename,
                input_file_size=input_file_size,
                input_file_hash=input_file_hash,
                api_key_hash=api_key_hash,
                from_cache=result.get("from_cache", False),
                model_version=model_version,
                segmentation_mask_path=segmentation_mask_path,
                raw_mask_path=raw_mask_path,
                skeleton_path=skeleton_path,
                keypoints_overlay_path=keypoints_overlay_path,
                probability_map_path=probability_map_path,
            )

            # Save damage metrics
            damage = result.get("damage_metrics", {})
            self.damage_repo.create(
                analysis_id=analysis.id,
                density_damage=damage.get("density_damage", 0.0),
                network_damage=damage.get("network_damage", 0.0),
                complexity_damage=damage.get("complexity_damage", 0.0),
                width_damage=damage.get("width_damage", 0.0),
                total_damage=damage.get("total_damage", 0.0),
            )

            # Save graph features
            features = result.get("graph_features", {})
            self.features_repo.create(
                analysis_id=analysis.id,
                total_crack_length=features.get("total_crack_length", 0.0),
                num_branches=features.get("num_branches", 0),
                num_endpoints=features.get("endpoints", 0),
                num_junctions=features.get("junctions", 0),
                longest_path=features.get("longest_path", 0.0),
                graph_diameter=features.get("graph_diameter", 0.0),
                mean_node_degree=features.get("mean_node_degree", 0.0),
                connectivity_score=result.get("connectivity_score", 0.0),
            )

            # Save post-processing stats
            post_proc = result.get("post_processing", {})
            self.stats_repo.create(
                analysis_id=analysis.id,
                raw_pixels=post_proc.get("raw_pixels", 0),
                filtered_pixels=post_proc.get("filtered_pixels", 0),
                final_pixels=post_proc.get("final_pixels", 0),
                filtering_applied=post_proc.get("filtering_applied", False),
            )

            logger.info(f"Saved analysis result to database: {request_id} (ID: {analysis.id})")
            return analysis.id

        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}", exc_info=True)
            raise

    def get_analysis_with_details(self, request_id: str) -> Optional[Dict]:
        """Get complete analysis record with all related data."""
        try:
            analysis = self.analysis_repo.get_by_request_id(request_id)
            if not analysis:
                return None

            damage = self.damage_repo.get_by_analysis_id(analysis.id)
            features = self.features_repo.get_by_analysis_id(analysis.id)
            stats = self.stats_repo.get_by_analysis_id(analysis.id)

            return {
                "id": analysis.id,
                "request_id": analysis.request_id,
                "si_score": analysis.si_score,
                "risk_level": analysis.risk_level,
                "latency_seconds": analysis.latency_seconds,
                "input_filename": analysis.input_filename,
                "input_file_size": analysis.input_file_size,
                "from_cache": analysis.from_cache,
                "model_version": analysis.model_version,
                "created_at": analysis.created_at.isoformat(),
                "damage_metrics": {
                    "density_damage": damage.density_damage if damage else 0.0,
                    "network_damage": damage.network_damage if damage else 0.0,
                    "complexity_damage": damage.complexity_damage if damage else 0.0,
                    "width_damage": damage.width_damage if damage else 0.0,
                    "total_damage": damage.total_damage if damage else 0.0,
                } if damage else {},
                "graph_features": {
                    "total_crack_length": features.total_crack_length if features else 0.0,
                    "num_branches": features.num_branches if features else 0,
                    "num_endpoints": features.num_endpoints if features else 0,
                    "num_junctions": features.num_junctions if features else 0,
                    "longest_path": features.longest_path if features else 0.0,
                    "graph_diameter": features.graph_diameter if features else 0.0,
                    "mean_node_degree": features.mean_node_degree if features else 0.0,
                    "connectivity_score": features.connectivity_score if features else 0.0,
                } if features else {},
                "post_processing": {
                    "raw_pixels": stats.raw_pixels if stats else 0,
                    "filtered_pixels": stats.filtered_pixels if stats else 0,
                    "final_pixels": stats.final_pixels if stats else 0,
                    "filtering_applied": stats.filtering_applied if stats else False,
                } if stats else {},
                "image_paths": {
                    "segmentation_mask": analysis.segmentation_mask_path,
                    "raw_mask": analysis.raw_mask_path,
                    "skeleton": analysis.skeleton_path,
                    "keypoints_overlay": analysis.keypoints_overlay_path,
                    "probability_map": analysis.probability_map_path,
                },
            }
        except Exception as e:
            logger.error(f"Failed to retrieve analysis: {e}", exc_info=True)
            return None

    def get_statistics(self) -> Dict:
        """Get aggregate statistics."""
        return self.analysis_repo.get_statistics()

    def cleanup_old_records(self, days: int = 90) -> int:
        """Delete analyses older than N days."""
        return self.analysis_repo.delete_older_than(days)


class APIAuditService:
    """Service for API audit logging."""

    def __init__(self, session: Session):
        self.session = session
        self.audit_repo = APIAuditLogRepository(session)

    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_id: Optional[str] = None,
        api_key: Optional[str] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log an API request."""
        try:
            api_key_hash = None
            if api_key:
                api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            self.audit_repo.create(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                request_id=request_id,
                api_key_hash=api_key_hash,
                error_message=error_message,
                ip_address=ip_address,
            )
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")

    def get_error_rate(self, minutes: int = 5) -> float:
        """Get error rate in the last N minutes."""
        return self.audit_repo.get_error_rate(minutes)
