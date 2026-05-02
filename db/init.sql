-- CrackGraphAI Database Initialization Script
-- This script is automatically run by Docker Compose

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set default search path
SET search_path TO public;

-- Create schema comment
COMMENT ON SCHEMA public IS 'CrackGraphAI analysis results and audit logs';

-- Enable query statistics
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analyses_request_id ON analyses(request_id);
CREATE INDEX IF NOT EXISTS idx_analyses_si_score ON analyses(si_score);
CREATE INDEX IF NOT EXISTS idx_analyses_risk_level ON analyses(risk_level);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_damage_metrics_analysis_id ON damage_metrics(analysis_id);
CREATE INDEX IF NOT EXISTS idx_graph_features_analysis_id ON graph_features(analysis_id);
CREATE INDEX IF NOT EXISTS idx_post_processing_stats_analysis_id ON post_processing_stats(analysis_id);
CREATE INDEX IF NOT EXISTS idx_api_audit_logs_created_at ON api_audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_audit_logs_status_code ON api_audit_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_api_audit_logs_endpoint ON api_audit_logs(endpoint);

-- Grant permissions
GRANT CONNECT ON DATABASE crackgraphai TO postgres;
GRANT USAGE ON SCHEMA public TO postgres;
GRANT CREATE ON SCHEMA public TO postgres;
