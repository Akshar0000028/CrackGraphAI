# CrackGraphAI Database Layer

Production-ready database integration for storing and retrieving crack analysis results.

## Overview

The database layer provides:
- **ORM Models**: SQLAlchemy models for all analysis data
- **Connection Management**: Production-grade connection pooling
- **Repository Pattern**: Clean abstraction for database operations
- **Service Layer**: Business logic for saving/retrieving results
- **API Integration**: Middleware and utilities for FastAPI integration
- **Audit Logging**: Complete audit trail of all API requests

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoints                        │
├─────────────────────────────────────────────────────────────┤
│                   API Integration Layer                      │
│  (AnalysisResultSaver, DatabaseMiddleware)                  │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                            │
│  (AnalysisService, APIAuditService)                         │
├─────────────────────────────────────────────────────────────┤
│                  Repository Layer                           │
│  (AnalysisRepository, DamageMetricsRepository, etc.)        │
├─────────────────────────────────────────────────────────────┤
│                   ORM Models                                │
│  (Analysis, DamageMetrics, GraphFeatures, etc.)             │
├─────────────────────────────────────────────────────────────┤
│              SQLAlchemy + Connection Pool                   │
├─────────────────────────────────────────────────────────────┤
│                  PostgreSQL Database                        │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

### Tables

#### `analyses` (Main analysis records)
- `id` (PK): Database primary key
- `request_id` (UNIQUE): Unique request identifier
- `api_key_hash`: Hash of API key for audit
- `input_filename`: Original filename
- `input_file_size`: File size in bytes
- `input_file_hash`: SHA256 hash for deduplication
- `si_score`: Structural Integrity score (0-1)
- `risk_level`: Risk classification
- `latency_seconds`: Processing time
- `from_cache`: Whether result was cached
- `model_version`: Model version string
- `segmentation_mask_path`: Path to segmentation mask image
- `raw_mask_path`: Path to raw mask image
- `skeleton_path`: Path to skeleton image
- `keypoints_overlay_path`: Path to keypoints overlay image
- `probability_map_path`: Path to probability map image
- `created_at` (INDEX): Timestamp
- `updated_at`: Last update timestamp

**Indexes**:
- `idx_request_id_created`: For fast lookup by request_id
- `idx_si_score_created`: For filtering by SI score
- `idx_risk_level_created`: For filtering by risk level

#### `damage_metrics` (Damage breakdown)
- `id` (PK): Primary key
- `analysis_id` (FK): Reference to analyses table
- `density_damage`: Crack coverage damage (0-1)
- `network_damage`: Junction severity damage (0-1)
- `complexity_damage`: Branching complexity damage (0-1)
- `width_damage`: Crack thickness damage (0-1)
- `total_damage`: Weighted total damage (0-1)
- `created_at`: Timestamp

#### `graph_features` (Topology metrics)
- `id` (PK): Primary key
- `analysis_id` (FK): Reference to analyses table
- `total_crack_length`: Total skeleton pixels
- `num_branches`: Number of branches
- `num_endpoints`: Number of endpoints
- `num_junctions`: Number of junctions
- `longest_path`: Longest path length
- `graph_diameter`: Graph diameter
- `mean_node_degree`: Average node degree
- `connectivity_score`: Connectivity metric (0-1)
- `created_at`: Timestamp

#### `post_processing_stats` (Filtering statistics)
- `id` (PK): Primary key
- `analysis_id` (FK): Reference to analyses table
- `raw_pixels`: Pixels before filtering
- `filtered_pixels`: Pixels removed by filtering
- `final_pixels`: Final crack pixels
- `filtering_applied`: Whether filtering was applied
- `created_at`: Timestamp

#### `api_audit_logs` (Request audit trail)
- `id` (PK): Primary key
- `request_id`: Reference to analysis request
- `api_key_hash`: Hash of API key used
- `endpoint`: API endpoint called
- `method`: HTTP method
- `status_code`: HTTP response code
- `response_time_ms`: Response time in milliseconds
- `error_message`: Error message if failed
- `ip_address`: Client IP address
- `created_at` (INDEX): Timestamp

**Indexes**:
- `idx_endpoint_created`: For filtering by endpoint
- `idx_status_created`: For filtering by status code

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements-db.txt
```

### 2. Configure Database

Update `.env` with your database connection:

```bash
# PostgreSQL (recommended for production)
DATABASE_URL=postgresql://user:password@localhost:5432/crackgraphai

# Connection pool settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
DB_ECHO=false
```

### 3. Initialize Database

```bash
python -m db.init_db
```

This will:
- Test database connectivity
- Create all tables
- Create all indexes
- Log success/failure

### 4. Integrate with API

In your FastAPI application:

```python
from db.database import db_manager, get_db
from db.api_integration import AnalysisResultSaver, DatabaseMiddleware

# Initialize database on startup
db_manager.create_tables()

# Create result saver
result_saver = AnalysisResultSaver(get_db)

# In your predict endpoint:
@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    # ... run inference ...
    
    # Save result to database
    analysis_id = result_saver.save_result(
        request_id=request_id,
        result=inference_result,
        input_filename=image.filename,
        input_file_size=len(content),
        input_file_bytes=content,
        api_key=api_key,
        model_version="2.0.0",
    )
    
    return inference_result
```

## Usage Examples

### Save Analysis Result

```python
from db.database import get_db
from db.service import AnalysisService

session = get_db()
service = AnalysisService(session)

analysis_id = service.save_analysis_result(
    request_id="req-12345",
    result={
        "si_score": 0.75,
        "risk_level": "Moderate",
        "latency_seconds": 0.45,
        "damage_metrics": {
            "density_damage": 0.3,
            "network_damage": 0.2,
            "complexity_damage": 0.15,
            "width_damage": 0.1,
            "total_damage": 0.25,
        },
        "graph_features": {
            "total_crack_length": 1500.0,
            "num_branches": 5,
            "endpoints": 10,
            "junctions": 3,
            "longest_path": 800.0,
            "graph_diameter": 900.0,
            "mean_node_degree": 2.1,
        },
        "post_processing": {
            "raw_pixels": 2000,
            "filtered_pixels": 100,
            "final_pixels": 1900,
            "filtering_applied": True,
        },
    },
    input_filename="crack_001.jpg",
    input_file_size=102400,
    api_key="your-api-key",
)

session.close()
```

### Retrieve Analysis Result

```python
from db.database import get_db
from db.service import AnalysisService

session = get_db()
service = AnalysisService(session)

result = service.get_analysis_with_details("req-12345")
print(result)

session.close()
```

### Get Statistics

```python
from db.database import get_db
from db.service import AnalysisService

session = get_db()
service = AnalysisService(session)

stats = service.get_statistics()
print(f"Total analyses: {stats['total_analyses']}")
print(f"Average SI score: {stats['avg_si_score']}")
print(f"Risk distribution: {stats['risk_distribution']}")

session.close()
```

### Log API Request

```python
from db.database import get_db
from db.service import APIAuditService

session = get_db()
service = APIAuditService(session)

service.log_request(
    endpoint="/predict",
    method="POST",
    status_code=200,
    response_time_ms=450.5,
    request_id="req-12345",
    api_key="your-api-key",
    ip_address="192.168.1.1",
)

session.close()
```

## Connection Pool Configuration

The database manager uses SQLAlchemy's QueuePool for production-grade connection management:

- **Pool Size**: 10 connections (configurable via `DB_POOL_SIZE`)
- **Max Overflow**: 20 additional connections (configurable via `DB_MAX_OVERFLOW`)
- **Pool Recycle**: 3600 seconds (configurable via `DB_POOL_RECYCLE`)
- **Pre-ping**: Enabled to detect stale connections

## Performance Considerations

### Indexes
All frequently queried columns are indexed:
- `request_id` for fast lookups
- `created_at` for time-range queries
- `si_score` and `risk_level` for filtering
- `status_code` for audit analysis

### Query Optimization
- Use `list_recent()` for paginated results
- Use `list_by_date_range()` for time-based queries
- Use `get_statistics()` for aggregate data
- Avoid loading base64 image data in list queries

### Cleanup
Old records can be automatically deleted:

```python
from db.database import get_db
from db.service import AnalysisService

session = get_db()
service = AnalysisService(session)

# Delete analyses older than 90 days
deleted_count = service.cleanup_old_records(days=90)
print(f"Deleted {deleted_count} old records")

session.close()
```

## Monitoring

### Health Check

```python
from db.database import db_manager

if db_manager.health_check():
    print("Database is healthy")
else:
    print("Database connection failed")
```

### Error Rate

```python
from db.database import get_db
from db.service import APIAuditService

session = get_db()
service = APIAuditService(session)

error_rate = service.get_error_rate(minutes=5)
print(f"Error rate (last 5 min): {error_rate:.2f}%")

session.close()
```

## Troubleshooting

### Connection Refused
- Ensure PostgreSQL is running
- Check `DATABASE_URL` is correct
- Verify firewall allows connection

### Table Already Exists
- Tables are created only if they don't exist
- Safe to run `init_db.py` multiple times

### Slow Queries
- Check indexes are created: `\d+ analyses` in psql
- Use `EXPLAIN ANALYZE` to profile queries
- Consider archiving old records

### Connection Pool Exhausted
- Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
- Check for connection leaks (ensure sessions are closed)
- Monitor active connections: `SELECT count(*) FROM pg_stat_activity;`

## Production Deployment

### Recommended Settings

```bash
# .env for production
DATABASE_URL=postgresql://user:secure_password@db.example.com:5432/crackgraphai
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_RECYCLE=1800
DB_POOL_PRE_PING=true
DB_ECHO=false
```

### Backup Strategy

```bash
# Daily backup
pg_dump -U postgres crackgraphai > backup_$(date +%Y%m%d).sql

# Restore from backup
psql -U postgres crackgraphai < backup_20240101.sql
```

### Monitoring Queries

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('crackgraphai'));

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'crackgraphai';

-- Check slow queries
SELECT query, calls, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```

## API Endpoints for Database Access

New endpoints can be added to retrieve historical data:

```python
@app.get("/api/v1/analyses/{request_id}")
async def get_analysis(request_id: str, token: str = Depends(verify_token)):
    """Get analysis result by request ID."""
    session = get_db()
    try:
        service = AnalysisService(session)
        result = service.get_analysis_with_details(request_id)
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return result
    finally:
        session.close()

@app.get("/api/v1/statistics")
async def get_stats(token: str = Depends(verify_token)):
    """Get aggregate statistics."""
    session = get_db()
    try:
        service = AnalysisService(session)
        return service.get_statistics()
    finally:
        session.close()
```

## Support

For issues or questions:
1. Check logs: `tail -f logs/crackgraphai.log`
2. Verify database connectivity: `python -m db.init_db`
3. Review PostgreSQL logs: `tail -f /var/log/postgresql/postgresql.log`
