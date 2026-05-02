# Database Setup Summary

A production-ready database layer has been added to CrackGraphAI. No existing code has been modified.

## What Was Added

### New Files Created

```
db/
├── __init__.py                 # Package initialization
├── models.py                   # SQLAlchemy ORM models (5 tables)
├── database.py                 # Connection management & pooling
├── repository.py               # Data access layer (CRUD operations)
├── service.py                  # Business logic layer
├── api_integration.py          # FastAPI integration utilities
├── init_db.py                  # Database initialization script
├── init.sql                    # SQL initialization script
└── README.md                   # Comprehensive database documentation

docker-compose.db.yml           # PostgreSQL + pgAdmin setup
requirements-db.txt             # Database dependencies
DB_INTEGRATION_GUIDE.md         # Integration instructions
DATABASE_SETUP_SUMMARY.md       # This file
```

### Updated Files

```
.env.example                    # Added database configuration variables
```

## Database Schema

### 5 Tables Created

1. **analyses** (Main records)
   - request_id, si_score, risk_level, latency_seconds
   - input_filename, input_file_hash (for deduplication)
   - image paths (segmentation_mask, skeleton, keypoints, etc.)
   - Indexes: request_id, si_score, risk_level, created_at

2. **damage_metrics** (Damage breakdown)
   - density_damage, network_damage, complexity_damage, width_damage, total_damage
   - Foreign key to analyses

3. **graph_features** (Topology metrics)
   - total_crack_length, num_branches, num_endpoints, num_junctions
   - longest_path, graph_diameter, mean_node_degree, connectivity_score
   - Foreign key to analyses

4. **post_processing_stats** (Filtering statistics)
   - raw_pixels, filtered_pixels, final_pixels, filtering_applied
   - Foreign key to analyses

5. **api_audit_logs** (Request audit trail)
   - endpoint, method, status_code, response_time_ms
   - api_key_hash, ip_address, error_message
   - Indexes: created_at, status_code, endpoint

## Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements-db.txt
```

### 2. Start PostgreSQL
```bash
docker-compose -f docker-compose.db.yml up -d
```

### 3. Initialize Database
```bash
python -m db.init_db
```

## Configuration

Add to `.env`:
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/crackgraphai
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
DB_ECHO=false
```

## Integration with Existing API

The database layer is **completely optional** and **non-intrusive**:

- ✅ No changes to existing API endpoints
- ✅ No changes to inference logic
- ✅ No changes to model loading
- ✅ No changes to response format
- ✅ Backward compatible

### To Enable Database Saving

Add 3 lines to your `/predict` endpoint:

```python
# Save result to database
result_saver.save_result(
    request_id=request_id,
    result=result,
    input_filename=image.filename,
    input_file_size=len(content),
    input_file_bytes=content,
    api_key=api_key,
)
```

See `DB_INTEGRATION_GUIDE.md` for complete integration instructions.

## Architecture

```
FastAPI Endpoints
    ↓
API Integration Layer (AnalysisResultSaver, DatabaseMiddleware)
    ↓
Service Layer (AnalysisService, APIAuditService)
    ↓
Repository Layer (AnalysisRepository, DamageMetricsRepository, etc.)
    ↓
ORM Models (SQLAlchemy)
    ↓
Connection Pool (QueuePool with 10 connections)
    ↓
PostgreSQL Database
```

## Key Features

### Production-Ready
- ✅ Connection pooling (10 connections, 20 overflow)
- ✅ Connection recycling (3600s)
- ✅ Pre-ping enabled (detects stale connections)
- ✅ Comprehensive error handling
- ✅ Logging at all layers

### Performance
- ✅ Optimized indexes on all query columns
- ✅ Foreign key relationships with cascading deletes
- ✅ Efficient pagination support
- ✅ Aggregate statistics queries

### Security
- ✅ API key hashing (SHA256)
- ✅ Audit logging of all requests
- ✅ IP address tracking
- ✅ Error message sanitization

### Monitoring
- ✅ Health check endpoint
- ✅ Error rate calculation
- ✅ Database statistics
- ✅ Query performance tracking

## Usage Examples

### Save Analysis Result
```python
from db.database import get_db
from db.service import AnalysisService

session = get_db()
service = AnalysisService(session)

analysis_id = service.save_analysis_result(
    request_id="req-12345",
    result=inference_result,
    input_filename="crack_001.jpg",
    input_file_size=102400,
    api_key="your-api-key",
)

session.close()
```

### Retrieve Analysis
```python
result = service.get_analysis_with_details("req-12345")
```

### Get Statistics
```python
stats = service.get_statistics()
# Returns: total_analyses, avg_si_score, risk_distribution, avg_latency
```

### Log API Request
```python
from db.service import APIAuditService

audit_service = APIAuditService(session)
audit_service.log_request(
    endpoint="/predict",
    method="POST",
    status_code=200,
    response_time_ms=450.5,
    request_id="req-12345",
    api_key="your-api-key",
    ip_address="192.168.1.1",
)
```

## Database Operations

### Create Tables
```bash
python -m db.init_db
```

### Health Check
```python
from db.database import db_manager
db_manager.health_check()  # Returns True/False
```

### Cleanup Old Records
```python
service.cleanup_old_records(days=90)  # Delete analyses older than 90 days
```

## Monitoring & Troubleshooting

### Check Database Health
```bash
python -m db.init_db
```

### View Database Size
```sql
SELECT pg_size_pretty(pg_database_size('crackgraphai'));
```

### Check Active Connections
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'crackgraphai';
```

### View Slow Queries
```sql
SELECT query, calls, mean_time FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
```

## Backup & Recovery

### Backup
```bash
pg_dump -U postgres crackgraphai > backup_$(date +%Y%m%d).sql
```

### Restore
```bash
psql -U postgres crackgraphai < backup_20240101.sql
```

## Documentation

- **Full Database Docs**: `db/README.md`
- **Integration Guide**: `DB_INTEGRATION_GUIDE.md`
- **Models**: `db/models.py` (well-commented)
- **Repository**: `db/repository.py` (CRUD operations)
- **Service**: `db/service.py` (business logic)

## Dependencies Added

```
sqlalchemy==2.0.23          # ORM framework
psycopg2-binary==2.9.9      # PostgreSQL driver
alembic==1.12.1             # Database migrations (optional)
```

## Next Steps

1. **Install**: `pip install -r requirements-db.txt`
2. **Start DB**: `docker-compose -f docker-compose.db.yml up -d`
3. **Initialize**: `python -m db.init_db`
4. **Configure**: Update `.env` with DATABASE_URL
5. **Integrate**: Add 3 lines to your API (see guide)
6. **Test**: Run your API and verify data is saved
7. **Monitor**: Check database health and statistics

## Support

- Check `db/README.md` for comprehensive documentation
- Check `DB_INTEGRATION_GUIDE.md` for integration steps
- Run `python -m db.init_db` to verify setup
- Check logs for any errors

## Status

✅ **Production Ready**
- All tables created with proper indexes
- Connection pooling configured
- Error handling implemented
- Logging enabled
- Documentation complete

---

**Version**: 1.0.0  
**Status**: Production Ready  
**No Breaking Changes**: ✓ All existing code remains unchanged
