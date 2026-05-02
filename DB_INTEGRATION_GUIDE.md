# Database Integration Guide for CrackGraphAI

This guide explains how to integrate the production-ready database layer into your existing CrackGraphAI API.

## Quick Start

### 1. Install Database Dependencies

```bash
pip install -r requirements-db.txt
```

### 2. Start PostgreSQL

Using Docker Compose (recommended):

```bash
docker-compose -f docker-compose.db.yml up -d
```

Or manually:
```bash
# Ensure PostgreSQL is running on localhost:5432
# Create database: createdb crackgraphai
```

### 3. Initialize Database

```bash
python -m db.init_db
```

### 4. Update .env

```bash
# Add to your .env file
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/crackgraphai
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
DB_ECHO=false
```

## Integration with Production API

### Option A: Minimal Integration (Recommended)

Add database saving to your existing `api/production_main.py`:

```python
# At the top of production_main.py, add imports:
from db.database import db_manager, get_db
from db.api_integration import AnalysisResultSaver, extract_api_key_from_request, get_client_ip
from db.service import APIAuditService

# Initialize database manager in lifespan:
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Starting CrackGraphAI Production API...")
    
    try:
        # Initialize database
        db_manager.create_tables()
        logger.info("Database initialized")
        
        # ... rest of existing startup code ...
        
        yield
    finally:
        # ... existing shutdown code ...
        db_manager.close()

# Create result saver as global
result_saver = AnalysisResultSaver(get_db)

# Modify the /predict endpoint to save results:
@app.post("/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    threshold: Optional[float] = None,
    return_uncertainty: bool = True,
    token: str = Depends(verify_token),
):
    """Single image prediction endpoint with database integration."""
    
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )
    
    # ... existing validation code ...
    
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    content = await image.read()
    
    # ... existing inference code ...
    
    try:
        result = await service.infer_with_protection(content, request_id, params)
        
        # SAVE TO DATABASE (new)
        api_key = extract_api_key_from_request(request)
        result_saver.save_result(
            request_id=request_id,
            result=result,
            input_filename=image.filename,
            input_file_size=len(content),
            input_file_bytes=content,
            api_key=api_key,
            model_version="2.0.0",
        )
        
        # LOG API REQUEST (new)
        session = get_db()
        try:
            audit_service = APIAuditService(session)
            audit_service.log_request(
                endpoint="/predict",
                method="POST",
                status_code=200,
                response_time_ms=(time.time() - start_time) * 1000,
                request_id=request_id,
                api_key=api_key,
                ip_address=get_client_ip(request),
            )
        finally:
            session.close()
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException as e:
        # Log error
        session = get_db()
        try:
            audit_service = APIAuditService(session)
            audit_service.log_request(
                endpoint="/predict",
                method="POST",
                status_code=e.status_code,
                response_time_ms=(time.time() - start_time) * 1000,
                request_id=request_id,
                api_key=extract_api_key_from_request(request),
                error_message=e.detail,
                ip_address=get_client_ip(request),
            )
        finally:
            session.close()
        raise
```

### Option B: Full Integration with New Endpoints

Add new endpoints to retrieve historical data:

```python
@app.get("/api/v1/analyses/{request_id}")
async def get_analysis(
    request_id: str,
    token: str = Depends(verify_token),
):
    """Retrieve analysis result by request ID."""
    result = result_saver.get_result(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/api/v1/statistics")
async def get_statistics(token: str = Depends(verify_token)):
    """Get aggregate statistics."""
    return result_saver.get_statistics()


@app.get("/api/v1/analyses")
async def list_analyses(
    limit: int = 100,
    offset: int = 0,
    risk_level: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """List recent analyses with optional filtering."""
    session = get_db()
    try:
        from db.repository import AnalysisRepository
        repo = AnalysisRepository(session)
        
        if risk_level:
            analyses = repo.list_by_risk_level(risk_level, limit=limit)
        else:
            analyses = repo.list_recent(limit=limit, offset=offset)
        
        return [
            {
                "id": a.id,
                "request_id": a.request_id,
                "si_score": a.si_score,
                "risk_level": a.risk_level,
                "latency_seconds": a.latency_seconds,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ]
    finally:
        session.close()
```

## File Structure

```
crackgraphai/
├── db/
│   ├── __init__.py              # Package exports
│   ├── models.py                # SQLAlchemy ORM models
│   ├── database.py              # Connection management
│   ├── repository.py            # Data access layer
│   ├── service.py               # Business logic
│   ├── api_integration.py       # FastAPI integration utilities
│   ├── init_db.py               # Database initialization script
│   ├── init.sql                 # SQL initialization script
│   └── README.md                # Database documentation
├── api/
│   ├── main.py                  # Development API
│   └── production_main.py        # Production API (with DB integration)
├── requirements-db.txt          # Database dependencies
├── docker-compose.db.yml        # PostgreSQL + pgAdmin setup
└── DB_INTEGRATION_GUIDE.md      # This file
```

## Database Schema Overview

### Main Tables

1. **analyses** - Core analysis records
   - Stores SI score, risk level, latency, file info
   - Indexed by request_id, si_score, risk_level, created_at

2. **damage_metrics** - Damage breakdown
   - Stores density, network, complexity, width damage
   - Foreign key to analyses

3. **graph_features** - Topology metrics
   - Stores crack length, branches, endpoints, junctions, etc.
   - Foreign key to analyses

4. **post_processing_stats** - Filtering statistics
   - Stores raw/filtered/final pixel counts
   - Foreign key to analyses

5. **api_audit_logs** - Request audit trail
   - Stores endpoint, method, status, response time, IP
   - Indexed by created_at, status_code, endpoint

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
print(f"Total: {stats['total_analyses']}")
print(f"Avg SI: {stats['avg_si_score']}")
print(f"Risk distribution: {stats['risk_distribution']}")

session.close()
```

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql://user:password@host:5432/crackgraphai

# Connection pool
DB_POOL_SIZE=10              # Number of connections to maintain
DB_MAX_OVERFLOW=20           # Additional connections allowed
DB_POOL_RECYCLE=3600         # Recycle connections after 1 hour
DB_POOL_PRE_PING=true        # Test connections before use
DB_ECHO=false                # Log SQL queries (debug only)
```

### Connection Pool Tuning

For different workloads:

```bash
# Light load (< 10 concurrent requests)
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Medium load (10-50 concurrent requests)
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Heavy load (> 50 concurrent requests)
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

## Monitoring

### Health Check

```python
from db.database import db_manager

if db_manager.health_check():
    print("✓ Database is healthy")
else:
    print("✗ Database connection failed")
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

### Database Size

```sql
-- Connect to PostgreSQL
psql -U postgres -d crackgraphai

-- Check database size
SELECT pg_size_pretty(pg_database_size('crackgraphai'));

-- Check table sizes
SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Troubleshooting

### Connection Refused

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection string
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

### Tables Not Created

```bash
# Run initialization
python -m db.init_db

# Check tables exist
psql $DATABASE_URL -c "\dt"
```

### Slow Queries

```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Check slow queries
SELECT query, calls, mean_time FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
```

### Connection Pool Exhausted

```bash
# Increase pool size in .env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Restart API
```

## Performance Tips

1. **Use Indexes**: All frequently queried columns are indexed
2. **Batch Operations**: Use batch endpoints for multiple images
3. **Archive Old Data**: Delete analyses older than 90 days
4. **Monitor Connections**: Keep pool size appropriate for load
5. **Enable Pre-ping**: Detects and removes stale connections

## Backup & Recovery

### Backup Database

```bash
# Full backup
pg_dump -U postgres crackgraphai > backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump -U postgres crackgraphai | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# From SQL file
psql -U postgres crackgraphai < backup_20240101.sql

# From compressed file
gunzip -c backup_20240101.sql.gz | psql -U postgres crackgraphai
```

## Production Deployment

### Recommended Setup

1. **Separate Database Server**: Don't run DB on same machine as API
2. **Connection Pooling**: Use PgBouncer for additional pooling
3. **Replication**: Set up PostgreSQL replication for HA
4. **Backups**: Automated daily backups to S3 or similar
5. **Monitoring**: Use pg_stat_statements and Prometheus

### Docker Production Setup

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: crackgraphai
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

## Support & Documentation

- **Database Models**: See `db/models.py`
- **Repository Layer**: See `db/repository.py`
- **Service Layer**: See `db/service.py`
- **Full Documentation**: See `db/README.md`

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements-db.txt`
2. ✅ Start PostgreSQL: `docker-compose -f docker-compose.db.yml up -d`
3. ✅ Initialize database: `python -m db.init_db`
4. ✅ Update .env with DATABASE_URL
5. ✅ Integrate with API (see Option A or B above)
6. ✅ Test with: `python -m db.init_db` (should show success)
7. ✅ Deploy and monitor

---

**Database Layer Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✓
