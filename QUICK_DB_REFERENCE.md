# Quick Database Reference

## Installation (1 minute)

```bash
# 1. Install dependencies
pip install -r requirements-db.txt

# 2. Start PostgreSQL
docker-compose -f docker-compose.db.yml up -d

# 3. Initialize database
python -m db.init_db
```

## Configuration (1 minute)

Add to `.env`:
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/crackgraphai
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
DB_ECHO=false
```

## Integration (5 minutes)

### Option 1: Minimal (Just Save Results)

```python
from db.database import get_db
from db.api_integration import AnalysisResultSaver

result_saver = AnalysisResultSaver(get_db)

# In your /predict endpoint:
result_saver.save_result(
    request_id=request_id,
    result=inference_result,
    input_filename=image.filename,
    input_file_size=len(content),
    input_file_bytes=content,
    api_key=api_key,
)
```

### Option 2: Full (Save + Audit + Retrieve)

See `DB_INTEGRATION_GUIDE.md` for complete code examples.

## Common Operations

### Save Analysis
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
session = get_db()
service = AnalysisService(session)

result = service.get_analysis_with_details("req-12345")
print(result)

session.close()
```

### Get Statistics
```python
session = get_db()
service = AnalysisService(session)

stats = service.get_statistics()
# {
#   "total_analyses": 1234,
#   "avg_si_score": 0.72,
#   "risk_distribution": {"Low": 500, "Moderate": 400, ...},
#   "avg_latency": 0.45
# }

session.close()
```

### Log API Request
```python
from db.service import APIAuditService

session = get_db()
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

session.close()
```

## Database Schema

### analyses
- `request_id` (UNIQUE) - Unique request ID
- `si_score` - Structural Integrity score (0-1)
- `risk_level` - Risk classification
- `latency_seconds` - Processing time
- `input_filename` - Original filename
- `input_file_hash` - SHA256 hash (for deduplication)
- `created_at` - Timestamp

### damage_metrics
- `analysis_id` (FK) - Reference to analyses
- `density_damage` - Crack coverage (0-1)
- `network_damage` - Junction severity (0-1)
- `complexity_damage` - Branching complexity (0-1)
- `width_damage` - Crack thickness (0-1)
- `total_damage` - Weighted total (0-1)

### graph_features
- `analysis_id` (FK) - Reference to analyses
- `total_crack_length` - Skeleton pixels
- `num_branches` - Branch count
- `num_endpoints` - Endpoint count
- `num_junctions` - Junction count
- `longest_path` - Longest path length
- `graph_diameter` - Graph diameter
- `mean_node_degree` - Average degree
- `connectivity_score` - Connectivity (0-1)

### post_processing_stats
- `analysis_id` (FK) - Reference to analyses
- `raw_pixels` - Pixels before filtering
- `filtered_pixels` - Pixels removed
- `final_pixels` - Final crack pixels
- `filtering_applied` - Boolean flag

### api_audit_logs
- `request_id` - Reference to analysis
- `api_key_hash` - Hashed API key
- `endpoint` - API endpoint
- `method` - HTTP method
- `status_code` - Response code
- `response_time_ms` - Response time
- `error_message` - Error if failed
- `ip_address` - Client IP

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
from db.service import APIAuditService

session = get_db()
audit_service = APIAuditService(session)

error_rate = audit_service.get_error_rate(minutes=5)
print(f"Error rate (last 5 min): {error_rate:.2f}%")

session.close()
```

### Database Size
```sql
SELECT pg_size_pretty(pg_database_size('crackgraphai'));
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

# Check tables
psql $DATABASE_URL -c "\dt"
```

### Slow Queries
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

## Performance Tuning

### Light Load (< 10 concurrent)
```bash
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
```

### Medium Load (10-50 concurrent)
```bash
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### Heavy Load (> 50 concurrent)
```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

## Files Reference

| File | Purpose |
|------|---------|
| `db/models.py` | SQLAlchemy ORM models |
| `db/database.py` | Connection management |
| `db/repository.py` | Data access layer |
| `db/service.py` | Business logic |
| `db/api_integration.py` | FastAPI utilities |
| `db/init_db.py` | Initialization script |
| `db/README.md` | Full documentation |
| `DB_INTEGRATION_GUIDE.md` | Integration instructions |
| `requirements-db.txt` | Dependencies |
| `docker-compose.db.yml` | PostgreSQL setup |

## Documentation

- **Full Guide**: `DB_INTEGRATION_GUIDE.md`
- **Database Docs**: `db/README.md`
- **Setup Summary**: `DATABASE_SETUP_SUMMARY.md`
- **This File**: `QUICK_DB_REFERENCE.md`

## Status

✅ **Production Ready**
- All tables created with indexes
- Connection pooling configured
- Error handling implemented
- Logging enabled
- Documentation complete

---

**Quick Links**:
- Installation: 3 commands
- Configuration: 6 environment variables
- Integration: 3 lines of code
- Documentation: 4 markdown files
