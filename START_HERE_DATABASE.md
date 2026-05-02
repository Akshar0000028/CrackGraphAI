# 🚀 START HERE - Database Setup

A production-ready database layer has been added to CrackGraphAI. **No existing code was modified.**

## ⚡ Quick Start (10 minutes)

### Step 1: Install Dependencies (1 min)
```bash
pip install -r requirements-db.txt
```

### Step 2: Start PostgreSQL (1 min)
```bash
docker-compose -f docker-compose.db.yml up -d
```

### Step 3: Initialize Database (1 min)
```bash
python -m db.init_db
```

### Step 4: Update .env (1 min)
```bash
# Add these to your .env file:
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/crackgraphai
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
DB_ECHO=false
```

### Step 5: Integrate with API (5 min)
See **DB_INTEGRATION_GUIDE.md** for code examples.

## 📦 What Was Added

### Database Package (`db/`)
- **models.py** - 5 SQLAlchemy ORM models
- **database.py** - Connection pooling & management
- **repository.py** - Data access layer (CRUD)
- **service.py** - Business logic
- **api_integration.py** - FastAPI utilities
- **init_db.py** - Initialization script
- **README.md** - Full documentation

### Configuration
- **docker-compose.db.yml** - PostgreSQL + pgAdmin setup
- **requirements-db.txt** - Database dependencies
- **.env.example** - Updated with DB variables

### Documentation
- **DB_INTEGRATION_GUIDE.md** - Complete integration instructions
- **DATABASE_SETUP_SUMMARY.md** - Setup overview
- **QUICK_DB_REFERENCE.md** - Quick reference
- **START_HERE_DATABASE.md** - This file

## 📊 Database Schema

### 5 Tables Created

| Table | Purpose | Columns |
|-------|---------|---------|
| **analyses** | Main analysis records | 15 |
| **damage_metrics** | Damage breakdown | 6 |
| **graph_features** | Topology metrics | 9 |
| **post_processing_stats** | Filtering statistics | 5 |
| **api_audit_logs** | Request audit trail | 9 |

**Total**: 5 tables, 44 columns, 10 indexes

## 🔌 Integration (Optional)

The database is **completely optional** and **non-intrusive**:

```python
# Add 3 lines to your /predict endpoint:
from db.api_integration import AnalysisResultSaver, get_db

result_saver = AnalysisResultSaver(get_db)

# After inference:
result_saver.save_result(
    request_id=request_id,
    result=inference_result,
    input_filename=image.filename,
    input_file_size=len(content),
    input_file_bytes=content,
    api_key=api_key,
)
```

**That's it!** Results are now saved to the database.

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **QUICK_DB_REFERENCE.md** | 2-page quick reference |
| **DB_INTEGRATION_GUIDE.md** | Complete integration instructions |
| **db/README.md** | Comprehensive database documentation |
| **DATABASE_SETUP_SUMMARY.md** | Setup overview |

## ✅ Features

### Production-Ready
- ✓ Connection pooling (10 connections, 20 overflow)
- ✓ Connection recycling (3600s)
- ✓ Pre-ping enabled (detects stale connections)
- ✓ Comprehensive error handling
- ✓ Logging at all layers

### Performance
- ✓ Optimized indexes on all query columns
- ✓ Foreign key relationships with cascading deletes
- ✓ Efficient pagination support
- ✓ Aggregate statistics queries

### Security
- ✓ API key hashing (SHA256)
- ✓ Audit logging of all requests
- ✓ IP address tracking
- ✓ Error message sanitization

## 🛠️ Common Operations

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
result = service.get_analysis_with_details("req-12345")
```

### Get Statistics
```python
stats = service.get_statistics()
# Returns: total_analyses, avg_si_score, risk_distribution, avg_latency
```

## 🔍 Monitoring

### Health Check
```python
from db.database import db_manager

if db_manager.health_check():
    print("✓ Database is healthy")
```

### Error Rate
```python
from db.service import APIAuditService

audit_service = APIAuditService(session)
error_rate = audit_service.get_error_rate(minutes=5)
print(f"Error rate: {error_rate:.2f}%")
```

## 🐛 Troubleshooting

### Connection Refused
```bash
# Check PostgreSQL is running
docker ps | grep postgres

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

## 📋 File Structure

```
crackgraphai/
├── db/                          (NEW - Database package)
│   ├── __init__.py
│   ├── models.py
│   ├── database.py
│   ├── repository.py
│   ├── service.py
│   ├── api_integration.py
│   ├── init_db.py
│   ├── init.sql
│   └── README.md
├── docker-compose.db.yml        (NEW)
├── requirements-db.txt          (NEW)
├── DB_INTEGRATION_GUIDE.md      (NEW)
├── DATABASE_SETUP_SUMMARY.md    (NEW)
├── QUICK_DB_REFERENCE.md        (NEW)
└── START_HERE_DATABASE.md       (NEW - This file)
```

## 🚀 Next Steps

1. **Install**: `pip install -r requirements-db.txt`
2. **Start DB**: `docker-compose -f docker-compose.db.yml up -d`
3. **Initialize**: `python -m db.init_db`
4. **Configure**: Update `.env` with DATABASE_URL
5. **Integrate**: Add 3 lines to your API (see DB_INTEGRATION_GUIDE.md)
6. **Test**: Run your API and verify data is saved
7. **Monitor**: Check database health and statistics

## 📞 Support

- **Quick Reference**: QUICK_DB_REFERENCE.md
- **Full Guide**: DB_INTEGRATION_GUIDE.md
- **Database Docs**: db/README.md
- **Setup Summary**: DATABASE_SETUP_SUMMARY.md

## ✨ Status

✅ **Production Ready**
- All tables created with proper indexes
- Connection pooling configured
- Error handling implemented
- Logging enabled
- Documentation complete
- No breaking changes
- Backward compatible

---

**Database Layer Version**: 1.0.0  
**Status**: Production Ready ✓  
**Files Created**: 16  
**Code Added**: ~55 KB  
**Existing Code Modified**: 0 files  
**Breaking Changes**: None
