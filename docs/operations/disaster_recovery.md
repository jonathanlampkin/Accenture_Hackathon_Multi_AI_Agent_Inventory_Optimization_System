# Disaster Recovery Guide

This document outlines the disaster recovery procedures for the Inventory Optimization System. It covers the recovery process for various failure scenarios, from minor component failures to complete system outages.

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Service Component | RTO | RPO | Notes |
|-------------------|-----|-----|-------|
| Database | 1 hour | 24 hours | Using daily backups with WAL archiving |
| API Services | 30 minutes | N/A | Stateless, can be restored from containers |
| ML Models | 4 hours | 7 days | Models registered in MLflow | 
| Message Queue | 2 hours | Varies | Depends on queue persistence configuration |
| Cache | 15 minutes | N/A | Can be rebuilt, no critical state |

## Prerequisites for Recovery

Before proceeding with any recovery operation, ensure the following:

1. You have access to the backup system and S3 bucket
2. You have appropriate credentials for all services
3. You know which components need to be recovered
4. You have reviewed the most recent successful backup/archive logs

## Database Recovery

### Complete Database Restoration

Use this procedure when the entire database needs to be restored from a backup.

```bash
# 1. Download the most recent backup from S3 (if using cloud storage)
python scripts/backup_database.py --download-backup latest

# 2. Restore the database
python scripts/backup_database.py --restore backups/inventory_20230615_120000.dump --recreate-db
```

### Point-in-Time Recovery

For recovery to a specific point in time (requires WAL archiving to be enabled):

```bash
# 1. First restore the base backup
python scripts/backup_database.py --restore backups/inventory_20230615_120000.dump --recreate-db

# 2. Apply WAL files up to the desired recovery point
psql -c "SELECT pg_wal_replay_resume();" postgres://user:password@localhost:5432/inventory
```

### Verifying Recovery

After restoration, verify the database is operational:

```bash
# Check database connectivity
psql -c "SELECT 1;" postgres://user:password@localhost:5432/inventory

# Verify data integrity by checking row counts in critical tables
psql -c "SELECT COUNT(*) FROM products;" postgres://user:password@localhost:5432/inventory
psql -c "SELECT COUNT(*) FROM inventories;" postgres://user:password@localhost:5432/inventory
psql -c "SELECT COUNT(*) FROM forecasts;" postgres://user:password@localhost:5432/inventory
```

## Application Service Recovery

### Container Restart

For issues with application containers:

```bash
# Restart specific service
docker-compose restart api

# Restart all services
docker-compose down && docker-compose up -d
```

### Full Application Redeployment

When a complete redeployment is necessary:

```bash
# Pull the latest images
docker-compose pull

# Deploy with zero downtime (if using orchestration)
kubectl rollout restart deployment inventory-api-deployment

# Or with docker-compose (will have downtime)
docker-compose down && docker-compose up -d
```

## ML Model Recovery

### Restoring Models from MLflow Registry

```bash
# List available models
mlflow models list

# Restore specific model version to production
mlflow models restore -m models:/inventory_forecast/Production -v 3
```

### Rebuilding Models

If models need to be retrained:

```bash
# Run the model training pipeline
python scripts/train_models.py --all

# Verify model metrics
python scripts/evaluate_models.py --all
```

## Message Queue Recovery

### RabbitMQ Recovery

```bash
# Check queue status
rabbitmqctl list_queues

# Restore RabbitMQ from backup (if available)
rabbitmqctl import_definitions /path/to/definitions.json

# Or restart the service
docker-compose restart rabbitmq
```

## Redis Cache Recovery

Redis is used for caching and can be safely flushed and rebuilt:

```bash
# Clear Redis cache
redis-cli FLUSHALL

# Restart Redis
docker-compose restart redis
```

## Complete System Recovery

For a full system outage, follow these steps in order:

1. **Start Database**:
   ```bash
   docker-compose up -d postgres
   ```

2. **Restore Database** (if needed):
   ```bash
   python scripts/backup_database.py --restore backups/latest.dump
   ```

3. **Start Supporting Services**:
   ```bash
   docker-compose up -d redis rabbitmq
   ```

4. **Start Application Services**:
   ```bash
   docker-compose up -d api worker scheduler
   ```

5. **Verify System Health**:
   ```bash
   python scripts/health_check.py --all
   ```

## Monitoring Recovery Progress

During recovery operations, monitor these indicators:

1. **System Logs**:
   ```bash
   docker-compose logs -f
   ```

2. **Database Connections**:
   ```bash
   psql -c "SELECT count(*) FROM pg_stat_activity;" postgres://user:password@localhost:5432/inventory
   ```

3. **API Health Endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Prometheus Metrics**:
   Access the Prometheus dashboard at http://localhost:9090

## Post-Recovery Actions

After successful recovery:

1. **Validate Critical Functionality**:
   - Run the end-to-end test suite
   - Verify inventory calculations
   - Check forecast generation

2. **Create New Backups**:
   ```bash
   python scripts/backup_database.py
   ```

3. **Document Incident**:
   - Record recovery time
   - Document cause of failure
   - Note any issues encountered during recovery
   - Update procedures if needed

4. **Notify Stakeholders**:
   - Inform users that the system is operational
   - Provide details on any data loss if applicable

## Disaster Recovery Testing

Schedule regular DR testing:

- Monthly: Database restore test
- Quarterly: Full system recovery test
- Annually: Complete DR simulation including failover to standby environment

### DR Test Procedure

```bash
# 1. Create a test environment
docker-compose -f docker-compose.test.yml up -d

# 2. Restore latest backup to test environment
python scripts/backup_database.py --restore backups/latest.dump --target-db inventory_test

# 3. Run validation tests
pytest tests/recovery_validation/

# 4. Clean up test environment
docker-compose -f docker-compose.test.yml down
```

## Contact Information

| Role | Name | Contact | Hours |
|------|------|---------|-------|
| Primary DBA | Jane Smith | jane.smith@example.com, (555) 123-4567 | 9am-5pm EST |
| Secondary DBA | John Doe | john.doe@example.com, (555) 765-4321 | 8am-4pm PST |
| System Administrator | Alex Johnson | alex.johnson@example.com, (555) 987-6543 | 24/7 On-call |
| ML Operations Lead | Sam Wilson | sam.wilson@example.com, (555) 456-7890 | 10am-6pm EST |

For after-hours emergencies, contact the on-call team at: **emergency@example.com** or **(555) 999-8888** 