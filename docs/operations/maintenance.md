# System Maintenance Operations Guide

This document outlines the standard maintenance procedures for the Inventory Optimization System. Following these procedures will help ensure the system remains performant, reliable, and secure.

## Scheduled Maintenance Windows

Regular maintenance should be performed during the following low-traffic periods:

| Environment | Primary Window | Secondary Window | Notes |
|-------------|---------------|-----------------|-------|
| Production | Sunday, 1:00 AM - 5:00 AM UTC | Wednesday, 2:00 AM - 4:00 AM UTC | Notify users 72 hours in advance |
| Staging | Tuesday, 2:00 PM - 6:00 PM UTC | As needed | Notify development team 24 hours in advance |
| Development | As needed | N/A | Coordinate with active developers |

## Database Maintenance

### PostgreSQL VACUUM and ANALYZE

Regular VACUUM and ANALYZE operations are critical for maintaining database performance.

#### Automated Maintenance

Automated maintenance is configured through PostgreSQL's built-in autovacuum. Current settings:

```
autovacuum = on
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
```

#### Manual Maintenance

For tables with high churn, perform manual VACUUM ANALYZE:

```sql
-- Vacuum and analyze a specific table
VACUUM ANALYZE inventories;
VACUUM ANALYZE inventory_transactions;
VACUUM ANALYZE forecasts;

-- Vacuum the entire database
VACUUM ANALYZE;
```

**Schedule**: Run manual VACUUM ANALYZE weekly during the primary maintenance window.

### Index Maintenance

Rebuild indexes with high fragmentation:

```sql
-- Rebuild specific index
REINDEX INDEX ix_inventories_product_location;

-- Rebuild all indexes on a table
REINDEX TABLE inventory_transactions;
```

**Schedule**: Run monthly during primary maintenance window.

### Database Statistics Update

Update statistics to ensure the query planner has current information:

```sql
-- Update statistics for specific tables
ANALYZE inventories;
ANALYZE inventory_transactions;
ANALYZE forecasts;

-- Update statistics for the entire database
ANALYZE;
```

**Schedule**: Run weekly, and after any significant data changes.

## Application Maintenance

### Docker Image Updates

Update application container images regularly:

```bash
# Pull latest images
docker-compose pull

# Restart services with new images
docker-compose up -d
```

**Schedule**: Apply critical updates immediately; routine updates during maintenance windows.

### Config Updates

To update application configuration:

```bash
# Edit configuration
nano .env

# Restart affected services
docker-compose restart api
```

**Schedule**: As needed, preferably during maintenance windows.

### Log Rotation

Application logs are automatically rotated using logrotate with the following settings:

```
# /etc/logrotate.d/inventory
/var/log/inventory/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 app app
    sharedscripts
    postrotate
        systemctl reload inventory-api || true
    endscript
}
```

**Schedule**: Automatic daily rotation; manual verification monthly.

## Model Maintenance

### MLflow Model Registry Cleanup

Clean up old and unused models:

```bash
# List models
mlflow models list

# Archive old model versions
mlflow models archive -m models:/inventory_forecast/2

# Delete unused models
mlflow models delete -m models:/deprecated_model
```

**Schedule**: Monthly during primary maintenance window.

### Model Retraining

Regularly retrain models to incorporate new data:

```bash
# Retrain all forecast models
python scripts/train_models.py --all

# Retrain specific product models
python scripts/train_models.py --product-ids 123,456,789
```

**Schedule**: Weekly for high-volume products; monthly for all products.

## Data Maintenance

### Data Archiving

Archive old data to maintain database performance:

```bash
# Archive inventory transactions older than 90 days
python scripts/archive_data.py --tx-days 90

# Archive forecasts older than 180 days
python scripts/archive_data.py --forecast-days 180

# Archive audit logs older than 365 days
python scripts/archive_data.py --audit-days 365
```

**Schedule**: Monthly during primary maintenance window.

### Data Validation

Run data validation checks to ensure data integrity:

```bash
# Validate all data
python scripts/validate_data.py --all

# Validate specific data types
python scripts/validate_data.py --types inventory,forecast
```

**Schedule**: Weekly, and after any significant data changes.

## Security Maintenance

### SSL Certificate Renewal

Renew SSL certificates before expiration:

```bash
# Check certificate expiration
openssl x509 -enddate -noout -in /etc/ssl/certs/inventory-api.crt

# Renew certificates (using certbot)
certbot renew
```

**Schedule**: Automatic renewal with cron; manual verification quarterly.

### Security Updates

Apply security updates promptly:

```bash
# Update system packages
apt update && apt upgrade -y

# Update Python dependencies
pip install --upgrade -r requirements.txt
```

**Schedule**: Critical updates immediately; routine updates during maintenance windows.

### User Access Review

Review user access rights regularly:

```bash
# List users and roles
psql -c "SELECT u.username, r.name FROM users u JOIN user_roles ur ON u.id = ur.user_id JOIN roles r ON ur.role_id = r.id ORDER BY u.username, r.name;" postgres://user:password@localhost:5432/inventory
```

**Schedule**: Quarterly review of all user accounts and permissions.

## Monitoring and Alerting Maintenance

### Prometheus and Grafana

Maintain monitoring infrastructure:

```bash
# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Update Grafana dashboards (from version control)
git pull && cp dashboards/*.json /var/lib/grafana/dashboards/
```

**Schedule**: Check monthly; update as needed.

### Alert Configuration Review

Review and update alerting thresholds:

```bash
# Review current alert rules
cat prometheus/alert_rules.yml

# Update rules as needed
nano prometheus/alert_rules.yml

# Apply changes
docker-compose restart prometheus
```

**Schedule**: Quarterly review of alert configurations and thresholds.

## Backup Maintenance

### Backup Verification

Regularly verify backups are valid:

```bash
# Test restore latest backup to temporary database
python scripts/backup_database.py --restore backups/latest.dump --target-db inventory_test

# Verify backup integrity
python scripts/verify_backup.py --backup backups/latest.dump
```

**Schedule**: Monthly verification of backup integrity.

### Backup Retention Policy

Manage backup retention according to policy:

```bash
# Cleanup old local backups (retain last 5)
python scripts/backup_database.py --prune-local --local-retain 5

# Cleanup old S3 backups (retain last 30)
python scripts/backup_database.py --prune-s3 --s3-retain 30
```

**Schedule**: Automatic cleanup during backup creation; manual verification monthly.

## Scaling Procedures

### Handling Increased Load

Procedures for scaling during high-demand periods:

#### Vertical Scaling

```bash
# Update container resource limits
nano docker-compose.yml  # Modify memory and CPU limits

# Apply changes
docker-compose up -d
```

#### Horizontal Scaling

```bash
# Add more worker instances
docker-compose up -d --scale worker=3
```

**Schedule**: As needed, particularly before known high-traffic periods.

## Maintenance Checklist

Use this checklist for regular maintenance:

### Weekly Maintenance

- [ ] Run VACUUM ANALYZE on high-traffic tables
- [ ] Update database statistics
- [ ] Review system logs for errors
- [ ] Verify backup creation success
- [ ] Check disk usage
- [ ] Run data validation checks
- [ ] Retrain high-volume product models

### Monthly Maintenance

- [ ] Perform full database VACUUM ANALYZE
- [ ] Archive old data
- [ ] Verify backup integrity
- [ ] Clean up old models in MLflow
- [ ] Review Prometheus and Grafana dashboards
- [ ] Update Docker images
- [ ] Retrain all product models

### Quarterly Maintenance

- [ ] Rebuild fragmented indexes
- [ ] Review alert configurations
- [ ] Review user access rights
- [ ] Test disaster recovery procedures
- [ ] Review and update documentation
- [ ] Security vulnerability scan

## Maintenance Records

Document all maintenance activities in the maintenance log:

```bash
# Log maintenance activity
python scripts/log_maintenance.py --activity "Database VACUUM ANALYZE" --details "Performed full database vacuum"
```

Maintenance logs are stored in `/var/log/inventory/maintenance.log` and should include:

- Date and time
- Activity performed
- Duration
- Outcome (success/failure)
- Any issues encountered
- Operator name

## Contact Information

| Role | Name | Contact | Hours |
|------|------|---------|-------|
| Primary DBA | Jane Smith | jane.smith@example.com, (555) 123-4567 | 9am-5pm EST |
| System Administrator | Alex Johnson | alex.johnson@example.com, (555) 987-6543 | 24/7 On-call |
| ML Operations Lead | Sam Wilson | sam.wilson@example.com, (555) 456-7890 | 10am-6pm EST |

For urgent maintenance issues, contact the on-call team at: **oncall@example.com** or **(555) 999-8888** 