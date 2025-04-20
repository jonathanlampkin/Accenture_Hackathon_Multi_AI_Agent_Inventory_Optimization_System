#!/bin/bash
#
# Setup Cron Jobs for Inventory Optimization System
#
# This script sets up scheduled cron jobs for the Inventory Optimization System,
# including database backups, data archiving, and maintenance tasks.
#
# Usage:
#   ./setup_cron_jobs.sh [--user username] [--env /path/to/.env]
#
# Example:
#   ./setup_cron_jobs.sh --user app --env /opt/inventory/production.env
#

set -e

# Default values
CRON_USER=$(whoami)
ENV_FILE=""
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=$(which python || which python3)

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --user)
      CRON_USER="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate Python executable
if [ -z "$PYTHON_BIN" ]; then
  echo "Error: Python executable not found"
  exit 1
fi

# Create temporary crontab file
TEMP_CRONTAB=$(mktemp)

# Get current crontab
crontab -u "$CRON_USER" -l > "$TEMP_CRONTAB" 2>/dev/null || echo "# New crontab" > "$TEMP_CRONTAB"

# Add header
cat << EOF >> "$TEMP_CRONTAB"

#
# Inventory Optimization System Cron Jobs
# Last updated: $(date)
#
EOF

# Environment setup
if [ -n "$ENV_FILE" ]; then
  echo "# Load environment variables" >> "$TEMP_CRONTAB"
  echo "BASH_ENV=$ENV_FILE" >> "$TEMP_CRONTAB"
  echo "" >> "$TEMP_CRONTAB"
fi

# Set PATH to include the app scripts directory
echo "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$APP_DIR/scripts" >> "$TEMP_CRONTAB"
echo "" >> "$TEMP_CRONTAB"

# Database backup jobs
cat << EOF >> "$TEMP_CRONTAB"
# Database Backups
# Daily backup at 1:00 AM
0 1 * * * cd $APP_DIR && $PYTHON_BIN scripts/backup_database.py --s3-bucket inventory-backups >> logs/backups/daily_backup.log 2>&1

# Weekly full backup with verification on Sundays at 2:00 AM
0 2 * * 0 cd $APP_DIR && $PYTHON_BIN scripts/backup_database.py --s3-bucket inventory-backups --verify >> logs/backups/weekly_backup.log 2>&1

EOF

# Data archiving jobs
cat << EOF >> "$TEMP_CRONTAB"
# Data Archiving
# Archive transaction data monthly (1st day at 3:00 AM)
0 3 1 * * cd $APP_DIR && $PYTHON_BIN scripts/archive_data.py --tx-days 90 --forecast-days 180 --audit-days 365 >> logs/archiving/monthly_archive.log 2>&1

# Export archived data to CSV quarterly (1st day of quarter at 4:00 AM)
0 4 1 1,4,7,10 * cd $APP_DIR && $PYTHON_BIN scripts/archive_data.py --export --export-dir data/archives >> logs/archiving/quarterly_export.log 2>&1

EOF

# Database maintenance jobs
cat << EOF >> "$TEMP_CRONTAB"
# Database Maintenance
# Weekly VACUUM ANALYZE (Saturdays at 2:30 AM)
30 2 * * 6 cd $APP_DIR && $PYTHON_BIN scripts/db_maintenance.py --vacuum-analyze >> logs/maintenance/weekly_vacuum.log 2>&1

# Monthly index maintenance (1st Saturday at 3:30 AM)
30 3 1-7 * 6 cd $APP_DIR && $PYTHON_BIN scripts/db_maintenance.py --reindex >> logs/maintenance/monthly_reindex.log 2>&1

EOF

# Model maintenance jobs
cat << EOF >> "$TEMP_CRONTAB"
# Model Maintenance
# Weekly retraining of high-volume product models (Mondays at 1:00 AM)
0 1 * * 1 cd $APP_DIR && $PYTHON_BIN scripts/train_models.py --high-volume >> logs/models/weekly_training.log 2>&1

# Monthly retraining of all models (1st day at 2:00 AM)
0 2 1 * * cd $APP_DIR && $PYTHON_BIN scripts/train_models.py --all >> logs/models/monthly_training.log 2>&1

# Monthly cleanup of old models (2nd day at 3:00 AM)
0 3 2 * * cd $APP_DIR && $PYTHON_BIN scripts/cleanup_models.py --older-than 90 >> logs/models/model_cleanup.log 2>&1

EOF

# Monitoring and validation jobs
cat << EOF >> "$TEMP_CRONTAB"
# Monitoring and Validation
# Daily data validation (daily at 6:00 AM)
0 6 * * * cd $APP_DIR && $PYTHON_BIN scripts/validate_data.py --all >> logs/validation/daily_validation.log 2>&1

# Weekly monitoring system check (Sundays at 7:00 AM)
0 7 * * 0 cd $APP_DIR && $PYTHON_BIN scripts/check_monitoring.py >> logs/monitoring/weekly_check.log 2>&1

EOF

# Security jobs
cat << EOF >> "$TEMP_CRONTAB"
# Security Maintenance
# Check for SSL certificate expiration (weekly on Monday at 8:00 AM)
0 8 * * 1 cd $APP_DIR && $PYTHON_BIN scripts/check_certificates.py >> logs/security/cert_check.log 2>&1

# Check for dependency updates (weekly on Tuesday at 8:00 AM)
0 8 * * 2 cd $APP_DIR && $PYTHON_BIN scripts/check_dependencies.py >> logs/security/dep_check.log 2>&1

EOF

# Install the new crontab
crontab -u "$CRON_USER" "$TEMP_CRONTAB"
rm "$TEMP_CRONTAB"

# Create log directories
mkdir -p "$APP_DIR/logs/"{backups,archiving,maintenance,models,validation,monitoring,security}

echo "Cron jobs have been set up successfully for user $CRON_USER"
echo "Log files will be written to $APP_DIR/logs/"

# Print installed crontab for verification
echo ""
echo "Installed crontab:"
echo "----------------"
crontab -u "$CRON_USER" -l 