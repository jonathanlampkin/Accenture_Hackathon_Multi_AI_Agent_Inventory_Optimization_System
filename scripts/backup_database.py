#!/usr/bin/env python
"""
Database backup script for scheduled execution.

This script uses the BackupManager utility to create database backups, upload
them to S3, and manage backup retention. It is designed to be run as a scheduled
task (e.g., via a cron job).

Example usage:
    python scripts/backup_database.py --s3-bucket inventory-backups
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.backup_manager import BackupManager

# Configure logging
log_dir = project_root / "logs" / "backups"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("backup")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Create and manage database backups."
    )
    
    # Database connection arguments
    parser.add_argument(
        "--db-host",
        type=str,
        default=os.environ.get("DB_HOST", "localhost"),
        help="Database host (default: from DB_HOST env var or 'localhost')",
    )
    
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.environ.get("DB_PORT", "5432")),
        help="Database port (default: from DB_PORT env var or 5432)",
    )
    
    parser.add_argument(
        "--db-name",
        type=str,
        default=os.environ.get("DB_NAME", "inventory"),
        help="Database name (default: from DB_NAME env var or 'inventory')",
    )
    
    parser.add_argument(
        "--db-user",
        type=str,
        default=os.environ.get("DB_USER", "postgres"),
        help="Database user (default: from DB_USER env var or 'postgres')",
    )
    
    parser.add_argument(
        "--db-password",
        type=str,
        default=os.environ.get("DB_PASSWORD"),
        help="Database password (default: from DB_PASSWORD env var)",
    )
    
    # Backup storage arguments
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=os.environ.get("BACKUP_DIR", str(project_root / "backups")),
        help="Local directory for backups (default: from BACKUP_DIR env var or './backups')",
    )
    
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket for backup storage (default: from S3_BUCKET env var)",
    )
    
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=os.environ.get("S3_PREFIX", "database_backups"),
        help="S3 key prefix (default: from S3_PREFIX env var or 'database_backups')",
    )
    
    parser.add_argument(
        "--aws-region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: from AWS_REGION env var or 'us-east-1')",
    )
    
    # Backup control arguments
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to S3 even if configured",
    )
    
    parser.add_argument(
        "--no-prune-local",
        action="store_true",
        help="Skip pruning local backups",
    )
    
    parser.add_argument(
        "--no-prune-s3",
        action="store_true",
        help="Skip pruning S3 backups",
    )
    
    parser.add_argument(
        "--local-retain",
        type=int,
        default=int(os.environ.get("LOCAL_RETAIN", "5")),
        help="Number of local backups to retain (default: from LOCAL_RETAIN env var or 5)",
    )
    
    parser.add_argument(
        "--s3-retain",
        type=int,
        default=int(os.environ.get("S3_RETAIN", "10")),
        help="Number of S3 backups to retain (default: from S3_RETAIN env var or 10)",
    )
    
    return parser.parse_args()


def main():
    """Run the database backup job."""
    args = parse_args()
    
    logger.info("Starting database backup job")
    logger.info(f"Target database: {args.db_name} on {args.db_host}:{args.db_port}")
    
    try:
        # Initialize backup manager
        backup_manager = BackupManager(
            db_host=args.db_host,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            db_port=args.db_port,
            backup_dir=args.backup_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            aws_region=args.aws_region,
        )
        
        # Run backup job
        results = backup_manager.run_backup_job(
            upload_to_s3=not args.no_upload,
            prune_local=not args.no_prune_local,
            prune_s3=not args.no_prune_s3,
            local_retain_count=args.local_retain,
            s3_retain_count=args.s3_retain,
        )
        
        # Log results
        if results["success"]:
            logger.info("Backup job completed successfully")
            
            backup_path = results.get("backup_path")
            if backup_path:
                logger.info(f"Backup created: {backup_path}")
                
                # Check file size
                size_bytes = os.path.getsize(backup_path)
                logger.info(f"Backup size: {size_bytes / 1024 / 1024:.2f} MB")
            
            s3_key = results.get("s3_key")
            if s3_key:
                logger.info(f"Backup uploaded to S3: {args.s3_bucket}/{s3_key}")
            
            local_pruned = results.get("local_pruned", 0)
            if local_pruned > 0:
                logger.info(f"Pruned {local_pruned} local backups")
            
            s3_pruned = results.get("s3_pruned", 0)
            if s3_pruned > 0:
                logger.info(f"Pruned {s3_pruned} S3 backups")
        else:
            error = results.get("error", "Unknown error")
            logger.error(f"Backup job failed: {error}")
            return 1
        
        return 0
    
    except Exception as e:
        logger.exception(f"Unhandled exception during backup job: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 