#!/usr/bin/env python
"""
Data archiving script.

This script manages historical data archiving and cleanup.
It moves old data to archive tables, exports data to external storage,
and performs cleanups to maintain optimal database performance.

Example usage:
    python scripts/data_archiving.py --archive-transactions --months 6
    python scripts/data_archiving.py --export-archive --storage s3
    python scripts/data_archiving.py --purge-archives --years 2
"""
import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import boto3
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = project_root / "logs" / "archiving"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"data_archiving_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("data_archiving")


class DataArchiver:
    """Utility for archiving historical data."""

    def __init__(
        self,
        db_url: str,
        archive_local_dir: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "archives",
    ):
        """Initialize DataArchiver.
        
        Args:
            db_url: Database connection URL
            archive_local_dir: Local directory for archives
            s3_bucket: S3 bucket name for remote archives
            s3_prefix: S3 key prefix for remote archives
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Set up archive directories
        self.archive_local_dir = archive_local_dir or os.path.join(project_root, "archives")
        os.makedirs(self.archive_local_dir, exist_ok=True)
        
        # S3 settings
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None
        if s3_bucket:
            self.s3_client = boto3.client("s3")
    
    def _init_archive_tables(self):
        """Create archive tables if they don't exist."""
        with self.SessionLocal() as session:
            try:
                # Create archive_inventory_transactions table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS archive_inventory_transactions (
                        LIKE inventory_transactions INCLUDING ALL,
                        archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """))
                
                # Create archive_forecasts table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS archive_forecasts (
                        LIKE forecasts INCLUDING ALL,
                        archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """))
                
                # Create indexes on archive tables
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_archive_inv_trans_created 
                    ON archive_inventory_transactions (created_at)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_archive_inv_trans_archived
                    ON archive_inventory_transactions (archived_at)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_archive_forecasts_created
                    ON archive_forecasts (created_at)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_archive_forecasts_archived
                    ON archive_forecasts (archived_at)
                """))
                
                session.commit()
                logger.info("Archive tables initialized successfully")
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error initializing archive tables: {str(e)}")
                raise
    
    def archive_transactions(self, months: int = 6, batch_size: int = 10000) -> int:
        """Archive old inventory transactions.
        
        Args:
            months: Archive transactions older than this many months
            batch_size: Number of records to process in each batch
            
        Returns:
            Number of archived records
        """
        self._init_archive_tables()
        
        cutoff_date = datetime.now() - timedelta(days=30 * months)
        total_archived = 0
        
        logger.info(f"Archiving inventory transactions older than {cutoff_date}")
        
        with self.SessionLocal() as session:
            try:
                # Count records to archive
                count_query = text("""
                    SELECT COUNT(*) 
                    FROM inventory_transactions 
                    WHERE created_at < :cutoff_date
                """)
                
                total_to_archive = session.execute(
                    count_query, {"cutoff_date": cutoff_date}
                ).scalar()
                
                logger.info(f"Found {total_to_archive} records to archive")
                
                if total_to_archive == 0:
                    return 0
                
                # Process in batches
                while True:
                    # Move batch of records to archive table
                    archive_query = text("""
                        WITH batch AS (
                            SELECT id FROM inventory_transactions
                            WHERE created_at < :cutoff_date
                            ORDER BY created_at
                            LIMIT :batch_size
                        )
                        INSERT INTO archive_inventory_transactions
                        SELECT t.*, NOW() as archived_at
                        FROM inventory_transactions t
                        JOIN batch b ON t.id = b.id
                        RETURNING t.id
                    """)
                    
                    result = session.execute(
                        archive_query, 
                        {"cutoff_date": cutoff_date, "batch_size": batch_size}
                    )
                    
                    archived_ids = [row[0] for row in result]
                    batch_count = len(archived_ids)
                    
                    if batch_count == 0:
                        break
                    
                    # Delete archived records from main table
                    if archived_ids:
                        delete_query = text("""
                            DELETE FROM inventory_transactions
                            WHERE id IN :ids
                        """)
                        
                        session.execute(delete_query, {"ids": tuple(archived_ids)})
                    
                    total_archived += batch_count
                    logger.info(f"Archived {total_archived}/{total_to_archive} records")
                    
                    # Commit each batch
                    session.commit()
                    
                    # Exit if we've processed all records
                    if batch_count < batch_size:
                        break
                
                logger.info(f"Successfully archived {total_archived} inventory transactions")
                return total_archived
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error archiving transactions: {str(e)}")
                raise
    
    def archive_forecasts(self, months: int = 6, batch_size: int = 10000) -> int:
        """Archive old forecasts.
        
        Args:
            months: Archive forecasts older than this many months
            batch_size: Number of records to process in each batch
            
        Returns:
            Number of archived records
        """
        self._init_archive_tables()
        
        cutoff_date = datetime.now() - timedelta(days=30 * months)
        total_archived = 0
        
        logger.info(f"Archiving forecasts older than {cutoff_date}")
        
        with self.SessionLocal() as session:
            try:
                # Count records to archive
                count_query = text("""
                    SELECT COUNT(*) 
                    FROM forecasts 
                    WHERE created_at < :cutoff_date
                """)
                
                total_to_archive = session.execute(
                    count_query, {"cutoff_date": cutoff_date}
                ).scalar()
                
                logger.info(f"Found {total_to_archive} forecasts to archive")
                
                if total_to_archive == 0:
                    return 0
                
                # Process in batches
                while True:
                    # Move batch of records to archive table
                    archive_query = text("""
                        WITH batch AS (
                            SELECT id FROM forecasts
                            WHERE created_at < :cutoff_date
                            ORDER BY created_at
                            LIMIT :batch_size
                        )
                        INSERT INTO archive_forecasts
                        SELECT f.*, NOW() as archived_at
                        FROM forecasts f
                        JOIN batch b ON f.id = b.id
                        RETURNING f.id
                    """)
                    
                    result = session.execute(
                        archive_query, 
                        {"cutoff_date": cutoff_date, "batch_size": batch_size}
                    )
                    
                    archived_ids = [row[0] for row in result]
                    batch_count = len(archived_ids)
                    
                    if batch_count == 0:
                        break
                    
                    # Delete archived records from main table
                    if archived_ids:
                        delete_query = text("""
                            DELETE FROM forecasts
                            WHERE id IN :ids
                        """)
                        
                        session.execute(delete_query, {"ids": tuple(archived_ids)})
                    
                    total_archived += batch_count
                    logger.info(f"Archived {total_archived}/{total_to_archive} forecasts")
                    
                    # Commit each batch
                    session.commit()
                    
                    # Exit if we've processed all records
                    if batch_count < batch_size:
                        break
                
                logger.info(f"Successfully archived {total_archived} forecasts")
                return total_archived
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error archiving forecasts: {str(e)}")
                raise
    
    def export_archive_transactions(
        self, 
        months: int = 6, 
        storage: str = "local",
        compress: bool = True,
        batch_size: int = 50000
    ) -> List[str]:
        """Export archived inventory transactions to files.
        
        Args:
            months: Export transactions archived in the last N months
            storage: Storage type ('local' or 's3')
            compress: Whether to compress the output files
            batch_size: Number of records in each export file
            
        Returns:
            List of exported file paths
        """
        cutoff_date = datetime.now() - timedelta(days=30 * months)
        export_files = []
        
        logger.info(f"Exporting archived inventory transactions from the last {months} months")
        
        with self.SessionLocal() as session:
            try:
                # Count records to export
                count_query = text("""
                    SELECT COUNT(*) 
                    FROM archive_inventory_transactions 
                    WHERE archived_at >= :cutoff_date
                """)
                
                total_to_export = session.execute(
                    count_query, {"cutoff_date": cutoff_date}
                ).scalar()
                
                logger.info(f"Found {total_to_export} records to export")
                
                if total_to_export == 0:
                    return []
                
                # Get earliest/latest dates for filename
                range_query = text("""
                    SELECT 
                        MIN(created_at)::date, 
                        MAX(created_at)::date
                    FROM archive_inventory_transactions 
                    WHERE archived_at >= :cutoff_date
                """)
                
                date_range = session.execute(
                    range_query, {"cutoff_date": cutoff_date}
                ).first()
                
                start_date, end_date = date_range
                date_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
                
                # Process in batches
                offset = 0
                batch_num = 1
                
                while True:
                    # Get batch of records
                    batch_query = text("""
                        SELECT *
                        FROM archive_inventory_transactions
                        WHERE archived_at >= :cutoff_date
                        ORDER BY created_at
                        LIMIT :batch_size OFFSET :offset
                    """)
                    
                    result = session.execute(
                        batch_query, 
                        {
                            "cutoff_date": cutoff_date, 
                            "batch_size": batch_size,
                            "offset": offset
                        }
                    )
                    
                    records = [dict(row) for row in result]
                    batch_count = len(records)
                    
                    if batch_count == 0:
                        break
                    
                    # Create export file
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    filename = f"inventory_transactions_{date_str}_batch{batch_num}_{timestamp}.csv"
                    
                    if compress:
                        filename += ".gz"
                    
                    if storage == "local":
                        file_path = os.path.join(self.archive_local_dir, filename)
                        self._write_to_csv(records, file_path, compress)
                        export_files.append(file_path)
                    elif storage == "s3":
                        if not self.s3_client:
                            raise ValueError("S3 client not initialized")
                        
                        local_path = os.path.join(self.archive_local_dir, filename)
                        self._write_to_csv(records, local_path, compress)
                        
                        # Upload to S3
                        s3_key = f"{self.s3_prefix}/transactions/{filename}"
                        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                        
                        # Delete local file after upload
                        os.remove(local_path)
                        
                        export_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                    else:
                        raise ValueError(f"Unsupported storage type: {storage}")
                    
                    logger.info(f"Exported batch {batch_num} to {export_files[-1]}")
                    
                    offset += batch_size
                    batch_num += 1
                    
                    # Exit if we've processed all records
                    if batch_count < batch_size:
                        break
                
                logger.info(f"Successfully exported archives to {len(export_files)} files")
                return export_files
                
            except Exception as e:
                logger.error(f"Error exporting archived transactions: {str(e)}")
                raise
    
    def export_archive_forecasts(
        self, 
        months: int = 6, 
        storage: str = "local",
        compress: bool = True,
        batch_size: int = 50000
    ) -> List[str]:
        """Export archived forecasts to files.
        
        Args:
            months: Export forecasts archived in the last N months
            storage: Storage type ('local' or 's3')
            compress: Whether to compress the output files
            batch_size: Number of records in each export file
            
        Returns:
            List of exported file paths
        """
        cutoff_date = datetime.now() - timedelta(days=30 * months)
        export_files = []
        
        logger.info(f"Exporting archived forecasts from the last {months} months")
        
        with self.SessionLocal() as session:
            try:
                # Count records to export
                count_query = text("""
                    SELECT COUNT(*) 
                    FROM archive_forecasts 
                    WHERE archived_at >= :cutoff_date
                """)
                
                total_to_export = session.execute(
                    count_query, {"cutoff_date": cutoff_date}
                ).scalar()
                
                logger.info(f"Found {total_to_export} forecasts to export")
                
                if total_to_export == 0:
                    return []
                
                # Get earliest/latest dates for filename
                range_query = text("""
                    SELECT 
                        MIN(created_at)::date, 
                        MAX(created_at)::date
                    FROM archive_forecasts 
                    WHERE archived_at >= :cutoff_date
                """)
                
                date_range = session.execute(
                    range_query, {"cutoff_date": cutoff_date}
                ).first()
                
                start_date, end_date = date_range
                date_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
                
                # Process in batches
                offset = 0
                batch_num = 1
                
                while True:
                    # Get batch of records
                    batch_query = text("""
                        SELECT *
                        FROM archive_forecasts
                        WHERE archived_at >= :cutoff_date
                        ORDER BY created_at
                        LIMIT :batch_size OFFSET :offset
                    """)
                    
                    result = session.execute(
                        batch_query, 
                        {
                            "cutoff_date": cutoff_date, 
                            "batch_size": batch_size,
                            "offset": offset
                        }
                    )
                    
                    records = [dict(row) for row in result]
                    batch_count = len(records)
                    
                    if batch_count == 0:
                        break
                    
                    # Create export file
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    filename = f"forecasts_{date_str}_batch{batch_num}_{timestamp}.csv"
                    
                    if compress:
                        filename += ".gz"
                    
                    if storage == "local":
                        file_path = os.path.join(self.archive_local_dir, filename)
                        self._write_to_csv(records, file_path, compress)
                        export_files.append(file_path)
                    elif storage == "s3":
                        if not self.s3_client:
                            raise ValueError("S3 client not initialized")
                        
                        local_path = os.path.join(self.archive_local_dir, filename)
                        self._write_to_csv(records, local_path, compress)
                        
                        # Upload to S3
                        s3_key = f"{self.s3_prefix}/forecasts/{filename}"
                        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                        
                        # Delete local file after upload
                        os.remove(local_path)
                        
                        export_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                    else:
                        raise ValueError(f"Unsupported storage type: {storage}")
                    
                    logger.info(f"Exported batch {batch_num} to {export_files[-1]}")
                    
                    offset += batch_size
                    batch_num += 1
                    
                    # Exit if we've processed all records
                    if batch_count < batch_size:
                        break
                
                logger.info(f"Successfully exported archives to {len(export_files)} files")
                return export_files
                
            except Exception as e:
                logger.error(f"Error exporting archived forecasts: {str(e)}")
                raise
    
    def purge_archive_transactions(self, years: int = 2) -> int:
        """Purge old archive records.
        
        Args:
            years: Purge archives older than this many years
            
        Returns:
            Number of purged records
        """
        cutoff_date = datetime.now() - timedelta(days=365 * years)
        
        logger.info(f"Purging archived inventory transactions older than {cutoff_date}")
        
        with self.SessionLocal() as session:
            try:
                # Delete old archive records
                delete_query = text("""
                    DELETE FROM archive_inventory_transactions
                    WHERE created_at < :cutoff_date
                """)
                
                result = session.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_count = result.rowcount
                
                session.commit()
                
                logger.info(f"Purged {deleted_count} archived inventory transactions")
                return deleted_count
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error purging archived transactions: {str(e)}")
                raise
    
    def purge_archive_forecasts(self, years: int = 2) -> int:
        """Purge old forecast archives.
        
        Args:
            years: Purge archives older than this many years
            
        Returns:
            Number of purged records
        """
        cutoff_date = datetime.now() - timedelta(days=365 * years)
        
        logger.info(f"Purging archived forecasts older than {cutoff_date}")
        
        with self.SessionLocal() as session:
            try:
                # Delete old archive records
                delete_query = text("""
                    DELETE FROM archive_forecasts
                    WHERE created_at < :cutoff_date
                """)
                
                result = session.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_count = result.rowcount
                
                session.commit()
                
                logger.info(f"Purged {deleted_count} archived forecasts")
                return deleted_count
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error purging archived forecasts: {str(e)}")
                raise
    
    def _write_to_csv(self, records: List[Dict], file_path: str, compress: bool = True):
        """Write records to CSV file.
        
        Args:
            records: List of records to write
            file_path: Output file path
            compress: Whether to gzip compress the output
        """
        if not records:
            return
        
        # Get column names from first record
        fieldnames = list(records[0].keys())
        
        # Handle JSON serialization for complex types
        for record in records:
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    record[key] = json.dumps(value)
                elif isinstance(value, datetime):
                    record[key] = value.isoformat()
        
        # Write to file
        if compress:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Manage historical data archiving and cleanup."
    )
    
    # Database connection arguments
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory"),
        help="Database connection URL",
    )
    
    # Storage arguments
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=os.environ.get("ARCHIVE_DIR"),
        help="Local directory for archives",
    )
    
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("S3_ARCHIVE_BUCKET"),
        help="S3 bucket for archive storage",
    )
    
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=os.environ.get("S3_ARCHIVE_PREFIX", "archives"),
        help="S3 key prefix for archives",
    )
    
    # Archive operations
    parser.add_argument(
        "--archive-transactions",
        action="store_true",
        help="Archive old inventory transactions",
    )
    
    parser.add_argument(
        "--archive-forecasts",
        action="store_true",
        help="Archive old forecasts",
    )
    
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Age threshold in months for archiving",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing records",
    )
    
    # Export operations
    parser.add_argument(
        "--export-transactions",
        action="store_true",
        help="Export archived transactions to files",
    )
    
    parser.add_argument(
        "--export-forecasts",
        action="store_true",
        help="Export archived forecasts to files",
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        choices=["local", "s3"],
        default="local",
        help="Storage type for exports",
    )
    
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress export files",
    )
    
    # Purge operations
    parser.add_argument(
        "--purge-transactions",
        action="store_true",
        help="Purge old archived transactions",
    )
    
    parser.add_argument(
        "--purge-forecasts",
        action="store_true",
        help="Purge old archived forecasts",
    )
    
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Age threshold in years for purging",
    )
    
    return parser.parse_args()


def main():
    """Run the script."""
    args = parse_args()
    
    # Initialize data archiver
    archiver = DataArchiver(
        db_url=args.db_url,
        archive_local_dir=args.archive_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )
    
    # Record start time
    start_time = time.time()
    
    # Archive operations
    if args.archive_transactions:
        logger.info(f"Archiving inventory transactions older than {args.months} months")
        try:
            archived_count = archiver.archive_transactions(
                months=args.months,
                batch_size=args.batch_size
            )
            logger.info(f"Archived {archived_count} inventory transactions")
        except Exception as e:
            logger.error(f"Failed to archive transactions: {str(e)}")
    
    if args.archive_forecasts:
        logger.info(f"Archiving forecasts older than {args.months} months")
        try:
            archived_count = archiver.archive_forecasts(
                months=args.months,
                batch_size=args.batch_size
            )
            logger.info(f"Archived {archived_count} forecasts")
        except Exception as e:
            logger.error(f"Failed to archive forecasts: {str(e)}")
    
    # Export operations
    if args.export_transactions:
        logger.info(f"Exporting archived transactions from last {args.months} months to {args.storage}")
        try:
            export_files = archiver.export_archive_transactions(
                months=args.months,
                storage=args.storage,
                compress=not args.no_compress,
                batch_size=args.batch_size
            )
            logger.info(f"Exported transactions to {len(export_files)} files")
        except Exception as e:
            logger.error(f"Failed to export transactions: {str(e)}")
    
    if args.export_forecasts:
        logger.info(f"Exporting archived forecasts from last {args.months} months to {args.storage}")
        try:
            export_files = archiver.export_archive_forecasts(
                months=args.months,
                storage=args.storage,
                compress=not args.no_compress,
                batch_size=args.batch_size
            )
            logger.info(f"Exported forecasts to {len(export_files)} files")
        except Exception as e:
            logger.error(f"Failed to export forecasts: {str(e)}")
    
    # Purge operations
    if args.purge_transactions:
        logger.info(f"Purging archived transactions older than {args.years} years")
        try:
            purged_count = archiver.purge_archive_transactions(years=args.years)
            logger.info(f"Purged {purged_count} archived transactions")
        except Exception as e:
            logger.error(f"Failed to purge transactions: {str(e)}")
    
    if args.purge_forecasts:
        logger.info(f"Purging archived forecasts older than {args.years} years")
        try:
            purged_count = archiver.purge_archive_forecasts(years=args.years)
            logger.info(f"Purged {purged_count} archived forecasts")
        except Exception as e:
            logger.error(f"Failed to purge forecasts: {str(e)}")
    
    # Record end time and total duration
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"All operations completed in {total_duration:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 