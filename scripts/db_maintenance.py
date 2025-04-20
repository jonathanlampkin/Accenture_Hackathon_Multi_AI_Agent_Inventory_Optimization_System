#!/usr/bin/env python
"""
Database Maintenance Script for Inventory Optimization System.

This script provides utilities for database maintenance operations:
- Archiving old inventory and forecast data to historical tables
- Creating database backups
- Purging old data based on retention policies
- Scheduling regular maintenance tasks

Example usage:
    python scripts/db_maintenance.py --archive --retention-days 365
    python scripts/db_maintenance.py --backup --backup-dir /path/to/backups
    python scripts/db_maintenance.py --purge --retention-days 90 --dry-run
    python scripts/db_maintenance.py --schedule daily
"""
import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = project_root / "logs" / "db_maintenance"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"db_maintenance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("db_maintenance")


class DBMaintenance:
    """Database maintenance operations for Inventory Optimization System."""

    def __init__(
        self,
        db_url: str,
        backup_dir: Optional[str] = None,
    ):
        """Initialize database maintenance.
        
        Args:
            db_url: Database connection URL
            backup_dir: Directory to store backups
        """
        self.db_url = db_url
        
        # Parse connection details from URL
        # Expected format: postgresql://username:password@host:port/dbname
        if db_url.startswith("postgresql://"):
            # Extract connection details
            conn_parts = db_url.replace("postgresql://", "").split("/")
            auth_host = conn_parts[0].split("@")
            
            if ":" in auth_host[0]:
                auth = auth_host[0].split(":")
                self.user = auth[0]
                self.password = auth[1]
            else:
                self.user = auth_host[0]
                self.password = None
            
            if ":" in auth_host[1]:
                host_port = auth_host[1].split(":")
                self.host = host_port[0]
                self.port = host_port[1]
            else:
                self.host = auth_host[1]
                self.port = "5432"  # Default PostgreSQL port
            
            self.dbname = conn_parts[1]
        else:
            raise ValueError("Only PostgreSQL databases are supported")
        
        # Set backup directory
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = project_root / "backups"
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Create SQLAlchemy engine and session
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Set table names for archiving
        self.source_tables = {
            "inventory": "inventory",
            "inventory_transaction": "inventory_transaction",
            "forecast": "forecast",
        }
        
        self.archive_tables = {
            "inventory": "inventory_history",
            "inventory_transaction": "inventory_transaction_history",
            "forecast": "forecast_history",
        }
        
        # Check if archive tables exist, create them if not
        self._ensure_archive_tables_exist()
    
    def _ensure_archive_tables_exist(self):
        """Ensure archive tables exist, create them if they don't."""
        try:
            session = self.SessionLocal()
            
            # Check if archive tables exist
            for source_table, archive_table in self.archive_tables.items():
                # Check if archive table exists
                query = text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{archive_table}'
                    );
                """)
                
                result = session.execute(query).scalar()
                
                if not result:
                    logger.info(f"Creating archive table '{archive_table}'")
                    
                    # Create archive table with same structure as source table
                    query = text(f"""
                        CREATE TABLE {archive_table} AS 
                        SELECT * FROM {self.source_tables[source_table]} 
                        WHERE 1=0;
                        
                        -- Add archive timestamp column
                        ALTER TABLE {archive_table} 
                        ADD COLUMN archived_at TIMESTAMP WITH TIME ZONE 
                        DEFAULT CURRENT_TIMESTAMP;
                        
                        -- Add index on created_at column
                        CREATE INDEX idx_{archive_table}_created_at 
                        ON {archive_table} (created_at);
                        
                        -- Add index on archived_at column
                        CREATE INDEX idx_{archive_table}_archived_at 
                        ON {archive_table} (archived_at);
                    """)
                    
                    session.execute(query)
                    session.commit()
                    logger.info(f"Archive table '{archive_table}' created successfully")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error ensuring archive tables exist: {str(e)}")
            raise
    
    def create_backup(self, compress: bool = True) -> str:
        """Create a backup of the database.
        
        Args:
            compress: Whether to compress the backup
            
        Returns:
            Path to the backup file
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{self.dbname}_{timestamp}.sql"
            backup_path = self.backup_dir / backup_filename
            
            # Command for pg_dump
            cmd = [
                "pg_dump",
                f"--host={self.host}",
                f"--port={self.port}",
                f"--username={self.user}",
                f"--dbname={self.dbname}",
                "--format=plain",
                f"--file={backup_path}",
            ]
            
            logger.info(f"Creating database backup: {backup_path}")
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            if self.password:
                env["PGPASSWORD"] = self.password
            
            # Execute pg_dump
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info(f"Database backup created successfully: {backup_path}")
            
            # Compress the backup if requested
            if compress:
                compressed_path = f"{backup_path}.gz"
                logger.info(f"Compressing backup: {compressed_path}")
                
                with open(backup_path, "rb") as f_in:
                    # Use gzip to compress the backup
                    cmd = ["gzip", "-c"]
                    process = subprocess.Popen(
                        cmd,
                        stdin=f_in,
                        stdout=open(compressed_path, "wb"),
                        stderr=subprocess.PIPE,
                    )
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"Error compressing backup: {stderr.decode()}")
                        return str(backup_path)
                    
                    # Remove uncompressed backup
                    os.remove(backup_path)
                    logger.info(f"Backup compressed successfully: {compressed_path}")
                    return compressed_path
            
            return str(backup_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating backup: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore a database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            logger.info(f"Restoring database from backup: {backup_path}")
            
            # Decompress if compressed
            if backup_path.suffix == ".gz":
                logger.info("Decompressing backup...")
                
                decompressed_path = backup_path.with_suffix("")
                cmd = ["gunzip", "-c", str(backup_path)]
                
                with open(decompressed_path, "wb") as f_out:
                    process = subprocess.Popen(
                        cmd,
                        stdout=f_out,
                        stderr=subprocess.PIPE,
                    )
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"Error decompressing backup: {stderr.decode()}")
                        return False
                
                backup_path = decompressed_path
                logger.info(f"Backup decompressed successfully: {backup_path}")
            
            # Command for psql
            cmd = [
                "psql",
                f"--host={self.host}",
                f"--port={self.port}",
                f"--username={self.user}",
                f"--dbname={self.dbname}",
                "-f", str(backup_path),
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            if self.password:
                env["PGPASSWORD"] = self.password
            
            # Execute psql
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info("Database restored successfully")
            
            # Remove decompressed file if it was created
            if backup_path.name != str(backup_path):
                os.remove(backup_path)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error restoring backup: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            return False
    
    def archive_data(self, table: str, retention_days: int, batch_size: int = 10000, dry_run: bool = False) -> int:
        """Archive old data to historical tables.
        
        Args:
            table: Table to archive data from
            retention_days: Number of days to retain data
            batch_size: Number of rows to archive in each batch
            dry_run: If True, don't actually archive data
            
        Returns:
            Number of rows archived
        """
        try:
            if table not in self.source_tables:
                logger.error(f"Invalid table name: {table}")
                return 0
            
            source_table = self.source_tables[table]
            archive_table = self.archive_tables[table]
            
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Archiving data from '{source_table}' to '{archive_table}' older than {cutoff_date_str}")
            
            session = self.SessionLocal()
            
            # Count rows to archive
            query = text(f"""
                SELECT COUNT(*) FROM {source_table}
                WHERE created_at < :cutoff_date
            """)
            
            total_rows = session.execute(query, {"cutoff_date": cutoff_date}).scalar()
            
            logger.info(f"Found {total_rows} rows to archive from '{source_table}'")
            
            if total_rows == 0:
                logger.info(f"No data to archive for table '{source_table}'")
                session.close()
                return 0
            
            if dry_run:
                logger.info(f"Dry run, would archive {total_rows} rows from '{source_table}'")
                session.close()
                return total_rows
            
            # Archive in batches to avoid locking issues
            rows_archived = 0
            batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
            
            for batch in range(batches):
                logger.info(f"Processing batch {batch + 1}/{batches} for table '{source_table}'")
                
                # Begin transaction
                transaction = session.begin()
                
                try:
                    # Archive batch of data
                    query = text(f"""
                        WITH rows_to_archive AS (
                            SELECT * FROM {source_table}
                            WHERE created_at < :cutoff_date
                            ORDER BY created_at
                            LIMIT :batch_size
                        )
                        INSERT INTO {archive_table}
                        SELECT *, CURRENT_TIMESTAMP AS archived_at
                        FROM rows_to_archive
                        RETURNING id
                    """)
                    
                    result = session.execute(
                        query,
                        {
                            "cutoff_date": cutoff_date,
                            "batch_size": batch_size,
                        }
                    )
                    
                    # Get IDs of archived rows
                    archived_ids = [row[0] for row in result]
                    
                    if not archived_ids:
                        transaction.rollback()
                        break
                    
                    # Delete archived rows from source table
                    query = text(f"""
                        DELETE FROM {source_table}
                        WHERE id IN :archived_ids
                    """)
                    
                    session.execute(query, {"archived_ids": tuple(archived_ids)})
                    
                    # Commit transaction
                    transaction.commit()
                    
                    batch_count = len(archived_ids)
                    rows_archived += batch_count
                    
                    logger.info(f"Archived {batch_count} rows from '{source_table}' (total: {rows_archived})")
                    
                except Exception as e:
                    transaction.rollback()
                    logger.error(f"Error archiving batch for table '{source_table}': {str(e)}")
                    break
                
                # Sleep briefly to reduce database load
                time.sleep(0.5)
            
            session.close()
            
            if rows_archived > 0:
                logger.info(f"Successfully archived {rows_archived} rows from '{source_table}' to '{archive_table}'")
            
            return rows_archived
            
        except Exception as e:
            logger.error(f"Error archiving data for table '{table}': {str(e)}")
            return 0
    
    def purge_archived_data(self, table: str, retention_days: int, batch_size: int = 10000, dry_run: bool = False) -> int:
        """Purge old data from archive tables.
        
        Args:
            table: Table to purge data from
            retention_days: Number of days to retain archived data
            batch_size: Number of rows to purge in each batch
            dry_run: If True, don't actually purge data
            
        Returns:
            Number of rows purged
        """
        try:
            if table not in self.archive_tables:
                logger.error(f"Invalid archive table name: {table}")
                return 0
            
            archive_table = self.archive_tables[table]
            
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Purging data from '{archive_table}' older than {cutoff_date_str}")
            
            session = self.SessionLocal()
            
            # Count rows to purge
            query = text(f"""
                SELECT COUNT(*) FROM {archive_table}
                WHERE archived_at < :cutoff_date
            """)
            
            total_rows = session.execute(query, {"cutoff_date": cutoff_date}).scalar()
            
            logger.info(f"Found {total_rows} rows to purge from '{archive_table}'")
            
            if total_rows == 0:
                logger.info(f"No data to purge for table '{archive_table}'")
                session.close()
                return 0
            
            if dry_run:
                logger.info(f"Dry run, would purge {total_rows} rows from '{archive_table}'")
                session.close()
                return total_rows
            
            # Purge in batches to avoid locking issues
            rows_purged = 0
            batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
            
            for batch in range(batches):
                logger.info(f"Processing batch {batch + 1}/{batches} for table '{archive_table}'")
                
                # Begin transaction
                transaction = session.begin()
                
                try:
                    # Delete batch of data
                    query = text(f"""
                        WITH rows_to_delete AS (
                            SELECT id FROM {archive_table}
                            WHERE archived_at < :cutoff_date
                            ORDER BY archived_at
                            LIMIT :batch_size
                        )
                        DELETE FROM {archive_table}
                        WHERE id IN (SELECT id FROM rows_to_delete)
                        RETURNING id
                    """)
                    
                    result = session.execute(
                        query,
                        {
                            "cutoff_date": cutoff_date,
                            "batch_size": batch_size,
                        }
                    )
                    
                    # Get number of deleted rows
                    deleted_ids = [row[0] for row in result]
                    
                    if not deleted_ids:
                        transaction.rollback()
                        break
                    
                    # Commit transaction
                    transaction.commit()
                    
                    batch_count = len(deleted_ids)
                    rows_purged += batch_count
                    
                    logger.info(f"Purged {batch_count} rows from '{archive_table}' (total: {rows_purged})")
                    
                except Exception as e:
                    transaction.rollback()
                    logger.error(f"Error purging batch for table '{archive_table}': {str(e)}")
                    break
                
                # Sleep briefly to reduce database load
                time.sleep(0.5)
            
            session.close()
            
            if rows_purged > 0:
                logger.info(f"Successfully purged {rows_purged} rows from '{archive_table}'")
            
            return rows_purged
            
        except Exception as e:
            logger.error(f"Error purging data for table '{table}': {str(e)}")
            return 0
    
    def clean_old_backups(self, retention_days: int, dry_run: bool = False) -> int:
        """Clean old database backups.
        
        Args:
            retention_days: Number of days to retain backups
            dry_run: If True, don't actually delete backups
            
        Returns:
            Number of backups deleted
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Cleaning backup files older than {cutoff_date_str}")
            
            # Find all backup files
            backup_files = list(self.backup_dir.glob(f"{self.dbname}_*.sql*"))
            
            if not backup_files:
                logger.info("No backup files found")
                return 0
            
            # Count files to delete
            files_to_delete = []
            
            for backup_file in backup_files:
                # Extract timestamp from filename
                # Expected format: dbname_YYYYMMDD_HHMMSS.sql[.gz]
                try:
                    timestamp_str = backup_file.stem.split("_", 1)[1]
                    if "." in timestamp_str:  # Handle .sql.gz case
                        timestamp_str = timestamp_str.rsplit(".", 1)[0]
                    
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if timestamp < cutoff_date:
                        files_to_delete.append(backup_file)
                except (ValueError, IndexError):
                    logger.warning(f"Could not extract timestamp from filename: {backup_file}")
                    continue
            
            if not files_to_delete:
                logger.info("No old backup files found to delete")
                return 0
            
            logger.info(f"Found {len(files_to_delete)} old backup files to delete")
            
            if dry_run:
                logger.info(f"Dry run, would delete {len(files_to_delete)} backup files")
                for file in files_to_delete:
                    logger.info(f"Would delete: {file}")
                return len(files_to_delete)
            
            # Delete old backup files
            for file in files_to_delete:
                try:
                    os.remove(file)
                    logger.info(f"Deleted backup file: {file}")
                except Exception as e:
                    logger.error(f"Error deleting backup file {file}: {str(e)}")
            
            return len(files_to_delete)
            
        except Exception as e:
            logger.error(f"Error cleaning old backups: {str(e)}")
            return 0


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Database maintenance for Inventory Optimization System."
    )
    
    # Database connection
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory"),
        help="Database connection URL",
    )
    
    # Backup options
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a database backup",
    )
    
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=os.environ.get("BACKUP_DIR"),
        help="Directory to store backups",
    )
    
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Do not compress the backup",
    )
    
    parser.add_argument(
        "--clean-backups",
        action="store_true",
        help="Clean old backup files",
    )
    
    # Archive options
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive old data to historical tables",
    )
    
    parser.add_argument(
        "--archive-table",
        type=str,
        choices=["all", "inventory", "inventory_transaction", "forecast"],
        default="all",
        help="Table to archive data from",
    )
    
    # Purge options
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge old data from archive tables",
    )
    
    parser.add_argument(
        "--purge-table",
        type=str,
        choices=["all", "inventory", "inventory_transaction", "forecast"],
        default="all",
        help="Table to purge data from",
    )
    
    # Common options
    parser.add_argument(
        "--retention-days",
        type=int,
        default=365,
        help="Number of days to retain data or backups",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of rows to process in each batch",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually modify data, just show what would be done",
    )
    
    # Schedule options
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["daily", "weekly", "monthly"],
        help="Schedule regular maintenance tasks",
    )
    
    return parser.parse_args()


def main():
    """Run the script."""
    args = parse_args()
    
    logger.info("Starting database maintenance")
    
    # Initialize the database maintenance with connection details
    db_maintenance = DBMaintenance(
        db_url=args.db_url,
        backup_dir=args.backup_dir,
    )
    
    # Perform requested operations
    if args.backup:
        logger.info("Creating database backup")
        compress = not args.no_compress
        backup_path = db_maintenance.create_backup(compress=compress)
        logger.info(f"Backup created: {backup_path}")
    
    if args.clean_backups:
        logger.info(f"Cleaning old backups (retention: {args.retention_days} days)")
        deleted_count = db_maintenance.clean_old_backups(
            retention_days=args.retention_days,
            dry_run=args.dry_run,
        )
        logger.info(f"Cleaned {deleted_count} old backup files")
    
    if args.archive:
        logger.info(f"Archiving old data (retention: {args.retention_days} days)")
        
        tables = [args.archive_table]
        if args.archive_table == "all":
            tables = list(db_maintenance.source_tables.keys())
        
        total_archived = 0
        for table in tables:
            archived_count = db_maintenance.archive_data(
                table=table,
                retention_days=args.retention_days,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            total_archived += archived_count
        
        logger.info(f"Archived {total_archived} rows in total")
    
    if args.purge:
        logger.info(f"Purging old archived data (retention: {args.retention_days} days)")
        
        tables = [args.purge_table]
        if args.purge_table == "all":
            tables = list(db_maintenance.source_tables.keys())
        
        total_purged = 0
        for table in tables:
            purged_count = db_maintenance.purge_archived_data(
                table=table,
                retention_days=args.retention_days,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            total_purged += purged_count
        
        logger.info(f"Purged {total_purged} rows in total")
    
    if args.schedule:
        logger.info(f"Scheduling {args.schedule} maintenance tasks")
        
        # Create cron job for scheduled maintenance
        if args.schedule == "daily":
            cron_time = "0 0 * * *"  # Every day at midnight
        elif args.schedule == "weekly":
            cron_time = "0 0 * * 0"  # Every Sunday at midnight
        elif args.schedule == "monthly":
            cron_time = "0 0 1 * *"  # First day of each month at midnight
        
        script_path = os.path.abspath(__file__)
        cron_cmd = f"cd {project_root} && {sys.executable} {script_path} "
        
        # Add requested operations to cron command
        if args.backup:
            cron_cmd += "--backup "
            if args.backup_dir:
                cron_cmd += f"--backup-dir={args.backup_dir} "
            if args.no_compress:
                cron_cmd += "--no-compress "
        
        if args.clean_backups:
            cron_cmd += "--clean-backups "
        
        if args.archive:
            cron_cmd += "--archive "
            if args.archive_table != "all":
                cron_cmd += f"--archive-table={args.archive_table} "
        
        if args.purge:
            cron_cmd += "--purge "
            if args.purge_table != "all":
                cron_cmd += f"--purge-table={args.purge_table} "
        
        cron_cmd += f"--retention-days={args.retention_days} "
        cron_cmd += f"--batch-size={args.batch_size} "
        
        # Log full command
        logger.info(f"Cron schedule: {cron_time}")
        logger.info(f"Cron command: {cron_cmd}")
        
        # Add to crontab (just showing the command, not actually adding it)
        if args.dry_run:
            logger.info("Dry run, would add the following to crontab:")
            logger.info(f"{cron_time} {cron_cmd} >> {log_dir}/cron.log 2>&1")
        else:
            try:
                # Get existing crontab
                process = subprocess.run(
                    ["crontab", "-l"],
                    capture_output=True,
                    text=True,
                )
                
                if process.returncode == 0:
                    crontab = process.stdout
                else:
                    crontab = ""
                
                # Add new cron job
                crontab += f"\n# Database maintenance for Inventory Optimization System\n"
                crontab += f"{cron_time} {cron_cmd} >> {log_dir}/cron.log 2>&1\n"
                
                # Write to temporary file
                with open("/tmp/crontab.tmp", "w") as f:
                    f.write(crontab)
                
                # Install new crontab
                subprocess.run(
                    ["crontab", "/tmp/crontab.tmp"],
                    check=True,
                )
                
                # Remove temporary file
                os.remove("/tmp/crontab.tmp")
                
                logger.info("Scheduled maintenance tasks in crontab")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error scheduling maintenance tasks: {e.stderr}")
            except Exception as e:
                logger.error(f"Error scheduling maintenance tasks: {str(e)}")
    
    logger.info("Database maintenance completed")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 