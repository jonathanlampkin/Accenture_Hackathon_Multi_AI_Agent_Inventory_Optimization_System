"""
Database backup and disaster recovery utility.

This module provides functionality for creating database backups, managing
backup retention, and restoring from backups in case of data loss or corruption.
"""
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BackupManager:
    """Utility for creating, managing, and restoring database backups."""

    def __init__(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
        db_password: Optional[str] = None,
        db_port: int = 5432,
        backup_dir: str = "backups",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "db_backups",
        aws_region: str = "us-east-1",
    ):
        """Initialize BackupManager.
        
        Args:
            db_host: Database hostname
            db_name: Database name
            db_user: Database user
            db_password: Database password (optional, can be in environment variable)
            db_port: Database port
            backup_dir: Local directory for backups
            s3_bucket: Optional S3 bucket name for remote storage
            s3_prefix: Prefix for S3 storage paths
            aws_region: AWS region for S3
        """
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_port = db_port
        self.backup_dir = Path(backup_dir)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.aws_region = aws_region
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize S3 client if bucket provided
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client("s3", region_name=self.aws_region)

    def _get_pg_env(self) -> Dict[str, str]:
        """Get environment variables for pg_dump and psql commands.
        
        Returns:
            Dict[str, str]: Environment variables
        """
        env = os.environ.copy()
        
        if self.db_password:
            env["PGPASSWORD"] = self.db_password
        
        return env

    def create_backup(
        self, backup_format: str = "custom", compression_level: int = 9
    ) -> str:
        """Create a database backup.
        
        Args:
            backup_format: Format for pg_dump (custom, plain, etc.)
            compression_level: Compression level (0-9)
            
        Returns:
            str: Path to the created backup file
            
        Raises:
            subprocess.CalledProcessError: If pg_dump fails
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{self.db_name}_{timestamp}.dump"
        backup_path = self.backup_dir / backup_filename
        
        # Build pg_dump command
        cmd = [
            "pg_dump",
            f"--host={self.db_host}",
            f"--port={self.db_port}",
            f"--username={self.db_user}",
            f"--dbname={self.db_name}",
            f"--format={backup_format}",
            f"--compress={compression_level}",
            f"--file={backup_path}",
        ]
        
        logger.info(f"Creating backup to {backup_path}")
        start_time = time.time()
        
        try:
            subprocess.run(cmd, env=self._get_pg_env(), check=True, capture_output=True)
            
            elapsed_time = time.time() - start_time
            backup_size = os.path.getsize(backup_path)
            
            logger.info(
                f"Backup completed in {elapsed_time:.2f} seconds. "
                f"Size: {backup_size / 1024 / 1024:.2f} MB"
            )
            
            return str(backup_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup failed: {e.stderr.decode() if e.stderr else str(e)}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            raise

    def upload_backup_to_s3(self, backup_path: str) -> str:
        """Upload a backup file to S3.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            str: S3 object key
            
        Raises:
            ValueError: If S3 bucket is not configured
            ClientError: If S3 upload fails
        """
        if not self.s3_client or not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        backup_filename = os.path.basename(backup_path)
        s3_key = f"{self.s3_prefix}/{backup_filename}"
        
        logger.info(f"Uploading backup to S3: {self.s3_bucket}/{s3_key}")
        
        try:
            start_time = time.time()
            
            with open(backup_path, "rb") as file_obj:
                self.s3_client.upload_fileobj(file_obj, self.s3_bucket, s3_key)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Upload completed in {elapsed_time:.2f} seconds")
            
            return s3_key
        except ClientError as e:
            logger.error(f"Upload to S3 failed: {str(e)}")
            raise

    def download_backup_from_s3(self, s3_key: str) -> str:
        """Download a backup file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            str: Path to the downloaded backup file
            
        Raises:
            ValueError: If S3 bucket is not configured
            ClientError: If S3 download fails
        """
        if not self.s3_client or not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        backup_filename = os.path.basename(s3_key)
        local_path = self.backup_dir / backup_filename
        
        logger.info(f"Downloading backup from S3: {self.s3_bucket}/{s3_key}")
        
        try:
            start_time = time.time()
            
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            
            elapsed_time = time.time() - start_time
            logger.info(f"Download completed in {elapsed_time:.2f} seconds")
            
            return str(local_path)
        except ClientError as e:
            logger.error(f"Download from S3 failed: {str(e)}")
            raise

    def list_local_backups(self) -> List[Dict[str, Union[str, int, float]]]:
        """List all local backup files with metadata.
        
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of backup metadata
        """
        backups = []
        
        for file_path in sorted(self.backup_dir.glob(f"{self.db_name}_*.dump")):
            try:
                # Extract timestamp from filename
                timestamp_str = file_path.stem.split("_", 1)[1]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Get file stats
                stat = file_path.stat()
                size_bytes = stat.st_size
                created_time = stat.st_mtime
                
                backups.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "timestamp": timestamp.isoformat(),
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / 1024 / 1024,
                    "created_at": datetime.fromtimestamp(created_time).isoformat(),
                })
            except Exception as e:
                logger.warning(f"Error processing backup file {file_path}: {str(e)}")
        
        return backups

    def list_s3_backups(self) -> List[Dict[str, Union[str, int, float]]]:
        """List all backup files in S3 with metadata.
        
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of backup metadata
            
        Raises:
            ValueError: If S3 bucket is not configured
            ClientError: If S3 operation fails
        """
        if not self.s3_client or not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        backups = []
        
        try:
            # List objects in the S3 bucket with the given prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix,
            )
            
            for page in page_iterator:
                if "Contents" not in page:
                    continue
                
                for obj in page["Contents"]:
                    key = obj["Key"]
                    filename = os.path.basename(key)
                    
                    # Only include backup files for this database
                    if not filename.startswith(f"{self.db_name}_"):
                        continue
                    
                    # Extract timestamp from filename
                    try:
                        timestamp_str = filename.split(".")[0].split("_", 1)[1]
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        backups.append({
                            "filename": filename,
                            "s3_key": key,
                            "timestamp": timestamp.isoformat(),
                            "size_bytes": obj["Size"],
                            "size_mb": obj["Size"] / 1024 / 1024,
                            "last_modified": obj["LastModified"].isoformat(),
                        })
                    except Exception as e:
                        logger.warning(f"Error processing S3 backup {key}: {str(e)}")
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return backups
        except ClientError as e:
            logger.error(f"Failed to list S3 backups: {str(e)}")
            raise

    def prune_local_backups(
        self, retain_count: int = 5, min_retention_days: int = 7
    ) -> int:
        """Prune local backup files, keeping a specified number of recent backups.
        
        Args:
            retain_count: Number of recent backups to retain
            min_retention_days: Minimum days to retain backups
            
        Returns:
            int: Number of backups removed
        """
        backups = self.list_local_backups()
        
        if len(backups) <= retain_count:
            logger.info(f"Only {len(backups)} local backups exist, none will be pruned")
            return 0
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Keep the most recent backups
        to_keep = backups[:retain_count]
        candidates_for_deletion = backups[retain_count:]
        
        # Also keep any backups within the min retention period
        min_retention_cutoff = datetime.now().timestamp() - (min_retention_days * 24 * 60 * 60)
        
        removed_count = 0
        for backup in candidates_for_deletion:
            # Get created timestamp
            created_time = datetime.fromisoformat(backup["created_at"]).timestamp()
            
            # If the backup is within the min retention period, keep it
            if created_time >= min_retention_cutoff:
                logger.info(
                    f"Retaining backup {backup['filename']} as it's within the "
                    f"{min_retention_days} day retention period"
                )
                continue
            
            # Delete the backup
            backup_path = backup["path"]
            try:
                os.remove(backup_path)
                logger.info(f"Deleted backup {backup_path}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to delete backup {backup_path}: {str(e)}")
        
        logger.info(f"Pruned {removed_count} local backups")
        return removed_count

    def prune_s3_backups(
        self, retain_count: int = 10, min_retention_days: int = 30
    ) -> int:
        """Prune S3 backup files, keeping a specified number of recent backups.
        
        Args:
            retain_count: Number of recent backups to retain
            min_retention_days: Minimum days to retain backups
            
        Returns:
            int: Number of backups removed
            
        Raises:
            ValueError: If S3 bucket is not configured
            ClientError: If S3 operation fails
        """
        if not self.s3_client or not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        backups = self.list_s3_backups()
        
        if len(backups) <= retain_count:
            logger.info(f"Only {len(backups)} S3 backups exist, none will be pruned")
            return 0
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Keep the most recent backups
        to_keep = backups[:retain_count]
        candidates_for_deletion = backups[retain_count:]
        
        # Also keep any backups within the min retention period
        min_retention_cutoff = datetime.now().timestamp() - (min_retention_days * 24 * 60 * 60)
        
        removed_count = 0
        for backup in candidates_for_deletion:
            # Get last modified timestamp
            last_modified_time = datetime.fromisoformat(backup["last_modified"]).timestamp()
            
            # If the backup is within the min retention period, keep it
            if last_modified_time >= min_retention_cutoff:
                logger.info(
                    f"Retaining S3 backup {backup['filename']} as it's within the "
                    f"{min_retention_days} day retention period"
                )
                continue
            
            # Delete the backup
            s3_key = backup["s3_key"]
            try:
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                logger.info(f"Deleted S3 backup {s3_key}")
                removed_count += 1
            except ClientError as e:
                logger.error(f"Failed to delete S3 backup {s3_key}: {str(e)}")
        
        logger.info(f"Pruned {removed_count} S3 backups")
        return removed_count

    def restore_from_backup(
        self, backup_path: str, target_db: Optional[str] = None, recreate_db: bool = False
    ) -> bool:
        """Restore database from a backup file.
        
        Args:
            backup_path: Path to the backup file
            target_db: Target database name (defaults to self.db_name)
            recreate_db: Whether to drop and recreate the database
            
        Returns:
            bool: True if restore was successful
            
        Raises:
            FileNotFoundError: If backup file doesn't exist
            subprocess.CalledProcessError: If restore fails
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        target_db = target_db or self.db_name
        
        logger.info(f"Restoring database {target_db} from backup {backup_path}")
        
        # If recreating database, drop and create it first
        if recreate_db:
            logger.warning(f"Dropping and recreating database {target_db}")
            
            # Connect to postgres database to drop/create target
            try:
                # Drop database if it exists
                drop_cmd = [
                    "psql",
                    f"--host={self.db_host}",
                    f"--port={self.db_port}",
                    f"--username={self.db_user}",
                    "--dbname=postgres",
                    "-c", f"DROP DATABASE IF EXISTS {target_db}",
                ]
                
                subprocess.run(drop_cmd, env=self._get_pg_env(), check=True, capture_output=True)
                
                # Create new database
                create_cmd = [
                    "psql",
                    f"--host={self.db_host}",
                    f"--port={self.db_port}",
                    f"--username={self.db_user}",
                    "--dbname=postgres",
                    "-c", f"CREATE DATABASE {target_db}",
                ]
                
                subprocess.run(create_cmd, env=self._get_pg_env(), check=True, capture_output=True)
                
                logger.info(f"Database {target_db} recreated successfully")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to recreate database: {e.stderr.decode() if e.stderr else str(e)}"
                )
                return False
        
        # Restore from backup
        try:
            # For custom format, use pg_restore
            if backup_path.endswith(".dump"):
                restore_cmd = [
                    "pg_restore",
                    f"--host={self.db_host}",
                    f"--port={self.db_port}",
                    f"--username={self.db_user}",
                    f"--dbname={target_db}",
                    "--no-owner",  # Don't include commands to set ownership
                    "--no-privileges",  # Don't include access privileges
                    backup_path,
                ]
            # For plain format, use psql
            else:
                restore_cmd = [
                    "psql",
                    f"--host={self.db_host}",
                    f"--port={self.db_port}",
                    f"--username={self.db_user}",
                    f"--dbname={target_db}",
                    "-f", backup_path,
                ]
            
            start_time = time.time()
            subprocess.run(restore_cmd, env=self._get_pg_env(), check=True, capture_output=True)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Restore completed in {elapsed_time:.2f} seconds")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Restore failed: {e.stderr.decode() if e.stderr else str(e)}")
            return False

    def run_backup_job(
        self,
        upload_to_s3: bool = True,
        prune_local: bool = True,
        prune_s3: bool = True,
        local_retain_count: int = 5,
        s3_retain_count: int = 10,
    ) -> Dict[str, Union[str, bool, int]]:
        """Run a complete backup job including pruning old backups.
        
        Args:
            upload_to_s3: Whether to upload the backup to S3
            prune_local: Whether to prune old local backups
            prune_s3: Whether to prune old S3 backups
            local_retain_count: Number of local backups to retain
            s3_retain_count: Number of S3 backups to retain
            
        Returns:
            Dict[str, Union[str, bool, int]]: Backup job results
        """
        results = {
            "success": False,
            "backup_path": None,
            "s3_key": None,
            "local_pruned": 0,
            "s3_pruned": 0,
            "error": None,
        }
        
        try:
            # Create backup
            backup_path = self.create_backup()
            results["backup_path"] = backup_path
            
            # Upload to S3 if configured
            if upload_to_s3 and self.s3_bucket:
                s3_key = self.upload_backup_to_s3(backup_path)
                results["s3_key"] = s3_key
            
            # Prune old backups
            if prune_local:
                local_pruned = self.prune_local_backups(retain_count=local_retain_count)
                results["local_pruned"] = local_pruned
            
            if prune_s3 and self.s3_bucket:
                s3_pruned = self.prune_s3_backups(retain_count=s3_retain_count)
                results["s3_pruned"] = s3_pruned
            
            results["success"] = True
        except Exception as e:
            logger.exception(f"Backup job failed: {str(e)}")
            results["error"] = str(e)
        
        return results 