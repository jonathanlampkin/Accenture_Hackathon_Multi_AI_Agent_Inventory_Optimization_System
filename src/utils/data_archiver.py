"""
Data archiving utility for managing historical data.

This module provides functionality for archiving older data from the main
database tables to archive tables or external storage to maintain performance
as the database grows larger.
"""
import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        MetaData, String, Table, Text, create_engine, func,
                        select)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.models.database import get_db, get_engine
from src.models.forecast import Forecast, ForecastJob
from src.models.inventory import Inventory, InventoryTransaction
from src.models.user import AuditLog

logger = logging.getLogger(__name__)


class DataArchiver:
    """Utility for archiving old data from primary database tables."""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        archive_db_url: Optional[str] = None,
        batch_size: int = 1000,
    ):
        """Initialize DataArchiver.
        
        Args:
            engine: SQLAlchemy engine for primary database
            archive_db_url: SQLAlchemy URL for archive database
            batch_size: Number of records to process in each batch
        """
        self.engine = engine or get_engine()
        
        # If no archive DB URL provided, use the same DB but with "_archive" schema
        if archive_db_url:
            self.archive_engine = create_engine(archive_db_url)
            self.same_db = False
        else:
            self.archive_engine = self.engine
            self.same_db = True
        
        self.batch_size = batch_size
        self.archive_metadata = MetaData(schema="archive" if self.same_db else None)
        
        # Initialize archive tables (will be created if they don't exist)
        self._init_archive_tables()

    def _init_archive_tables(self) -> None:
        """Initialize archive tables in the archive database."""
        # Create archive schema if using the same database
        if self.same_db:
            with self.archive_engine.connect() as conn:
                conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS archive"))
                conn.commit()
        
        # Archive tables for inventory data
        self.inventory_transactions_archive = Table(
            "inventory_transactions_archive",
            self.archive_metadata,
            Column("id", Integer, primary_key=True),
            Column("original_id", Integer, index=True),
            Column("inventory_id", Integer),
            Column("quantity_change", Integer),
            Column("previous_quantity", Integer),
            Column("new_quantity", Integer),
            Column("transaction_type", String(50)),
            Column("reference_id", String(100)),
            Column("notes", Text),
            Column("created_by", Integer),
            Column("created_at", DateTime),
            Column("archived_at", DateTime, default=func.now()),
        )
        
        # Archive tables for forecast data
        self.forecasts_archive = Table(
            "forecasts_archive",
            self.archive_metadata,
            Column("id", Integer, primary_key=True),
            Column("original_id", Integer, index=True),
            Column("product_id", Integer),
            Column("model_id", Integer),
            Column("forecast_dates", JSONB),
            Column("forecast_values", JSONB),
            Column("lower_bounds", JSONB),
            Column("upper_bounds", JSONB),
            Column("metrics", JSONB),
            Column("horizon", Integer),
            Column("status", String(20)),
            Column("error_message", Text),
            Column("created_at", DateTime),
            Column("updated_at", DateTime),
            Column("archived_at", DateTime, default=func.now()),
        )
        
        self.forecast_jobs_archive = Table(
            "forecast_jobs_archive",
            self.archive_metadata,
            Column("id", Integer, primary_key=True),
            Column("original_id", Integer, index=True),
            Column("product_id", Integer),
            Column("job_id", String(255)),
            Column("status", String(20)),
            Column("parameters", JSONB),
            Column("result", JSONB),
            Column("error_message", Text),
            Column("created_by", Integer),
            Column("created_at", DateTime),
            Column("updated_at", DateTime),
            Column("archived_at", DateTime, default=func.now()),
        )
        
        # Archive tables for user audit logs
        self.audit_logs_archive = Table(
            "audit_logs_archive",
            self.archive_metadata,
            Column("id", Integer, primary_key=True),
            Column("original_id", Integer, index=True),
            Column("user_id", Integer),
            Column("action", String(255)),
            Column("entity_type", String(50)),
            Column("entity_id", String(50)),
            Column("details", Text),
            Column("ip_address", String(45)),
            Column("user_agent", String(255)),
            Column("created_at", DateTime),
            Column("archived_at", DateTime, default=func.now()),
        )
        
        # Create all archive tables
        self.archive_metadata.create_all(self.archive_engine)
        logger.info("Archive tables initialized")

    def archive_inventory_transactions(
        self, older_than_days: int = 90, dry_run: bool = False
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Archive inventory transactions older than the specified days.
        
        Args:
            older_than_days: Archive transactions older than this many days
            dry_run: If True, only return what would be archived without making changes
            
        Returns:
            Tuple[int, List[Dict[str, Any]]]: Number of records archived and a sample
        """
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        
        # Query to find transactions to archive
        with Session(self.engine) as db:
            query = (
                select(InventoryTransaction)
                .where(InventoryTransaction.created_at < cutoff_date)
                .order_by(InventoryTransaction.created_at)
                .limit(self.batch_size if not dry_run else 10)
            )
            
            transactions = db.execute(query).scalars().all()
            
            if not transactions:
                logger.info(f"No inventory transactions found older than {older_than_days} days")
                return 0, []
            
            # Get sample data for return
            sample_data = [
                {
                    "id": t.id,
                    "inventory_id": t.inventory_id,
                    "transaction_type": t.transaction_type,
                    "quantity_change": t.quantity_change,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in transactions[:5]
            ]
            
            if dry_run:
                total_count = db.query(InventoryTransaction).filter(
                    InventoryTransaction.created_at < cutoff_date
                ).count()
                
                logger.info(
                    f"Dry run: Would archive {total_count} inventory transactions "
                    f"older than {older_than_days} days"
                )
                
                return total_count, sample_data
            
            # Process in batches to avoid memory issues
            archived_count = 0
            
            while transactions:
                # Insert into archive table
                with self.archive_engine.begin() as conn:
                    archive_data = [
                        {
                            "original_id": t.id,
                            "inventory_id": t.inventory_id,
                            "quantity_change": t.quantity_change,
                            "previous_quantity": t.previous_quantity,
                            "new_quantity": t.new_quantity,
                            "transaction_type": t.transaction_type,
                            "reference_id": t.reference_id,
                            "notes": t.notes,
                            "created_by": t.created_by,
                            "created_at": t.created_at,
                            "archived_at": datetime.datetime.utcnow(),
                        }
                        for t in transactions
                    ]
                    
                    conn.execute(self.inventory_transactions_archive.insert(), archive_data)
                
                # Delete from source table
                transaction_ids = [t.id for t in transactions]
                db.query(InventoryTransaction).filter(
                    InventoryTransaction.id.in_(transaction_ids)
                ).delete(synchronize_session=False)
                
                db.commit()
                
                archived_count += len(transactions)
                logger.info(f"Archived {archived_count} inventory transactions so far")
                
                # Get next batch
                query = (
                    select(InventoryTransaction)
                    .where(InventoryTransaction.created_at < cutoff_date)
                    .order_by(InventoryTransaction.created_at)
                    .limit(self.batch_size)
                )
                
                transactions = db.execute(query).scalars().all()
            
            logger.info(f"Successfully archived {archived_count} inventory transactions")
            return archived_count, sample_data

    def archive_forecasts(
        self, older_than_days: int = 180, retain_latest_per_product: bool = True, dry_run: bool = False
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Archive forecasts older than the specified days.
        
        Args:
            older_than_days: Archive forecasts older than this many days
            retain_latest_per_product: If True, keep the most recent forecast for each product
            dry_run: If True, only return what would be archived without making changes
            
        Returns:
            Tuple[int, List[Dict[str, Any]]]: Number of records archived and a sample
        """
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        
        with Session(self.engine) as db:
            # Subquery to find the latest forecast ID for each product
            if retain_latest_per_product:
                latest_forecasts = (
                    db.query(
                        Forecast.product_id,
                        func.max(Forecast.created_at).label("max_date"),
                    )
                    .group_by(Forecast.product_id)
                    .subquery()
                )
                
                # Query to find forecasts to archive (excluding the latest for each product)
                query = (
                    select(Forecast)
                    .where(Forecast.created_at < cutoff_date)
                    .where(
                        ~sa.exists()
                        .where(
                            (Forecast.product_id == latest_forecasts.c.product_id)
                            & (Forecast.created_at == latest_forecasts.c.max_date)
                        )
                    )
                    .order_by(Forecast.created_at)
                    .limit(self.batch_size if not dry_run else 10)
                )
            else:
                # Query to find all forecasts older than cutoff
                query = (
                    select(Forecast)
                    .where(Forecast.created_at < cutoff_date)
                    .order_by(Forecast.created_at)
                    .limit(self.batch_size if not dry_run else 10)
                )
            
            forecasts = db.execute(query).scalars().all()
            
            if not forecasts:
                logger.info(f"No forecasts found older than {older_than_days} days")
                return 0, []
            
            # Get sample data for return
            sample_data = [
                {
                    "id": f.id,
                    "product_id": f.product_id,
                    "model_id": f.model_id,
                    "horizon": f.horizon,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in forecasts[:5]
            ]
            
            if dry_run:
                if retain_latest_per_product:
                    # Count using the same exclusion logic for dry run
                    total_count = db.query(Forecast).filter(
                        Forecast.created_at < cutoff_date,
                        ~Forecast.id.in_(
                            db.query(Forecast.id)
                            .join(
                                latest_forecasts,
                                (Forecast.product_id == latest_forecasts.c.product_id)
                                & (Forecast.created_at == latest_forecasts.c.max_date),
                            )
                            .subquery()
                        ),
                    ).count()
                else:
                    total_count = db.query(Forecast).filter(
                        Forecast.created_at < cutoff_date
                    ).count()
                
                logger.info(
                    f"Dry run: Would archive {total_count} forecasts older than {older_than_days} days"
                    f" (retaining latest per product: {retain_latest_per_product})"
                )
                
                return total_count, sample_data
            
            # Process in batches to avoid memory issues
            archived_count = 0
            
            while forecasts:
                # Insert into archive table
                with self.archive_engine.begin() as conn:
                    archive_data = [
                        {
                            "original_id": f.id,
                            "product_id": f.product_id,
                            "model_id": f.model_id,
                            "forecast_dates": f.forecast_dates,
                            "forecast_values": f.forecast_values,
                            "lower_bounds": f.lower_bounds,
                            "upper_bounds": f.upper_bounds,
                            "metrics": f.metrics,
                            "horizon": f.horizon,
                            "status": f.status.value if hasattr(f.status, "value") else str(f.status),
                            "error_message": f.error_message,
                            "created_at": f.created_at,
                            "updated_at": f.updated_at,
                            "archived_at": datetime.datetime.utcnow(),
                        }
                        for f in forecasts
                    ]
                    
                    conn.execute(self.forecasts_archive.insert(), archive_data)
                
                # Delete from source table
                forecast_ids = [f.id for f in forecasts]
                db.query(Forecast).filter(
                    Forecast.id.in_(forecast_ids)
                ).delete(synchronize_session=False)
                
                db.commit()
                
                archived_count += len(forecasts)
                logger.info(f"Archived {archived_count} forecasts so far")
                
                # Construct query for next batch based on retention policy
                if retain_latest_per_product:
                    query = (
                        select(Forecast)
                        .where(Forecast.created_at < cutoff_date)
                        .where(
                            ~sa.exists()
                            .where(
                                (Forecast.product_id == latest_forecasts.c.product_id)
                                & (Forecast.created_at == latest_forecasts.c.max_date)
                            )
                        )
                        .order_by(Forecast.created_at)
                        .limit(self.batch_size)
                    )
                else:
                    query = (
                        select(Forecast)
                        .where(Forecast.created_at < cutoff_date)
                        .order_by(Forecast.created_at)
                        .limit(self.batch_size)
                    )
                
                forecasts = db.execute(query).scalars().all()
            
            logger.info(f"Successfully archived {archived_count} forecasts")
            return archived_count, sample_data

    def archive_audit_logs(
        self, older_than_days: int = 365, dry_run: bool = False
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Archive audit logs older than the specified days.
        
        Args:
            older_than_days: Archive audit logs older than this many days
            dry_run: If True, only return what would be archived without making changes
            
        Returns:
            Tuple[int, List[Dict[str, Any]]]: Number of records archived and a sample
        """
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        
        with Session(self.engine) as db:
            # Query to find audit logs to archive
            query = (
                select(AuditLog)
                .where(AuditLog.created_at < cutoff_date)
                .order_by(AuditLog.created_at)
                .limit(self.batch_size if not dry_run else 10)
            )
            
            audit_logs = db.execute(query).scalars().all()
            
            if not audit_logs:
                logger.info(f"No audit logs found older than {older_than_days} days")
                return 0, []
            
            # Get sample data for return
            sample_data = [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "entity_type": log.entity_type,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in audit_logs[:5]
            ]
            
            if dry_run:
                total_count = db.query(AuditLog).filter(
                    AuditLog.created_at < cutoff_date
                ).count()
                
                logger.info(
                    f"Dry run: Would archive {total_count} audit logs "
                    f"older than {older_than_days} days"
                )
                
                return total_count, sample_data
            
            # Process in batches to avoid memory issues
            archived_count = 0
            
            while audit_logs:
                # Insert into archive table
                with self.archive_engine.begin() as conn:
                    archive_data = [
                        {
                            "original_id": log.id,
                            "user_id": log.user_id,
                            "action": log.action,
                            "entity_type": log.entity_type,
                            "entity_id": log.entity_id,
                            "details": log.details,
                            "ip_address": log.ip_address,
                            "user_agent": log.user_agent,
                            "created_at": log.created_at,
                            "archived_at": datetime.datetime.utcnow(),
                        }
                        for log in audit_logs
                    ]
                    
                    conn.execute(self.audit_logs_archive.insert(), archive_data)
                
                # Delete from source table
                log_ids = [log.id for log in audit_logs]
                db.query(AuditLog).filter(
                    AuditLog.id.in_(log_ids)
                ).delete(synchronize_session=False)
                
                db.commit()
                
                archived_count += len(audit_logs)
                logger.info(f"Archived {archived_count} audit logs so far")
                
                # Get next batch
                query = (
                    select(AuditLog)
                    .where(AuditLog.created_at < cutoff_date)
                    .order_by(AuditLog.created_at)
                    .limit(self.batch_size)
                )
                
                audit_logs = db.execute(query).scalars().all()
            
            logger.info(f"Successfully archived {archived_count} audit logs")
            return archived_count, sample_data

    def export_archive_to_csv(
        self, table_name: str, output_dir: str, chunk_size: int = 10000
    ) -> str:
        """Export archive table data to CSV files.
        
        Args:
            table_name: Name of the archive table to export
            output_dir: Directory to save CSV files
            chunk_size: Number of records per CSV file
            
        Returns:
            str: Path to the output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the appropriate table
        if table_name == "inventory_transactions":
            table = self.inventory_transactions_archive
        elif table_name == "forecasts":
            table = self.forecasts_archive
        elif table_name == "forecast_jobs":
            table = self.forecast_jobs_archive
        elif table_name == "audit_logs":
            table = self.audit_logs_archive
        else:
            raise ValueError(f"Unsupported table: {table_name}")
        
        # Query to get all data from the archive table
        query = select(table)
        
        with self.archive_engine.connect() as conn:
            result = conn.execute(query)
            column_names = result.keys()
            
            # Process in chunks
            chunk_num = 1
            chunk = result.fetchmany(chunk_size)
            
            while chunk:
                # Generate CSV filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{table_name}_archive_{timestamp}_part{chunk_num}.csv"
                output_path = os.path.join(output_dir, filename)
                
                # Convert to DataFrame and save to CSV
                df = pd.DataFrame(chunk, columns=column_names)
                df.to_csv(output_path, index=False)
                
                logger.info(f"Exported {len(df)} records to {output_path}")
                
                # Get next chunk
                chunk = result.fetchmany(chunk_size)
                chunk_num += 1
        
        logger.info(f"Finished exporting {table_name} to CSV files in {output_dir}")
        return output_dir

    def run_archiving_job(
        self,
        inventory_tx_days: Optional[int] = 90,
        forecast_days: Optional[int] = 180,
        retain_latest_forecasts: bool = True,
        audit_log_days: Optional[int] = 365,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run a complete archiving job for all configured tables.
        
        Args:
            inventory_tx_days: Days threshold for inventory transactions (None to skip)
            forecast_days: Days threshold for forecasts (None to skip)
            retain_latest_forecasts: Whether to retain the latest forecast per product
            audit_log_days: Days threshold for audit logs (None to skip)
            dry_run: If True, simulate the archiving without making changes
            
        Returns:
            Dict[str, Any]: Summary of archiving results
        """
        results = {}
        
        # Archive inventory transactions
        if inventory_tx_days is not None:
            tx_count, tx_sample = self.archive_inventory_transactions(
                older_than_days=inventory_tx_days, dry_run=dry_run
            )
            results["inventory_transactions"] = {
                "archived_count": tx_count,
                "older_than_days": inventory_tx_days,
                "sample_data": tx_sample,
            }
        
        # Archive forecasts
        if forecast_days is not None:
            forecast_count, forecast_sample = self.archive_forecasts(
                older_than_days=forecast_days,
                retain_latest_per_product=retain_latest_forecasts,
                dry_run=dry_run,
            )
            results["forecasts"] = {
                "archived_count": forecast_count,
                "older_than_days": forecast_days,
                "retain_latest": retain_latest_forecasts,
                "sample_data": forecast_sample,
            }
        
        # Archive audit logs
        if audit_log_days is not None:
            log_count, log_sample = self.archive_audit_logs(
                older_than_days=audit_log_days, dry_run=dry_run
            )
            results["audit_logs"] = {
                "archived_count": log_count,
                "older_than_days": audit_log_days,
                "sample_data": log_sample,
            }
        
        return results 