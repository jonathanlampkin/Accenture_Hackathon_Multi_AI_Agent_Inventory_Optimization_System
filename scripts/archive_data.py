#!/usr/bin/env python
"""
Data archiving script for scheduled execution.

This script uses the DataArchiver utility to archive old data from the main
database to maintain performance. It is designed to be run as a scheduled task
(e.g., via a cron job).

Example usage:
    python scripts/archive_data.py --tx-days 90 --forecast-days 180 --audit-days 365
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

from src.utils.data_archiver import DataArchiver

# Configure logging
log_dir = project_root / "logs" / "archiving"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("archiving")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Archive old data from the main database."
    )
    
    # Data retention arguments
    parser.add_argument(
        "--tx-days",
        type=int,
        default=90,
        help="Archive inventory transactions older than this many days (default: 90)",
    )
    
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=180,
        help="Archive forecasts older than this many days (default: 180)",
    )
    
    parser.add_argument(
        "--audit-days",
        type=int,
        default=365,
        help="Archive audit logs older than this many days (default: 365)",
    )
    
    # Control arguments
    parser.add_argument(
        "--retain-forecasts",
        action="store_true",
        default=True,
        help="Retain the latest forecast for each product (default: True)",
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export archived data to CSV files",
    )
    
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(project_root / "data" / "archives"),
        help="Directory to export CSV files to",
    )
    
    parser.add_argument(
        "--archive-db",
        type=str,
        default=None,
        help="Database URL for the archive database (default: same as main DB)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to process in each batch (default: 1000)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate archiving without making changes",
    )
    
    return parser.parse_args()


def main():
    """Run the data archiving job."""
    args = parse_args()
    
    logger.info("Starting data archiving job")
    logger.info(f"Arguments: {args}")
    
    try:
        # Initialize the archiver
        archiver = DataArchiver(
            archive_db_url=args.archive_db,
            batch_size=args.batch_size,
        )
        
        # Run the archiving job
        results = archiver.run_archiving_job(
            inventory_tx_days=args.tx_days,
            forecast_days=args.forecast_days,
            retain_latest_forecasts=args.retain_forecasts,
            audit_log_days=args.audit_days,
            dry_run=args.dry_run,
        )
        
        # Log the results
        logger.info("Archiving job completed")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        # Export archived data if requested
        if args.export and not args.dry_run:
            logger.info(f"Exporting archived data to {args.export_dir}")
            
            # Create export directory
            os.makedirs(args.export_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(args.export_dir, f"export_{timestamp}")
            
            # Export each table that was archived
            for table_name in results.keys():
                if results[table_name]["archived_count"] > 0:
                    output_dir = os.path.join(export_dir, table_name)
                    archiver.export_archive_to_csv(table_name, output_dir)
            
            logger.info(f"Export completed to {export_dir}")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error during archiving job: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 