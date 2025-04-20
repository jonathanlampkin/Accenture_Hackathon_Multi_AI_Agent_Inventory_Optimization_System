#!/usr/bin/env python
"""
Database Performance Tuning Script.

This script analyzes and optimizes database performance for the Inventory Optimization System.
It identifies slow queries, creates missing indexes, provides query optimization suggestions,
and performs regular database maintenance operations.

Example usage:
    python scripts/db_performance.py --analyze
    python scripts/db_performance.py --suggest-indexes
    python scripts/db_performance.py --vacuum-analyze
    python scripts/db_performance.py --optimize-all
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = project_root / "logs" / "db_performance"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"db_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("db_performance")


class DBPerformanceOptimizer:
    """Database performance optimization utility."""

    def __init__(self, db_url: str):
        """Initialize DBPerformanceOptimizer.
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Parse connection parameters from URL for psycopg2
        if "postgresql" in db_url:
            # Example format: postgresql://user:pass@host:port/dbname
            url_parts = db_url.replace("postgresql://", "").split("@")
            user_pass = url_parts[0].split(":")
            host_port_db = url_parts[1].split("/")
            host_port = host_port_db[0].split(":")
            
            self.conn_params = {
                "user": user_pass[0],
                "password": user_pass[1] if len(user_pass) > 1 else None,
                "host": host_port[0],
                "port": int(host_port[1]) if len(host_port) > 1 else 5432,
                "dbname": host_port_db[1],
            }
        else:
            raise ValueError("Only PostgreSQL is supported for performance optimization")
    
    def _get_psycopg2_connection(self):
        """Get a psycopg2 connection.
        
        Returns:
            psycopg2 connection object
        """
        return psycopg2.connect(**self.conn_params)
    
    def enable_query_logging(self, min_duration_ms: int = 100):
        """Enable query logging for slow queries.
        
        Args:
            min_duration_ms: Minimum query duration to log (in milliseconds)
        """
        with self._get_psycopg2_connection() as conn:
            with conn.cursor() as cur:
                # Check if pg_stat_statements extension is installed
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'")
                if not cur.fetchone():
                    try:
                        cur.execute("CREATE EXTENSION pg_stat_statements")
                        conn.commit()
                        logger.info("Installed pg_stat_statements extension")
                    except psycopg2.Error as e:
                        logger.warning(f"Failed to install pg_stat_statements: {e}")
                        logger.warning("You may need to add it to shared_preload_libraries in postgresql.conf")
                
                # Set log_min_duration_statement
                cur.execute(f"ALTER DATABASE {self.conn_params['dbname']} SET log_min_duration_statement = {min_duration_ms}")
                conn.commit()
                
                logger.info(f"Enabled slow query logging for queries taking more than {min_duration_ms}ms")
    
    def analyze_slow_queries(self, min_calls: int = 10, min_avg_time_ms: int = 100, limit: int = 20):
        """Analyze slow queries using pg_stat_statements.
        
        Args:
            min_calls: Minimum number of query calls to include
            min_avg_time_ms: Minimum average time in milliseconds to include
            limit: Maximum number of queries to return
            
        Returns:
            List of dictionaries with query information
        """
        logger.info(f"Analyzing slow queries (min_calls={min_calls}, min_avg_time_ms={min_avg_time_ms})")
        
        with self._get_psycopg2_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    query = """
                    SELECT 
                        calls, 
                        round(total_exec_time::numeric / calls, 2) as avg_exec_time_ms,
                        round(total_exec_time::numeric, 2) as total_exec_time_ms,
                        round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) as percentage_cpu,
                        round(mean_exec_time::numeric, 2) as mean_exec_time_ms,
                        rows_per_call,
                        query
                    FROM (
                        SELECT 
                            calls, 
                            total_exec_time, 
                            mean_exec_time,
                            rows / calls::float as rows_per_call,
                            query
                        FROM pg_stat_statements
                        WHERE calls >= %s
                        AND mean_exec_time >= %s
                        AND userid = (SELECT usesysid FROM pg_user WHERE usename = current_user)
                        AND query !~ '^(SET|SHOW|BEGIN|COMMIT|ROLLBACK)'
                    ) sub
                    ORDER BY total_exec_time_ms DESC
                    LIMIT %s
                    """
                    
                    cur.execute(query, (min_calls, min_avg_time_ms, limit))
                    results = cur.fetchall()
                    
                    if not results:
                        logger.info("No slow queries found matching criteria")
                        return []
                    
                    # Convert to list of dictionaries
                    slow_queries = [dict(row) for row in results]
                    
                    # Log summary of findings
                    logger.info(f"Found {len(slow_queries)} slow queries")
                    
                    # Print top 5 slowest queries
                    for i, query_info in enumerate(slow_queries[:5]):
                        logger.info(f"Slow query #{i+1}:")
                        logger.info(f"  Avg time: {query_info['avg_exec_time_ms']}ms, Calls: {query_info['calls']}")
                        logger.info(f"  CPU%: {query_info['percentage_cpu']}%, Rows/call: {query_info['rows_per_call']}")
                        logger.info(f"  Query: {query_info['query'][:200]}...")
                    
                    return slow_queries
                    
                except psycopg2.Error as e:
                    logger.error(f"Error analyzing slow queries: {e}")
                    return []
    
    def identify_missing_indexes(self, min_calls: int = 100):
        """Identify potential missing indexes.
        
        Args:
            min_calls: Minimum number of scan calls to include
            
        Returns:
            List of dictionaries with missing index recommendations
        """
        logger.info(f"Identifying potential missing indexes (min_calls={min_calls})")
        
        with self._get_psycopg2_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    # Query to find tables with sequential scans
                    query = """
                    SELECT 
                        schemaname || '.' || relname as table_name,
                        seq_scan as sequential_scans,
                        idx_scan as index_scans,
                        seq_scan + idx_scan as total_scans,
                        CASE WHEN seq_scan + idx_scan > 0 
                            THEN round(100.0 * idx_scan / (seq_scan + idx_scan), 2) 
                            ELSE 0 END as index_scan_percent,
                        pg_size_pretty(pg_total_relation_size(schemaname || '.' || relname)) as table_size
                    FROM pg_stat_user_tables
                    WHERE seq_scan > %s
                    ORDER BY seq_scan DESC
                    """
                    
                    cur.execute(query, (min_calls,))
                    tables_with_seq_scans = cur.fetchall()
                    
                    if not tables_with_seq_scans:
                        logger.info("No tables with high sequential scans found")
                        return []
                    
                    # Convert to list of dictionaries
                    tables = [dict(row) for row in tables_with_seq_scans]
                    
                    # For each table, find columns frequently used in WHERE clauses
                    missing_indexes = []
                    
                    for table_info in tables:
                        table_name = table_info["table_name"]
                        schema, table = table_name.split('.')
                        
                        # Get existing indexes
                        cur.execute("""
                        SELECT 
                            i.relname as index_name,
                            array_to_string(array_agg(a.attname ORDER BY k.indnatts, k.indkey), ', ') as column_names
                        FROM pg_index k
                        JOIN pg_class i ON i.oid = k.indexrelid
                        JOIN pg_class c ON c.oid = k.indrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(k.indkey)
                        WHERE n.nspname = %s AND c.relname = %s
                        GROUP BY i.relname
                        ORDER BY i.relname
                        """, (schema, table))
                        
                        existing_indexes = cur.fetchall()
                        existing_index_columns = set()
                        
                        for idx in existing_indexes:
                            columns = [col.strip() for col in idx["column_names"].split(',')]
                            for col in columns:
                                existing_index_columns.add(col)
                        
                        # Get columns from table
                        cur.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        """, (schema, table))
                        
                        columns = cur.fetchall()
                        
                        # Check for foreign key columns not indexed
                        cur.execute("""
                        SELECT
                            kcu.column_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema = %s
                        AND tc.table_name = %s
                        """, (schema, table))
                        
                        fk_columns = [row["column_name"] for row in cur.fetchall()]
                        
                        # Suggest indexes for FK columns that aren't indexed
                        for fk_column in fk_columns:
                            if fk_column not in existing_index_columns:
                                missing_indexes.append({
                                    "table_name": table_name,
                                    "suggested_index": f"CREATE INDEX idx_{table}_{fk_column} ON {table_name} ({fk_column})",
                                    "reason": f"Foreign key column without an index",
                                    "column": fk_column,
                                    "priority": "high"
                                })
                        
                        # Look for columns commonly used in WHERE clauses
                        for col in columns:
                            column_name = col["column_name"]
                            data_type = col["data_type"]
                            
                            # Skip already indexed columns
                            if column_name in existing_index_columns:
                                continue
                            
                            # Check column name patterns that often benefit from indexes
                            is_candidate = False
                            priority = "medium"
                            reason = ""
                            
                            # Common ID columns
                            if column_name.endswith('_id') or column_name == 'id':
                                is_candidate = True
                                reason = "ID column"
                                priority = "high"
                            
                            # Date/timestamp columns often used for filtering
                            elif data_type in ('date', 'timestamp', 'timestamptz'):
                                is_candidate = True
                                reason = "Date/timestamp column"
                                priority = "high"
                            
                            # Status columns
                            elif column_name in ('status', 'state', 'type', 'category'):
                                is_candidate = True
                                reason = "Status/category column"
                                
                                # Create a partial index if it's a boolean or has few values
                                if data_type in ('boolean', 'varchar', 'char'):
                                    reason += " (consider partial index)"
                            
                            # Boolean flags
                            elif data_type == 'boolean':
                                is_candidate = True
                                reason = "Boolean flag (consider partial index)"
                                
                            if is_candidate:
                                missing_indexes.append({
                                    "table_name": table_name,
                                    "suggested_index": f"CREATE INDEX idx_{table}_{column_name} ON {table_name} ({column_name})",
                                    "reason": reason,
                                    "column": column_name,
                                    "priority": priority
                                })
                    
                    # Log summary of findings
                    logger.info(f"Found {len(missing_indexes)} potential missing indexes")
                    
                    # Print high priority suggestions
                    high_priority = [idx for idx in missing_indexes if idx["priority"] == "high"]
                    logger.info(f"High priority index suggestions: {len(high_priority)}")
                    
                    for i, idx in enumerate(high_priority):
                        logger.info(f"  Index #{i+1}: {idx['suggested_index']}")
                        logger.info(f"    Reason: {idx['reason']}")
                    
                    return missing_indexes
                    
                except psycopg2.Error as e:
                    logger.error(f"Error identifying missing indexes: {e}")
                    return []
    
    def suggest_query_optimizations(self, queries: List[Dict]) -> List[Dict]:
        """Suggest optimizations for slow queries.
        
        Args:
            queries: List of query info dictionaries from analyze_slow_queries
            
        Returns:
            List of dictionaries with optimization suggestions
        """
        logger.info(f"Suggesting optimizations for {len(queries)} queries")
        
        suggestions = []
        
        with self._get_psycopg2_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for query_info in queries:
                    query = query_info["query"]
                    
                    try:
                        # Get query execution plan
                        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                        
                        # We need to be careful here, as the query might have parameters
                        # Replace any parameters with literals for EXPLAIN
                        # This is a simplified approach and might not work for all queries
                        if "$1" in explain_query:
                            logger.info("Query contains parameters, skipping EXPLAIN ANALYZE")
                            continue
                        
                        cur.execute(explain_query)
                        plan = cur.fetchone()[0]
                        
                        # Extract key information from plan
                        plan_data = plan[0]["Plan"]
                        execution_time = plan[0]["Execution Time"]
                        planning_time = plan[0]["Planning Time"]
                        
                        suggestion = {
                            "query": query,
                            "execution_time_ms": execution_time,
                            "planning_time_ms": planning_time,
                            "suggestions": [],
                            "plan": plan
                        }
                        
                        # Check for sequence scans on large tables
                        self._find_seq_scans(plan_data, suggestion)
                        
                        # Check for hash joins with high memory usage
                        self._find_expensive_joins(plan_data, suggestion)
                        
                        # Check for sorts that could be avoided
                        self._find_expensive_sorts(plan_data, suggestion)
                        
                        # Check for inefficient index usage
                        self._find_inefficient_index_usage(plan_data, suggestion)
                        
                        if suggestion["suggestions"]:
                            suggestions.append(suggestion)
                        
                    except psycopg2.Error as e:
                        logger.warning(f"Error analyzing query plan: {str(e)[:200]}")
                        continue
        
        # Log summary of findings
        logger.info(f"Generated {len(suggestions)} query optimization suggestions")
        
        # Print top suggestions
        for i, suggestion in enumerate(suggestions[:3]):
            logger.info(f"Query optimization #{i+1}:")
            logger.info(f"  Execution time: {suggestion['execution_time_ms']}ms")
            logger.info(f"  Query: {suggestion['query'][:100]}...")
            
            for j, tip in enumerate(suggestion["suggestions"]):
                logger.info(f"  Suggestion {j+1}: {tip}")
        
        return suggestions
    
    def _find_seq_scans(self, plan: Dict, suggestion: Dict, current_path: str = ""):
        """Recursively find sequential scans in the plan.
        
        Args:
            plan: Query plan node
            suggestion: Suggestion dictionary to update
            current_path: Current path in the plan tree
        """
        node_type = plan.get("Node Type")
        
        if node_type == "Seq Scan":
            relation_name = plan.get("Relation Name")
            rows = plan.get("Plan Rows", 0)
            
            if rows > 1000:
                table_size = self._get_table_size(relation_name)
                
                if table_size and table_size > 10 * 1024 * 1024:  # 10 MB
                    filter_cond = plan.get("Filter", "")
                    
                    if filter_cond:
                        suggestion["suggestions"].append(
                            f"Consider adding an index to {relation_name} for condition: {filter_cond}"
                        )
                    else:
                        suggestion["suggestions"].append(
                            f"Large sequential scan on {relation_name} ({self._format_bytes(table_size)})"
                        )
        
        # Recursively check child plans
        if "Plans" in plan:
            for i, child in enumerate(plan["Plans"]):
                child_path = f"{current_path}/{i}" if current_path else str(i)
                self._find_seq_scans(child, suggestion, child_path)
    
    def _find_expensive_joins(self, plan: Dict, suggestion: Dict, current_path: str = ""):
        """Recursively find expensive joins in the plan.
        
        Args:
            plan: Query plan node
            suggestion: Suggestion dictionary to update
            current_path: Current path in the plan tree
        """
        node_type = plan.get("Node Type")
        
        if "Join" in node_type:
            rows = plan.get("Plan Rows", 0)
            actual_rows = plan.get("Actual Rows", 0)
            
            # Check for significant row estimation error
            if actual_rows > 0 and rows > 0:
                estimation_ratio = actual_rows / rows
                
                if estimation_ratio > 10 or estimation_ratio < 0.1:
                    suggestion["suggestions"].append(
                        f"{node_type}: Row estimation error ({rows} estimated vs {actual_rows} actual). "
                        "Consider running ANALYZE on the involved tables."
                    )
            
            # Check for hash joins with high memory usage
            if node_type == "Hash Join":
                hash_mem = plan.get("Hash Cond", "")
                peak_memory = plan.get("Peak Memory Usage", 0)
                
                if peak_memory > 1000:  # More than 1000 KB
                    suggestion["suggestions"].append(
                        f"Hash Join using significant memory ({peak_memory} KB) for condition {hash_mem}. "
                        "Consider optimizing join condition."
                    )
        
        # Recursively check child plans
        if "Plans" in plan:
            for i, child in enumerate(plan["Plans"]):
                child_path = f"{current_path}/{i}" if current_path else str(i)
                self._find_expensive_joins(child, suggestion, child_path)
    
    def _find_expensive_sorts(self, plan: Dict, suggestion: Dict, current_path: str = ""):
        """Recursively find expensive sorts in the plan.
        
        Args:
            plan: Query plan node
            suggestion: Suggestion dictionary to update
            current_path: Current path in the plan tree
        """
        node_type = plan.get("Node Type")
        
        if node_type == "Sort":
            sort_mem = plan.get("Sort Memory Used", 0)
            sort_method = plan.get("Sort Method", "")
            sort_key = plan.get("Sort Key", [])
            
            if "disk" in sort_method.lower():
                suggestion["suggestions"].append(
                    f"Sort spilling to disk for keys: {sort_key}. "
                    "Consider increasing work_mem or adding an index to avoid sorting."
                )
            elif sort_mem > 10000:  # More than 10MB
                suggestion["suggestions"].append(
                    f"Memory-intensive sort ({sort_mem} KB) for keys: {sort_key}. "
                    "Consider adding an index matching the sort order."
                )
        
        # Recursively check child plans
        if "Plans" in plan:
            for i, child in enumerate(plan["Plans"]):
                child_path = f"{current_path}/{i}" if current_path else str(i)
                self._find_expensive_sorts(child, suggestion, child_path)
    
    def _find_inefficient_index_usage(self, plan: Dict, suggestion: Dict, current_path: str = ""):
        """Recursively find inefficient index usage in the plan.
        
        Args:
            plan: Query plan node
            suggestion: Suggestion dictionary to update
            current_path: Current path in the plan tree
        """
        node_type = plan.get("Node Type")
        
        if node_type == "Index Scan":
            index_name = plan.get("Index Name", "")
            index_cond = plan.get("Index Cond", "")
            rows = plan.get("Plan Rows", 0)
            actual_rows = plan.get("Actual Rows", 0)
            
            # Check for index scans returning many rows (potentially inefficient)
            if actual_rows > 1000 and "Bitmap" not in node_type:
                # This might be an inefficient index scan
                suggestion["suggestions"].append(
                    f"Index scan on {index_name} returning {actual_rows} rows. "
                    "Consider using a bitmap index scan instead."
                )
        
        # Recursively check child plans
        if "Plans" in plan:
            for i, child in enumerate(plan["Plans"]):
                child_path = f"{current_path}/{i}" if current_path else str(i)
                self._find_inefficient_index_usage(child, suggestion, child_path)
    
    def _get_table_size(self, table_name: str) -> Optional[int]:
        """Get the size of a table in bytes.
        
        Args:
            table_name: Table name
            
        Returns:
            Table size in bytes, or None if table not found
        """
        with self._get_psycopg2_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Handle schema-qualified table names
                    if "." in table_name:
                        schema, table = table_name.split(".")
                        cur.execute(
                            "SELECT pg_total_relation_size(%s)",
                            (f"{schema}.{table}",)
                        )
                    else:
                        cur.execute(
                            "SELECT pg_total_relation_size(%s)",
                            (table_name,)
                        )
                    
                    result = cur.fetchone()
                    return result[0] if result else None
                    
                except psycopg2.Error:
                    return None
    
    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes as human-readable string.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
    
    def create_indexes(self, index_suggestions: List[Dict], dry_run: bool = True):
        """Create suggested indexes.
        
        Args:
            index_suggestions: List of index suggestions
            dry_run: If True, only print SQL without executing
            
        Returns:
            Number of indexes created
        """
        created_count = 0
        
        with self._get_psycopg2_connection() as conn:
            with conn.cursor() as cur:
                for suggestion in index_suggestions:
                    sql = suggestion["suggested_index"]
                    
                    try:
                        if dry_run:
                            logger.info(f"Would create index: {sql}")
                        else:
                            logger.info(f"Creating index: {sql}")
                            cur.execute(sql)
                            created_count += 1
                    except psycopg2.Error as e:
                        logger.error(f"Error creating index: {str(e)}")
        
        if not dry_run:
            conn.commit()
            logger.info(f"Created {created_count} indexes")
        else:
            logger.info(f"Would create {len(index_suggestions)} indexes (dry run)")
        
        return created_count
    
    def vacuum_analyze(self, tables: Optional[List[str]] = None):
        """Perform VACUUM ANALYZE on tables.
        
        Args:
            tables: List of tables to vacuum, or None for all tables
        """
        with self._get_psycopg2_connection() as conn:
            # Autocommit must be enabled for VACUUM
            conn.autocommit = True
            with conn.cursor() as cur:
                if tables:
                    for table in tables:
                        try:
                            logger.info(f"Running VACUUM ANALYZE on {table}")
                            cur.execute(f"VACUUM ANALYZE {table}")
                        except psycopg2.Error as e:
                            logger.error(f"Error vacuuming table {table}: {str(e)}")
                else:
                    try:
                        logger.info("Running VACUUM ANALYZE on all tables")
                        cur.execute("VACUUM ANALYZE")
                        logger.info("VACUUM ANALYZE completed successfully")
                    except psycopg2.Error as e:
                        logger.error(f"Error vacuuming tables: {str(e)}")
    
    def reindex_tables(self, tables: Optional[List[str]] = None):
        """Reindex tables to optimize index performance.
        
        Args:
            tables: List of tables to reindex, or None for all tables
        """
        with self._get_psycopg2_connection() as conn:
            # Autocommit must be enabled for REINDEX
            conn.autocommit = True
            with conn.cursor() as cur:
                if tables:
                    for table in tables:
                        try:
                            logger.info(f"Reindexing table {table}")
                            cur.execute(f"REINDEX TABLE {table}")
                        except psycopg2.Error as e:
                            logger.error(f"Error reindexing table {table}: {str(e)}")
                else:
                    try:
                        logger.info("Reindexing all tables (this might take a while)")
                        cur.execute("REINDEX DATABASE CONCURRENTLY %s", (self.conn_params["dbname"],))
                        logger.info("Reindexing completed successfully")
                    except psycopg2.Error as e:
                        logger.error(f"Error reindexing database: {str(e)}")
                        logger.info("Falling back to non-concurrent reindexing")
                        try:
                            cur.execute("REINDEX DATABASE %s", (self.conn_params["dbname"],))
                            logger.info("Non-concurrent reindexing completed successfully")
                        except psycopg2.Error as e2:
                            logger.error(f"Error with non-concurrent reindexing: {str(e2)}")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze and optimize database performance."
    )
    
    # Database connection arguments
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory"),
        help="Database connection URL",
    )
    
    # Analysis operations
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze slow queries",
    )
    
    parser.add_argument(
        "--min-calls",
        type=int,
        default=10,
        help="Minimum query calls for analysis",
    )
    
    parser.add_argument(
        "--min-time",
        type=int,
        default=100,
        help="Minimum average query time in milliseconds",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of queries to analyze",
    )
    
    # Index suggestions
    parser.add_argument(
        "--suggest-indexes",
        action="store_true",
        help="Suggest missing indexes",
    )
    
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create suggested indexes",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually create indexes, just print SQL",
    )
    
    # Maintenance operations
    parser.add_argument(
        "--vacuum-analyze",
        action="store_true",
        help="Run VACUUM ANALYZE on tables",
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex tables",
    )
    
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        help="Specific tables for vacuum or reindex operations",
    )
    
    # Logging configuration
    parser.add_argument(
        "--enable-query-logging",
        action="store_true",
        help="Enable logging of slow queries",
    )
    
    parser.add_argument(
        "--min-duration-ms",
        type=int,
        default=100,
        help="Minimum query duration to log in milliseconds",
    )
    
    # Combined operations
    parser.add_argument(
        "--optimize-all",
        action="store_true",
        help="Run all optimization steps (analyze, suggest indexes, vacuum, reindex)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for analysis results (JSON format)",
    )
    
    return parser.parse_args()


def main():
    """Run the script."""
    args = parse_args()
    
    # Initialize optimizer
    optimizer = DBPerformanceOptimizer(db_url=args.db_url)
    
    # Record start time
    start_time = time.time()
    
    results = {}
    
    # Query logging
    if args.enable_query_logging or args.optimize_all:
        logger.info(f"Enabling slow query logging (min_duration_ms={args.min_duration_ms})")
        try:
            optimizer.enable_query_logging(min_duration_ms=args.min_duration_ms)
        except Exception as e:
            logger.error(f"Failed to enable query logging: {str(e)}")
    
    # Analyze slow queries
    if args.analyze or args.optimize_all:
        logger.info("Analyzing slow queries")
        try:
            slow_queries = optimizer.analyze_slow_queries(
                min_calls=args.min_calls,
                min_avg_time_ms=args.min_time,
                limit=args.limit
            )
            results["slow_queries"] = slow_queries
            
            # Suggest query optimizations
            if slow_queries:
                logger.info("Suggesting query optimizations")
                query_suggestions = optimizer.suggest_query_optimizations(slow_queries)
                results["query_suggestions"] = query_suggestions
        except Exception as e:
            logger.error(f"Failed to analyze slow queries: {str(e)}")
    
    # Suggest missing indexes
    if args.suggest_indexes or args.optimize_all:
        logger.info("Suggesting missing indexes")
        try:
            missing_indexes = optimizer.identify_missing_indexes(min_calls=args.min_calls)
            results["missing_indexes"] = missing_indexes
            
            # Create suggested indexes if requested
            if args.create_indexes or args.optimize_all:
                logger.info("Creating suggested indexes")
                optimizer.create_indexes(missing_indexes, dry_run=args.dry_run)
        except Exception as e:
            logger.error(f"Failed to suggest indexes: {str(e)}")
    
    # Vacuum analyze
    if args.vacuum_analyze or args.optimize_all:
        logger.info("Running VACUUM ANALYZE")
        try:
            optimizer.vacuum_analyze(tables=args.tables)
        except Exception as e:
            logger.error(f"Failed to vacuum tables: {str(e)}")
    
    # Reindex tables
    if args.reindex or args.optimize_all:
        logger.info("Reindexing tables")
        try:
            optimizer.reindex_tables(tables=args.tables)
        except Exception as e:
            logger.error(f"Failed to reindex tables: {str(e)}")
    
    # Record end time and total duration
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"All operations completed in {total_duration:.2f} seconds")
    
    # Write results to file if requested
    if args.output_file and results:
        try:
            # Convert complex objects to serializable format
            serializable_results = json.dumps(results, default=str, indent=2)
            
            with open(args.output_file, 'w') as f:
                f.write(serializable_results)
            
            logger.info(f"Results written to {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write results: {str(e)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 