#!/usr/bin/env python
"""
Database query benchmark script.

This script tests the performance of important database queries with and without indexes,
allowing for quantitative measurement of performance improvements.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("benchmark")


class QueryBenchmark:
    """Benchmark performance of database queries."""

    def __init__(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
        db_password: Optional[str] = None,
        db_port: int = 5432,
        results_dir: str = "benchmark_results",
    ):
        """Initialize QueryBenchmark.
        
        Args:
            db_host: Database hostname
            db_name: Database name
            db_user: Database user
            db_password: Database password
            db_port: Database port
            results_dir: Directory to store benchmark results
        """
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password or os.environ.get("PGPASSWORD", "")
        self.db_port = db_port
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test database connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test database connection.
        
        Raises:
            Exception: If connection fails
        """
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port,
            )
            conn.close()
            logger.info(f"Successfully connected to database {self.db_name}")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def _execute_query(self, query: str, params: Optional[Dict] = None) -> Tuple[List[Dict], float]:
        """Execute a SQL query and measure execution time.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Tuple containing results and execution time
        """
        conn = None
        try:
            start_time = time.time()
            
            conn = psycopg2.connect(
                host=self.db_host,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port,
            )
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or {})
                results = cursor.fetchall()
                
            execution_time = time.time() - start_time
            return list(results), execution_time
        
        finally:
            if conn:
                conn.close()

    def _get_query_plan(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Get the execution plan for a query.
        
        Args:
            query: SQL query to analyze
            params: Query parameters
            
        Returns:
            Execution plan
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port,
            )
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}", params or {})
                plan = cursor.fetchone()[0]
                
            return plan
        
        finally:
            if conn:
                conn.close()

    def benchmark_query(
        self, name: str, query: str, params: Optional[Dict] = None, iterations: int = 5
    ) -> Dict:
        """Benchmark a query by running it multiple times.
        
        Args:
            name: Name of the query
            query: SQL query to benchmark
            params: Query parameters
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking query: {name}")
        
        # Get query plan first (this also warms up the cache)
        plan = self._get_query_plan(query, params)
        
        # Run the query multiple times
        execution_times = []
        result_counts = []
        
        for i in range(iterations):
            results, execution_time = self._execute_query(query, params)
            execution_times.append(execution_time)
            result_counts.append(len(results))
            logger.info(f"  Iteration {i+1}/{iterations}: {execution_time:.4f} seconds, {len(results)} results")
        
        # Calculate statistics
        stats = {
            "name": name,
            "mean_time": mean(execution_times),
            "median_time": median(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": stdev(execution_times) if len(execution_times) > 1 else 0,
            "iterations": iterations,
            "result_count": result_counts[0],
            "query": query,
            "timestamp": time.time(),
            "plan": plan,
        }
        
        logger.info(f"  Average execution time: {stats['mean_time']:.4f} seconds")
        return stats

    def benchmark_queries_from_file(
        self, query_file: str, iterations: int = 5, output_file: Optional[str] = None
    ) -> List[Dict]:
        """Benchmark queries from a JSON file.
        
        Args:
            query_file: JSON file containing queries
            iterations: Number of iterations per query
            output_file: Output file for results
            
        Returns:
            List of benchmark results
        """
        # Load queries from file
        with open(query_file, "r") as f:
            queries = json.load(f)
        
        # Run benchmarks
        results = []
        for query_info in queries:
            name = query_info["name"]
            query = query_info["query"]
            params = query_info.get("params", {})
            
            result = self.benchmark_query(name, query, params, iterations)
            results.append(result)
        
        # Save results to file if specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved benchmark results to {output_path}")
        
        return results

    def compare_results(self, before_file: str, after_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Compare benchmark results before and after optimization.
        
        Args:
            before_file: File with before results
            after_file: File with after results
            output_file: Output file for comparison
            
        Returns:
            DataFrame with comparison results
        """
        # Load results
        with open(self.results_dir / before_file, "r") as f:
            before = json.load(f)
        
        with open(self.results_dir / after_file, "r") as f:
            after = json.load(f)
        
        # Create comparison DataFrame
        comparison = []
        
        for b in before:
            for a in after:
                if b["name"] == a["name"]:
                    improvement = (b["mean_time"] - a["mean_time"]) / b["mean_time"] * 100
                    comparison.append({
                        "name": b["name"],
                        "before_time": b["mean_time"],
                        "after_time": a["mean_time"],
                        "improvement_pct": improvement,
                        "before_min": b["min_time"],
                        "after_min": a["min_time"],
                        "before_max": b["max_time"],
                        "after_max": a["max_time"],
                    })
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison)
        
        # Sort by improvement percentage
        df = df.sort_values("improvement_pct", ascending=False)
        
        # Save to CSV if specified
        if output_file:
            output_path = self.results_dir / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"Saved comparison to {output_path}")
        
        return df


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Benchmark database queries to measure index performance."
    )
    
    parser.add_argument(
        "--db-host",
        type=str,
        default=os.environ.get("DB_HOST", "localhost"),
        help="Database host",
    )
    
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.environ.get("DB_PORT", "5432")),
        help="Database port",
    )
    
    parser.add_argument(
        "--db-name",
        type=str,
        default=os.environ.get("DB_NAME", "inventory"),
        help="Database name",
    )
    
    parser.add_argument(
        "--db-user",
        type=str,
        default=os.environ.get("DB_USER", "postgres"),
        help="Database user",
    )
    
    parser.add_argument(
        "--db-password",
        type=str,
        default=os.environ.get("DB_PASSWORD"),
        help="Database password",
    )
    
    parser.add_argument(
        "--query-file",
        type=str,
        required=True,
        help="JSON file containing queries to benchmark",
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per query",
    )
    
    parser.add_argument(
        "--before-indexes",
        action="store_true",
        help="Run benchmark before applying indexes",
    )
    
    parser.add_argument(
        "--after-indexes",
        action="store_true",
        help="Run benchmark after applying indexes",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare before and after results",
    )
    
    parser.add_argument(
        "--before-file",
        type=str,
        default="before_indexes.json",
        help="File containing before results",
    )
    
    parser.add_argument(
        "--after-file",
        type=str,
        default="after_indexes.json",
        help="File containing after results",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to store benchmark results",
    )
    
    return parser.parse_args()


def main():
    """Run the benchmark."""
    args = parse_args()
    
    # Initialize benchmark
    benchmark = QueryBenchmark(
        db_host=args.db_host,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_port=args.db_port,
        results_dir=args.output_dir,
    )
    
    if args.before_indexes:
        # Run benchmark before indexes
        logger.info("Running benchmark before applying indexes")
        benchmark.benchmark_queries_from_file(
            args.query_file,
            iterations=args.iterations,
            output_file=args.before_file,
        )
    
    if args.after_indexes:
        # Run benchmark after indexes
        logger.info("Running benchmark after applying indexes")
        benchmark.benchmark_queries_from_file(
            args.query_file,
            iterations=args.iterations,
            output_file=args.after_file,
        )
    
    if args.compare:
        # Compare results
        logger.info("Comparing before and after results")
        df = benchmark.compare_results(
            args.before_file,
            args.after_file,
            output_file="comparison.csv",
        )
        
        # Print comparison table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print("\nPerformance Comparison:")
        print(df.to_string(index=False))
        
        # Print summary
        mean_improvement = df["improvement_pct"].mean()
        print(f"\nAverage performance improvement: {mean_improvement:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 