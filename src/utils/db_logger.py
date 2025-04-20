import sqlite3
import logging
from datetime import datetime
import os
from .. import config # Use relative import to get config path

logger = logging.getLogger(__name__)

DB_FILE = os.path.join(config.OUTPUT_DIR, 'inventory_results.db')

def initialize_db():
    """Creates the SQLite database and the results_log table if they don't exist."""
    try:
        # Ensure output directory exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                task_description TEXT,
                agent_role TEXT,
                result_summary TEXT,
                raw_output TEXT 
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at {DB_FILE}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database {DB_FILE}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during DB initialization: {e}", exc_info=True)


def log_result(task_description: str, agent_role: str, result_summary: str, raw_output: str):
    """Logs a task result to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results_log (task_description, agent_role, result_summary, raw_output)
            VALUES (?, ?, ?, ?)
        ''', (task_description, agent_role, result_summary, raw_output))
        conn.commit()
        conn.close()
        logger.debug(f"Logged result for task: {task_description[:50]}... by agent: {agent_role}")
    except sqlite3.Error as e:
        logger.error(f"Error logging result to database {DB_FILE}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during result logging: {e}", exc_info=True)

# Initialize the database when the module is loaded
# initialize_db() # Let's call this explicitly from main.py instead 