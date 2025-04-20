"""
Database setup and session management.
"""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Get database URL from environment or use default
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Get database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def init_db() -> None:
    """Initialize database by creating all tables."""
    # Import all models to ensure they are registered with Base
    from src.models.inventory import Inventory, Product, Location
    from src.models.forecast import Forecast, ForecastModel
    from src.models.user import User, Role
    
    # Create all tables
    Base.metadata.create_all(bind=engine) 