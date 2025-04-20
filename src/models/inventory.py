"""
Inventory-related database models.
"""
import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from src.models.database import Base

class ProductCategory(enum.Enum):
    """Product category enumeration."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BEVERAGE = "beverage"
    HOUSEHOLD = "household"
    HEALTH = "health"
    BEAUTY = "beauty"
    OTHER = "other"

class Product(Base):
    """Product model."""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(Enum(ProductCategory), default=ProductCategory.OTHER)
    price = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, nullable=False, default=0.0)
    min_stock_level = Column(Integer, nullable=False, default=0)
    max_stock_level = Column(Integer, nullable=True)
    lead_time_days = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    inventories = relationship("Inventory", back_populates="product")
    forecasts = relationship("Forecast", back_populates="product")
    
    def __repr__(self) -> str:
        return f"<Product {self.sku}: {self.name}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_products_category', 'category'),  # For filtering by category
        Index('ix_products_is_active', 'is_active'),  # For filtering active products 
        Index('ix_products_name', 'name'),  # For searching by name
    )

class Location(Base):
    """Location model for warehouses and stores."""
    __tablename__ = "locations"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    address = Column(Text, nullable=True)
    is_warehouse = Column(Boolean, nullable=False, default=False)
    is_store = Column(Boolean, nullable=False, default=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    inventories = relationship("Inventory", back_populates="location")
    
    def __repr__(self) -> str:
        return f"<Location {self.code}: {self.name}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_locations_is_warehouse', 'is_warehouse'),  # For filtering warehouses
        Index('ix_locations_is_store', 'is_store'),  # For filtering stores
        Index('ix_locations_is_active', 'is_active'),  # For filtering active locations
    )

class Inventory(Base):
    """Inventory model for tracking stock levels."""
    __tablename__ = "inventories"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    reserved_quantity = Column(Integer, nullable=False, default=0)
    last_restock_date = Column(DateTime, nullable=True)
    last_count_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="inventories")
    location = relationship("Location", back_populates="inventories")
    
    @property
    def available_quantity(self) -> int:
        """Get available quantity (total - reserved).
        
        Returns:
            int: Available quantity
        """
        return max(0, self.quantity - self.reserved_quantity)
    
    def __repr__(self) -> str:
        return f"<Inventory: {self.product_id} at {self.location_id} - {self.quantity} units>"
    
    # Additional indexes for common queries
    __table_args__ = (
        # Composite index for unique constraint and fast lookups
        Index('ix_inventories_product_location', 'product_id', 'location_id', unique=True),
        # Index for low stock queries
        Index('ix_inventories_quantity', 'quantity'),
        # Index for finding recently updated inventory
        Index('ix_inventories_updated_at', 'updated_at'),
    )

class InventoryTransaction(Base):
    """Inventory transaction model for tracking inventory changes."""
    __tablename__ = "inventory_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    inventory_id = Column(Integer, ForeignKey("inventories.id"), nullable=False)
    quantity_change = Column(Integer, nullable=False)
    previous_quantity = Column(Integer, nullable=False)
    new_quantity = Column(Integer, nullable=False)
    transaction_type = Column(String(50), nullable=False)
    reference_id = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    inventory = relationship("Inventory")
    
    def __repr__(self) -> str:
        return f"<InventoryTransaction: {self.inventory_id} {self.quantity_change:+d} units>"
    
    # Additional indexes for common queries
    __table_args__ = (
        # For filtering transactions by type
        Index('ix_inventory_transactions_type', 'transaction_type'),
        # For filtering transactions by date
        Index('ix_inventory_transactions_created_at', 'created_at'),
        # For finding transactions with reference IDs (e.g., order numbers)
        Index('ix_inventory_transactions_reference_id', 'reference_id'),
    ) 