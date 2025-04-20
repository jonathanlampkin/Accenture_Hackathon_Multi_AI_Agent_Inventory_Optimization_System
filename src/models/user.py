"""
User-related database models for authentication and authorization.
"""
import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Index, Integer, String, Table, Text
from sqlalchemy.orm import relationship

from src.models.database import Base

# Many-to-many relationship between users and roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    # Add index for faster lookups
    Index("ix_user_roles_user_id", "user_id"),
    Index("ix_user_roles_role_id", "role_id"),
)

class UserStatus(enum.Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class Role(Base):
    """Role model for user permissions."""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    permissions = Column(String(255), nullable=True)  # Comma-separated list of permissions
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self) -> str:
        return f"<Role {self.name}>"

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    status = Column(Enum(UserStatus), nullable=False, default=UserStatus.PENDING)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    tokens = relationship("Token", back_populates="user")
    
    def __repr__(self) -> str:
        return f"<User {self.username}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_users_status', 'status'),  # For filtering by status
        Index('ix_users_is_active', 'is_active'),  # For filtering active users
        Index('ix_users_last_login', 'last_login'),  # For finding users by last login time
        Index('ix_users_full_name', 'full_name'),  # For searching by name
    )

class TokenType(enum.Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET_PASSWORD = "reset_password"
    EMAIL_VERIFICATION = "email_verification"
    API_KEY = "api_key"

class Token(Base):
    """Token model for user authentication."""
    __tablename__ = "tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, index=True, nullable=False)
    token_type = Column(Enum(TokenType), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="tokens")
    
    def __repr__(self) -> str:
        return f"<Token {self.token_type.value} for User {self.user_id}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_tokens_token_type', 'token_type'),  # For filtering by token type
        Index('ix_tokens_user_id', 'user_id'),  # For finding tokens for a specific user
        Index('ix_tokens_expires_at', 'expires_at'),  # For finding expired tokens
        Index('ix_tokens_is_revoked', 'is_revoked'),  # For filtering revoked tokens
    )

class AuditLog(Base):
    """Audit log model for tracking user actions."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 can be up to 45 chars
    user_agent = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by User {self.user_id}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_audit_logs_user_id', 'user_id'),  # For finding logs for a specific user
        Index('ix_audit_logs_action', 'action'),  # For filtering by action
        Index('ix_audit_logs_entity_type', 'entity_type'),  # For filtering by entity type
        Index('ix_audit_logs_entity_id', 'entity_id'),  # For finding logs for a specific entity
        Index('ix_audit_logs_created_at', 'created_at'),  # For time-based filtering
    ) 