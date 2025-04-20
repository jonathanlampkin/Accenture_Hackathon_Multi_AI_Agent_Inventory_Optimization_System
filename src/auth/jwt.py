"""
JWT Authentication Utilities

This module provides JWT token handling functions for API authentication.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# Configuration - in production, these would be loaded from environment variables
SECRET_KEY = "super_secret_key_for_development_only"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Setup OAuth2 scheme for token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Logger
logger = logging.getLogger(__name__)

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    exp: Optional[datetime] = None

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("Created access token for user %s", data.get("sub"))
    
    return encoded_jwt

def verify_token(token: str) -> Dict:
    """
    Verify JWT token and return decoded payload
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username, exp=payload.get("exp"))
        logger.debug("Token verified for user %s", username)
        return payload
    except jwt.PyJWTError as e:
        logger.warning("Token verification failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Get current user from token
    
    Args:
        token: JWT token from request
        
    Returns:
        User data from token
    """
    return verify_token(token) 