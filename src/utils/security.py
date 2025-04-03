from typing import Dict, List, Any, Optional, Callable
import json
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import hmac
import base64
import secrets
from dataclasses import dataclass
from enum import Enum
import re
import jwt
from functools import wraps

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"

@dataclass
class ValidationRule:
    name: str
    pattern: str
    description: str
    severity: SecurityLevel
    validator: Callable[[Any], bool]

class SecurityManager:
    def __init__(self, config_file: str = "security_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.access_controls: Dict[str, List[str]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Initialize logging
        self.logger = logging.getLogger("security_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path("logs") / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules"""
        # Input validation rules
        self.add_validation_rule(
            "no_sql_injection",
            r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|--|;|/\*|\*/)",
            "Prevent SQL injection attacks",
            SecurityLevel.HIGH,
            lambda x: not bool(re.search(r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|--|;|/\*|\*/)", str(x)))
        )
        
        self.add_validation_rule(
            "no_xss",
            r"(?i)(<script|javascript:|on\w+=)",
            "Prevent XSS attacks",
            SecurityLevel.HIGH,
            lambda x: not bool(re.search(r"(?i)(<script|javascript:|on\w+=)", str(x)))
        )
        
        self.add_validation_rule(
            "no_command_injection",
            r"(?i)(;|\||&|`|\$\(|\$\{|\n)",
            "Prevent command injection attacks",
            SecurityLevel.HIGH,
            lambda x: not bool(re.search(r"(?i)(;|\||&|`|\$\(|\$\{|\n)", str(x)))
        )
        
        # Data validation rules
        self.add_validation_rule(
            "valid_email",
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "Validate email format",
            SecurityLevel.MEDIUM,
            lambda x: bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str(x)))
        )
        
        self.add_validation_rule(
            "valid_url",
            r"^https?://[^\s/$.?#].[^\s]*$",
            "Validate URL format",
            SecurityLevel.MEDIUM,
            lambda x: bool(re.match(r"^https?://[^\s/$.?#].[^\s]*$", str(x)))
        )
    
    def add_validation_rule(
        self,
        name: str,
        pattern: str,
        description: str,
        severity: SecurityLevel,
        validator: Callable[[Any], bool]
    ) -> None:
        """Add a validation rule"""
        self.validation_rules[name] = ValidationRule(
            name=name,
            pattern=pattern,
            description=description,
            severity=severity,
            validator=validator
        )
        
        self.logger.info(f"Added validation rule: {name}")
    
    def validate_input(
        self,
        input_data: Any,
        rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate input against rules"""
        results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        rules_to_check = rules or self.validation_rules.keys()
        
        for rule_name in rules_to_check:
            if rule_name not in self.validation_rules:
                continue
                
            rule = self.validation_rules[rule_name]
            try:
                if not rule.validator(input_data):
                    result = {
                        "rule": rule_name,
                        "description": rule.description,
                        "severity": rule.severity.value
                    }
                    
                    if rule.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                        results["valid"] = False
                        results["errors"].append(result)
                    else:
                        results["warnings"].append(result)
                        
            except Exception as e:
                self.logger.error(f"Error validating rule {rule_name}: {str(e)}")
                results["valid"] = False
                results["errors"].append({
                    "rule": rule_name,
                    "description": f"Validation error: {str(e)}",
                    "severity": SecurityLevel.HIGH.value
                })
        
        return results
    
    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            sanitized = re.sub(r"(?i)(<script|javascript:|on\w+=|;|\||&|`|\$\(|\$\{|\n)", "", input_data)
            return sanitized.strip()
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        return input_data
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{base64.b64encode(hashed).decode('utf-8')}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password"""
        try:
            salt, stored_hash = hashed_password.split(':')
            hashed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return hmac.compare_digest(
                base64.b64encode(hashed).decode('utf-8'),
                stored_hash
            )
        except Exception:
            return False
    
    def generate_token(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate a JWT token"""
        return jwt.encode(payload, secret, algorithm='HS256')
    
    def verify_token(self, token: str, secret: str) -> Dict[str, Any]:
        """Verify a JWT token"""
        try:
            return jwt.decode(token, secret, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return {}
    
    def add_access_control(self, resource: str, roles: List[str]) -> None:
        """Add access control for a resource"""
        self.access_controls[resource] = roles
        self.logger.info(f"Added access control for resource: {resource}")
    
    def check_access(self, resource: str, role: str) -> bool:
        """Check if a role has access to a resource"""
        return role in self.access_controls.get(resource, [])
    
    def audit(self, action: str, user: str, details: Dict[str, Any]) -> None:
        """Record an audit log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details
        }
        
        self.audit_log.append(entry)
        self.logger.info(f"Audit log: {action} by {user}")
    
    def get_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user: Optional[str] = None,
        action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered audit log entries"""
        filtered_log = self.audit_log.copy()
        
        if start_time:
            filtered_log = [
                entry for entry in filtered_log
                if datetime.fromisoformat(entry["timestamp"]) >= start_time
            ]
            
        if end_time:
            filtered_log = [
                entry for entry in filtered_log
                if datetime.fromisoformat(entry["timestamp"]) <= end_time
            ]
            
        if user:
            filtered_log = [
                entry for entry in filtered_log
                if entry["user"] == user
            ]
            
        if action:
            filtered_log = [
                entry for entry in filtered_log
                if entry["action"] == action
            ]
            
        return filtered_log
    
    def save_audit_log(self) -> None:
        """Save audit log to file"""
        log_file = Path("logs") / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
    
    def require_role(self, role: str):
        """Decorator to require a specific role"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get user from context or request
                user = kwargs.get('user')
                if not user or not self.check_access(func.__name__, role):
                    raise PermissionError(f"Access denied. Role '{role}' required.")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_validation(self, rules: List[str] = None):
        """Decorator to validate input"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Validate all input arguments
                for arg in args:
                    validation_result = self.validate_input(arg, rules)
                    if not validation_result["valid"]:
                        raise ValueError(f"Input validation failed: {validation_result['errors']}")
                
                for arg in kwargs.values():
                    validation_result = self.validate_input(arg, rules)
                    if not validation_result["valid"]:
                        raise ValueError(f"Input validation failed: {validation_result['errors']}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator 