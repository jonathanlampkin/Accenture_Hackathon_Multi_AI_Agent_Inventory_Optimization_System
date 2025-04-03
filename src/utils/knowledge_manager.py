from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from copy import deepcopy
import asyncio
from dataclasses import dataclass
from enum import Enum

class KnowledgeStatus(Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"

@dataclass
class KnowledgeVersion:
    version_id: str
    content: Dict[str, Any]
    status: KnowledgeStatus
    created_at: datetime
    created_by: str
    validation_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeManager:
    def __init__(self, storage_dir: str = "knowledge"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.knowledge_bases: Dict[str, List[KnowledgeVersion]] = {}
        self.validation_rules: Dict[str, List[callable]] = {}
        self.sharing_rules: Dict[str, List[str]] = {}
        
        # Initialize logging
        self.logger = logging.getLogger("knowledge_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.storage_dir / f"knowledge_manager_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
    def create_knowledge_base(self, name: str, description: str) -> None:
        """Create a new knowledge base"""
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base {name} already exists")
            
        self.knowledge_bases[name] = []
        self.validation_rules[name] = []
        self.sharing_rules[name] = []
        
        # Save metadata
        metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "versions": []
        }
        
        metadata_file = self.storage_dir / f"{name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Created knowledge base {name}")
    
    def add_validation_rule(self, knowledge_base: str, rule: callable) -> None:
        """Add a validation rule to a knowledge base"""
        if knowledge_base not in self.validation_rules:
            raise ValueError(f"Knowledge base {knowledge_base} does not exist")
            
        self.validation_rules[knowledge_base].append(rule)
        self.logger.info(f"Added validation rule to {knowledge_base}")
    
    def add_sharing_rule(self, knowledge_base: str, agent_id: str) -> None:
        """Add a sharing rule to a knowledge base"""
        if knowledge_base not in self.sharing_rules:
            raise ValueError(f"Knowledge base {knowledge_base} does not exist")
            
        self.sharing_rules[knowledge_base].append(agent_id)
        self.logger.info(f"Added sharing rule for {agent_id} to {knowledge_base}")
    
    async def add_knowledge(
        self,
        knowledge_base: str,
        content: Dict[str, Any],
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeVersion:
        """Add new knowledge to a knowledge base"""
        if knowledge_base not in self.knowledge_bases:
            raise ValueError(f"Knowledge base {knowledge_base} does not exist")
            
        # Create version
        version_id = self._generate_version_id(content)
        version = KnowledgeVersion(
            version_id=version_id,
            content=deepcopy(content),
            status=KnowledgeStatus.DRAFT,
            created_at=datetime.now(),
            created_by=created_by,
            metadata=metadata
        )
        
        # Validate knowledge
        validation_results = await self._validate_knowledge(knowledge_base, version)
        version.validation_results = validation_results
        
        if all(result["valid"] for result in validation_results.values()):
            version.status = KnowledgeStatus.VALIDATED
            self.logger.info(f"Knowledge validated in {knowledge_base}")
        else:
            self.logger.warning(f"Knowledge validation failed in {knowledge_base}")
            
        # Add version
        self.knowledge_bases[knowledge_base].append(version)
        
        # Save version
        self._save_version(knowledge_base, version)
        
        return version
    
    async def _validate_knowledge(
        self,
        knowledge_base: str,
        version: KnowledgeVersion
    ) -> Dict[str, Dict[str, Any]]:
        """Validate knowledge using all rules"""
        results = {}
        
        for rule in self.validation_rules[knowledge_base]:
            try:
                result = await rule(version.content)
                results[rule.__name__] = {
                    "valid": result,
                    "message": "Validation passed" if result else "Validation failed"
                }
            except Exception as e:
                results[rule.__name__] = {
                    "valid": False,
                    "message": f"Validation error: {str(e)}"
                }
                
        return results
    
    def _generate_version_id(self, content: Dict[str, Any]) -> str:
        """Generate a unique version ID"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:8]
    
    def _save_version(self, knowledge_base: str, version: KnowledgeVersion) -> None:
        """Save a knowledge version"""
        version_dir = self.storage_dir / knowledge_base / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save content
        content_file = version_dir / "content.json"
        with open(content_file, 'w') as f:
            json.dump(version.content, f, indent=2)
            
        # Save metadata
        metadata = {
            "version_id": version.version_id,
            "status": version.status.value,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by,
            "validation_results": version.validation_results,
            "metadata": version.metadata
        }
        
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_knowledge(
        self,
        knowledge_base: str,
        version_id: Optional[str] = None,
        status: Optional[KnowledgeStatus] = None
    ) -> List[KnowledgeVersion]:
        """Get knowledge from a knowledge base"""
        if knowledge_base not in self.knowledge_bases:
            raise ValueError(f"Knowledge base {knowledge_base} does not exist")
            
        versions = self.knowledge_bases[knowledge_base]
        
        if version_id:
            versions = [v for v in versions if v.version_id == version_id]
            
        if status:
            versions = [v for v in versions if v.status == status]
            
        return versions
    
    def publish_knowledge(self, knowledge_base: str, version_id: str) -> None:
        """Publish a knowledge version"""
        versions = self.get_knowledge(knowledge_base, version_id)
        
        if not versions:
            raise ValueError(f"Version {version_id} not found in {knowledge_base}")
            
        version = versions[0]
        
        if version.status != KnowledgeStatus.VALIDATED:
            raise ValueError(f"Version {version_id} is not validated")
            
        version.status = KnowledgeStatus.PUBLISHED
        self._save_version(knowledge_base, version)
        self.logger.info(f"Published version {version_id} in {knowledge_base}")
    
    def deprecate_knowledge(self, knowledge_base: str, version_id: str) -> None:
        """Deprecate a knowledge version"""
        versions = self.get_knowledge(knowledge_base, version_id)
        
        if not versions:
            raise ValueError(f"Version {version_id} not found in {knowledge_base}")
            
        version = versions[0]
        version.status = KnowledgeStatus.DEPRECATED
        self._save_version(knowledge_base, version)
        self.logger.info(f"Deprecated version {version_id} in {knowledge_base}")
    
    def get_knowledge_metrics(self) -> Dict[str, Any]:
        """Get knowledge management metrics"""
        metrics = {
            "total_bases": len(self.knowledge_bases),
            "total_versions": sum(len(versions) for versions in self.knowledge_bases.values()),
            "bases": {}
        }
        
        for name, versions in self.knowledge_bases.items():
            metrics["bases"][name] = {
                "total_versions": len(versions),
                "draft_versions": len([v for v in versions if v.status == KnowledgeStatus.DRAFT]),
                "validated_versions": len([v for v in versions if v.status == KnowledgeStatus.VALIDATED]),
                "published_versions": len([v for v in versions if v.status == KnowledgeStatus.PUBLISHED]),
                "deprecated_versions": len([v for v in versions if v.status == KnowledgeStatus.DEPRECATED])
            }
            
        return metrics
    
    def save_knowledge_report(self) -> None:
        """Save knowledge management report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_knowledge_metrics(),
            "knowledge_bases": {
                name: [v.to_dict() for v in versions]
                for name, versions in self.knowledge_bases.items()
            }
        }
        
        report_file = self.storage_dir / f"knowledge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 