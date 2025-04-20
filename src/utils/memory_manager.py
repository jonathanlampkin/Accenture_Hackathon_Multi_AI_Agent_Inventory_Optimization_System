from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib

class MemoryPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class MemoryStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"

@dataclass
class MemoryItem:
    memory_id: str
    content: Dict[str, Any]
    priority: MemoryPriority
    status: MemoryStatus
    created_at: datetime
    expires_at: Optional[datetime]
    accessed_at: datetime
    access_count: int
    metadata: Optional[Dict[str, Any]] = None

class MemoryManager:
    def __init__(self, storage_dir: str = "memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.memories: Dict[str, MemoryItem] = {}
        self.shared_memories: Dict[str, List[str]] = {}
        self.cleanup_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize logging
        self.logger = logging.getLogger("memory_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.storage_dir / f"memory_manager_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
    def add_memory(
        self,
        content: Dict[str, Any],
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new memory"""
        memory_id = self._generate_memory_id(content)
        
        memory = MemoryItem(
            memory_id=memory_id,
            content=content,
            priority=priority,
            status=MemoryStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=expires_at,
            accessed_at=datetime.now(),
            access_count=0,
            metadata=metadata
        )
        
        self.memories[memory_id] = memory
        self._save_memory(memory)
        
        # Start cleanup task if expiration is set
        if expires_at:
            self._start_cleanup_task(memory_id, expires_at)
            
        self.logger.info(f"Added memory {memory_id}")
        return memory_id
    
    def _generate_memory_id(self, content: Dict[str, Any]) -> str:
        """Generate a unique memory ID"""
        content_str = json.dumps(content, sort_keys=True)
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{content_str}{timestamp}".encode()).hexdigest()[:8]
    
    def _save_memory(self, memory: MemoryItem) -> None:
        """Save a memory to storage"""
        memory_dir = self.storage_dir / memory.memory_id
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Save content
        content_file = memory_dir / "content.json"
        with open(content_file, 'w') as f:
            json.dump(memory.content, f, indent=2)
            
        # Save metadata
        metadata = {
            "memory_id": memory.memory_id,
            "priority": memory.priority.value,
            "status": memory.status.value,
            "created_at": memory.created_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            "accessed_at": memory.accessed_at.isoformat(),
            "access_count": memory.access_count,
            "metadata": memory.metadata
        }
        
        metadata_file = memory_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _start_cleanup_task(self, memory_id: str, expires_at: datetime) -> None:
        """Start a task to clean up expired memory"""
        async def cleanup():
            now = datetime.now()
            if now >= expires_at:
                self.archive_memory(memory_id)
                if memory_id in self.cleanup_tasks:
                    del self.cleanup_tasks[memory_id]
            else:
                # Reschedule cleanup
                delay = (expires_at - now).total_seconds()
                await asyncio.sleep(delay)
                self.archive_memory(memory_id)
                if memory_id in self.cleanup_tasks:
                    del self.cleanup_tasks[memory_id]
                
        self.cleanup_tasks[memory_id] = asyncio.create_task(cleanup())
    
    def get_memory(self, memory_id: str, agent_id: Optional[str] = None) -> Optional[MemoryItem]:
        """Get a memory by ID"""
        if memory_id not in self.memories:
            return None
            
        memory = self.memories[memory_id]
        
        # Check sharing rules
        if agent_id and memory_id in self.shared_memories:
            if agent_id not in self.shared_memories[memory_id]:
                return None
                
        # Update access info
        memory.accessed_at = datetime.now()
        memory.access_count += 1
        self._save_memory(memory)
        
        return memory
    
    def get_memories(
        self,
        priority: Optional[MemoryPriority] = None,
        status: Optional[MemoryStatus] = None,
        agent_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """Get memories matching criteria"""
        memories = list(self.memories.values())
        
        if priority:
            memories = [m for m in memories if m.priority == priority]
            
        if status:
            memories = [m for m in memories if m.status == status]
            
        if agent_id:
            memories = [
                m for m in memories
                if m.memory_id not in self.shared_memories
                or agent_id in self.shared_memories[m.memory_id]
            ]
            
        return memories
    
    def share_memory(self, memory_id: str, agent_id: str) -> None:
        """Share a memory with an agent"""
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} does not exist")
            
        if memory_id not in self.shared_memories:
            self.shared_memories[memory_id] = []
            
        if agent_id not in self.shared_memories[memory_id]:
            self.shared_memories[memory_id].append(agent_id)
            self.logger.info(f"Shared memory {memory_id} with {agent_id}")
    
    def archive_memory(self, memory_id: str) -> None:
        """Archive a memory"""
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} does not exist")
            
        memory = self.memories[memory_id]
        memory.status = MemoryStatus.ARCHIVED
        self._save_memory(memory)
        self.logger.info(f"Archived memory {memory_id}")
    
    def expire_memory(self, memory_id: str) -> None:
        """Expire a memory"""
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} does not exist")
            
        memory = self.memories[memory_id]
        memory.status = MemoryStatus.EXPIRED
        self._save_memory(memory)
        self.logger.info(f"Expired memory {memory_id}")
    
    def cleanup_memories(self) -> None:
        """Clean up expired memories"""
        now = datetime.now()
        
        for memory_id, memory in list(self.memories.items()):
            if memory.status == MemoryStatus.EXPIRED:
                # Remove expired memories
                del self.memories[memory_id]
                if memory_id in self.shared_memories:
                    del self.shared_memories[memory_id]
                if memory_id in self.cleanup_tasks:
                    self.cleanup_tasks[memory_id].cancel()
                    del self.cleanup_tasks[memory_id]
                    
                # Remove storage
                memory_dir = self.storage_dir / memory_id
                if memory_dir.exists():
                    for file in memory_dir.iterdir():
                        file.unlink()
                    memory_dir.rmdir()
                    
                self.logger.info(f"Cleaned up memory {memory_id}")
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory management metrics"""
        metrics = {
            "total_memories": len(self.memories),
            "active_memories": len([m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]),
            "archived_memories": len([m for m in self.memories.values() if m.status == MemoryStatus.ARCHIVED]),
            "expired_memories": len([m for m in self.memories.values() if m.status == MemoryStatus.EXPIRED]),
            "shared_memories": len(self.shared_memories),
            "memory_usage": self._calculate_memory_usage(),
            "access_patterns": self._analyze_access_patterns()
        }
        
        return metrics
    
    def _calculate_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage by priority"""
        usage = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }
        
        for memory in self.memories.values():
            if memory.status == MemoryStatus.ACTIVE:
                usage[memory.priority.name.lower()] += 1
                
        return usage
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        patterns = {
            "total_accesses": sum(m.access_count for m in self.memories.values()),
            "average_accesses": sum(m.access_count for m in self.memories.values()) / len(self.memories) if self.memories else 0,
            "recent_accesses": len([m for m in self.memories.values() if (datetime.now() - m.accessed_at).total_seconds() < 3600]),
            "access_by_priority": {
                priority.name.lower(): sum(m.access_count for m in self.memories.values() if m.priority == priority)
                for priority in MemoryPriority
            }
        }
        
        return patterns
    
    def save_memory_report(self) -> None:
        """Save memory management report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_memory_metrics(),
            "memories": {
                memory_id: {
                    "content": memory.content,
                    "priority": memory.priority.value,
                    "status": memory.status.value,
                    "created_at": memory.created_at.isoformat(),
                    "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                    "accessed_at": memory.accessed_at.isoformat(),
                    "access_count": memory.access_count,
                    "metadata": memory.metadata,
                    "shared_with": self.shared_memories.get(memory_id, [])
                }
                for memory_id, memory in self.memories.items()
            }
        }
        
        report_file = self.storage_dir / f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 