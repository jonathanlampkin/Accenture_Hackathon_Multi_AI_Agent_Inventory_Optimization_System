from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import json
from pathlib import Path
import logging
from enum import Enum
import time

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Task:
    def __init__(
        self,
        task_id: str,
        description: str,
        agent_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout: int = 300,
        dependencies: List[str] = None
    ):
        self.task_id = task_id
        self.description = description
        self.agent_id = agent_id
        self.priority = priority
        self.timeout = timeout
        self.dependencies = dependencies or []
        
        self.status = TaskStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_id": self.agent_id,
            "priority": self.priority.value,
            "timeout": self.timeout,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": str(self.result) if self.result else None,
            "error": str(self.error) if self.error else None
        }

class TaskManager:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Initialize logging
        self.logger = logging.getLogger("task_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"task_manager_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
    async def add_task(self, task: Task) -> None:
        """Add a task to the manager"""
        self.tasks[task.task_id] = task
        self.logger.info(f"Added task {task.task_id}: {task.description}")
        
        # Check dependencies
        if not task.dependencies:
            await self.task_queue.put((task.priority.value, task.task_id))
        else:
            # Wait for dependencies
            for dep_id in task.dependencies:
                if dep_id in self.completed_tasks:
                    continue
                elif dep_id in self.failed_tasks:
                    task.status = TaskStatus.FAILED
                    task.error = Exception(f"Dependency {dep_id} failed")
                    self.failed_tasks[task.task_id] = task
                    return
                else:
                    # Wait for dependency to complete
                    while dep_id not in self.completed_tasks:
                        await asyncio.sleep(1)
            
            await self.task_queue.put((task.priority.value, task.task_id))
    
    async def run_task(self, task_id: str, func: callable, *args, **kwargs) -> None:
        """Run a task"""
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        try:
            # Create timeout
            async with asyncio.timeout(task.timeout):
                result = await func(*args, **kwargs)
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.completed_tasks[task_id] = task
                self.logger.info(f"Completed task {task_id}")
                
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = Exception(f"Task {task_id} timed out after {task.timeout} seconds")
            self.failed_tasks[task_id] = task
            self.logger.error(f"Task {task_id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            self.failed_tasks[task_id] = task
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
        finally:
            task.end_time = time.time()
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def process_tasks(self) -> None:
        """Process tasks from the queue"""
        while True:
            try:
                priority, task_id = await self.task_queue.get()
                task = self.tasks[task_id]
                
                if task.status == TaskStatus.PENDING:
                    # Create task coroutine
                    task_coro = self.run_task(
                        task_id,
                        task.func,
                        *task.args,
                        **task.kwargs
                    )
                    
                    # Run task
                    self.running_tasks[task_id] = asyncio.create_task(task_coro)
                    
            except Exception as e:
                self.logger.error(f"Error processing tasks: {str(e)}")
                await asyncio.sleep(1)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return {}
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task metrics"""
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "average_task_time": self._calculate_average_task_time(),
            "task_success_rate": self._calculate_success_rate()
        }
    
    def _calculate_average_task_time(self) -> float:
        """Calculate average task completion time"""
        completed_times = [
            t.end_time - t.start_time
            for t in self.completed_tasks.values()
            if t.end_time and t.start_time
        ]
        return sum(completed_times) / len(completed_times) if completed_times else 0
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        total_completed = len(self.completed_tasks) + len(self.failed_tasks)
        return len(self.completed_tasks) / total_completed if total_completed > 0 else 0
    
    def save_task_report(self) -> None:
        """Save task report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_task_metrics(),
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }
        
        report_file = self.log_dir / f"task_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 