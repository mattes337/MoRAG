# Task 5: Task Classification System

## Objective
Implement an intelligent task classification system that identifies GPU-intensive vs CPU-suitable tasks and routes them to appropriate workers with dynamic load balancing and failover capabilities.

## Current State Analysis

### Existing Task Routing
- Basic content type detection in `orchestrator.py`
- Simple queue routing based on content type (from Task 1)
- No dynamic worker capability assessment
- No load balancing or intelligent routing

### Required Enhancements
- Intelligent task analysis beyond content type
- Dynamic worker selection based on current load
- Failover routing when preferred workers unavailable
- Task complexity estimation for better scheduling

## Implementation Plan

### Step 1: Enhanced Task Classification Models

#### 1.1 Create Advanced Classification Models
**File**: `packages/morag-core/src/morag_core/models/task_classification.py`

```python
"""Advanced task classification models for intelligent routing."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class TaskComplexity(str, Enum):
    LOW = "low"           # < 1 minute expected
    MEDIUM = "medium"     # 1-10 minutes expected
    HIGH = "high"         # 10-60 minutes expected
    VERY_HIGH = "very_high"  # > 1 hour expected

class ResourceRequirement(str, Enum):
    CPU_LIGHT = "cpu_light"      # Basic CPU processing
    CPU_INTENSIVE = "cpu_intensive"  # Heavy CPU processing
    GPU_PREFERRED = "gpu_preferred"  # Benefits from GPU but can use CPU
    GPU_REQUIRED = "gpu_required"    # Requires GPU acceleration
    MEMORY_INTENSIVE = "memory_intensive"  # High memory usage
    DISK_INTENSIVE = "disk_intensive"      # Heavy disk I/O

class TaskCharacteristics(BaseModel):
    """Detailed characteristics of a task for classification."""
    content_type: str
    file_size_bytes: int
    file_extension: str
    estimated_duration_minutes: float
    complexity: TaskComplexity
    resource_requirements: List[ResourceRequirement]
    memory_requirement_gb: float
    disk_space_requirement_gb: float
    network_bandwidth_requirement_mbps: float = 0.0
    
    # Content-specific characteristics
    audio_duration_seconds: Optional[float] = None
    video_duration_seconds: Optional[float] = None
    video_resolution: Optional[str] = None
    document_page_count: Optional[int] = None
    image_dimensions: Optional[tuple] = None
    
    # Processing options that affect requirements
    enable_diarization: bool = False
    enable_topic_segmentation: bool = False
    extract_thumbnails: bool = False
    use_docling: bool = False
    
class WorkerCapabilityScore(BaseModel):
    """Score representing how well a worker can handle a task."""
    worker_id: str
    capability_score: float  # 0.0 to 1.0
    estimated_completion_time: float  # minutes
    resource_availability: float  # 0.0 to 1.0
    current_load: float  # 0.0 to 1.0
    priority_bonus: float = 0.0  # Bonus for preferred worker type
    
class TaskRoutingDecision(BaseModel):
    """Decision about where to route a task."""
    task_id: str
    selected_worker_id: Optional[str]
    selected_queue: str
    routing_reason: str
    alternative_workers: List[str] = Field(default_factory=list)
    estimated_completion_time: float
    confidence_score: float  # 0.0 to 1.0
    fallback_strategy: Optional[str] = None
    
class TaskClassificationResult(BaseModel):
    """Complete classification result for a task."""
    task_id: str
    characteristics: TaskCharacteristics
    routing_decision: TaskRoutingDecision
    classification_metadata: Dict[str, Any] = Field(default_factory=dict)
    classified_at: datetime = Field(default_factory=datetime.utcnow)
```

### Step 2: Intelligent Task Classifier

#### 2.1 Create Advanced Task Classifier
**File**: `packages/morag/src/morag/services/task_classifier.py`

```python
"""Intelligent task classification service."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import structlog
from PIL import Image
import ffmpeg

from morag_core.models.task_classification import (
    TaskCharacteristics, TaskComplexity, ResourceRequirement,
    WorkerCapabilityScore, TaskRoutingDecision, TaskClassificationResult
)
from morag_core.models.worker import WorkerInfo, WorkerCapability
from morag.services.worker_registry import WorkerRegistry

logger = structlog.get_logger(__name__)

class IntelligentTaskClassifier:
    """Advanced task classifier with intelligent routing."""
    
    def __init__(self, worker_registry: WorkerRegistry):
        self.worker_registry = worker_registry
        
        # Classification rules and weights
        self.complexity_rules = {
            'audio': self._classify_audio_complexity,
            'video': self._classify_video_complexity,
            'document': self._classify_document_complexity,
            'image': self._classify_image_complexity,
            'web': self._classify_web_complexity,
            'youtube': self._classify_youtube_complexity
        }
        
        # Resource requirement mappings
        self.resource_mappings = {
            'audio': [ResourceRequirement.GPU_PREFERRED],
            'video': [ResourceRequirement.GPU_REQUIRED, ResourceRequirement.MEMORY_INTENSIVE],
            'document': [ResourceRequirement.CPU_INTENSIVE],
            'image': [ResourceRequirement.GPU_PREFERRED],
            'web': [ResourceRequirement.CPU_LIGHT],
            'youtube': [ResourceRequirement.GPU_PREFERRED, ResourceRequirement.DISK_INTENSIVE]
        }
    
    async def classify_task(self, content_type: str, file_path: str,
                          options: Dict = None) -> TaskClassificationResult:
        """Classify a task and determine optimal routing."""
        try:
            task_id = f"task_{int(asyncio.get_event_loop().time() * 1000)}"
            options = options or {}
            
            # Analyze file characteristics
            characteristics = await self._analyze_file_characteristics(
                content_type, file_path, options
            )
            
            # Get available workers
            available_workers = self.worker_registry.get_available_workers()
            
            # Score workers for this task
            worker_scores = await self._score_workers_for_task(
                characteristics, available_workers
            )
            
            # Make routing decision
            routing_decision = await self._make_routing_decision(
                task_id, characteristics, worker_scores
            )
            
            result = TaskClassificationResult(
                task_id=task_id,
                characteristics=characteristics,
                routing_decision=routing_decision,
                classification_metadata={
                    'worker_scores': [score.model_dump() for score in worker_scores],
                    'available_workers_count': len(available_workers),
                    'classification_version': '1.0'
                }
            )
            
            logger.info("Task classified successfully",
                       task_id=task_id,
                       content_type=content_type,
                       complexity=characteristics.complexity.value,
                       selected_worker=routing_decision.selected_worker_id,
                       queue=routing_decision.selected_queue)
            
            return result
            
        except Exception as e:
            logger.error("Task classification failed",
                        content_type=content_type,
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def _analyze_file_characteristics(self, content_type: str, 
                                          file_path: str, options: Dict) -> TaskCharacteristics:
        """Analyze file to determine task characteristics."""
        file_path = Path(file_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Get complexity classification
        complexity_func = self.complexity_rules.get(content_type, self._classify_default_complexity)
        complexity, duration_estimate = await complexity_func(file_path, file_size, options)
        
        # Get resource requirements
        resource_requirements = self.resource_mappings.get(content_type, [ResourceRequirement.CPU_LIGHT])
        
        # Adjust requirements based on options
        if options.get('enable_diarization'):
            resource_requirements.append(ResourceRequirement.GPU_PREFERRED)
        if options.get('enable_topic_segmentation'):
            resource_requirements.append(ResourceRequirement.CPU_INTENSIVE)
        if options.get('extract_thumbnails'):
            resource_requirements.append(ResourceRequirement.GPU_PREFERRED)
        if options.get('use_docling'):
            resource_requirements.append(ResourceRequirement.MEMORY_INTENSIVE)
        
        # Calculate resource estimates
        memory_requirement = self._estimate_memory_requirement(content_type, file_size, options)
        disk_requirement = self._estimate_disk_requirement(content_type, file_size, options)
        
        # Get content-specific metadata
        content_metadata = await self._get_content_metadata(content_type, file_path)
        
        characteristics = TaskCharacteristics(
            content_type=content_type,
            file_size_bytes=file_size,
            file_extension=file_path.suffix.lower(),
            estimated_duration_minutes=duration_estimate,
            complexity=complexity,
            resource_requirements=list(set(resource_requirements)),  # Remove duplicates
            memory_requirement_gb=memory_requirement,
            disk_space_requirement_gb=disk_requirement,
            enable_diarization=options.get('enable_diarization', False),
            enable_topic_segmentation=options.get('enable_topic_segmentation', False),
            extract_thumbnails=options.get('extract_thumbnails', False),
            use_docling=options.get('use_docling', False),
            **content_metadata
        )
        
        return characteristics
    
    async def _score_workers_for_task(self, characteristics: TaskCharacteristics,
                                    workers: List[WorkerInfo]) -> List[WorkerCapabilityScore]:
        """Score each worker's capability to handle the task."""
        scores = []
        
        for worker in workers:
            score = await self._calculate_worker_score(worker, characteristics)
            scores.append(score)
        
        # Sort by capability score (highest first)
        scores.sort(key=lambda x: x.capability_score, reverse=True)
        
        return scores
    
    async def _calculate_worker_score(self, worker: WorkerInfo,
                                    characteristics: TaskCharacteristics) -> WorkerCapabilityScore:
        """Calculate how well a worker can handle a specific task."""
        base_score = 0.0
        
        # Check capability match
        if ResourceRequirement.GPU_REQUIRED in characteristics.resource_requirements:
            if worker.registration.capability in [WorkerCapability.GPU, WorkerCapability.HYBRID]:
                base_score += 0.4
            else:
                base_score = 0.0  # Cannot handle GPU-required tasks
        elif ResourceRequirement.GPU_PREFERRED in characteristics.resource_requirements:
            if worker.registration.capability == WorkerCapability.GPU:
                base_score += 0.4
            elif worker.registration.capability == WorkerCapability.HYBRID:
                base_score += 0.3
            else:
                base_score += 0.1  # Can handle but not optimal
        else:
            # CPU tasks
            if worker.registration.capability == WorkerCapability.CPU:
                base_score += 0.3
            elif worker.registration.capability == WorkerCapability.HYBRID:
                base_score += 0.2
            else:
                base_score += 0.1  # GPU worker can handle CPU tasks
        
        # Check hardware requirements
        if characteristics.memory_requirement_gb > worker.registration.hardware.memory_gb:
            base_score *= 0.5  # Insufficient memory
        
        # Check current load
        load_factor = 1.0 - (worker.metrics.active_tasks / worker.registration.max_concurrent_tasks)
        resource_availability = load_factor
        
        # Calculate estimated completion time
        base_time = characteristics.estimated_duration_minutes
        
        # Adjust for worker capability
        if worker.registration.capability == WorkerCapability.GPU and ResourceRequirement.GPU_PREFERRED in characteristics.resource_requirements:
            base_time *= 0.2  # GPU acceleration
        elif worker.registration.capability == WorkerCapability.CPU and ResourceRequirement.GPU_PREFERRED in characteristics.resource_requirements:
            base_time *= 2.0  # CPU fallback slower
        
        # Adjust for current load
        estimated_completion_time = base_time * (1.0 + worker.metrics.active_tasks * 0.5)
        
        # Final capability score
        capability_score = base_score * resource_availability
        
        return WorkerCapabilityScore(
            worker_id=worker.registration.worker_id,
            capability_score=capability_score,
            estimated_completion_time=estimated_completion_time,
            resource_availability=resource_availability,
            current_load=worker.metrics.active_tasks / worker.registration.max_concurrent_tasks
        )
    
    async def _make_routing_decision(self, task_id: str, characteristics: TaskCharacteristics,
                                   worker_scores: List[WorkerCapabilityScore]) -> TaskRoutingDecision:
        """Make the final routing decision based on worker scores."""
        
        if not worker_scores:
            # No workers available - use fallback queue
            return TaskRoutingDecision(
                task_id=task_id,
                selected_worker_id=None,
                selected_queue='cpu_standard',
                routing_reason="No workers available - using fallback queue",
                estimated_completion_time=characteristics.estimated_duration_minutes * 2,
                confidence_score=0.1,
                fallback_strategy="queue_fallback"
            )
        
        # Select best worker
        best_worker = worker_scores[0]
        
        if best_worker.capability_score < 0.3:
            # Low confidence - use fallback
            queue = 'gpu_fallback' if ResourceRequirement.GPU_PREFERRED in characteristics.resource_requirements else 'cpu_standard'
            return TaskRoutingDecision(
                task_id=task_id,
                selected_worker_id=None,
                selected_queue=queue,
                routing_reason=f"Low worker capability score: {best_worker.capability_score:.2f}",
                alternative_workers=[score.worker_id for score in worker_scores[:3]],
                estimated_completion_time=characteristics.estimated_duration_minutes * 1.5,
                confidence_score=best_worker.capability_score,
                fallback_strategy="low_confidence_fallback"
            )
        
        # Determine queue based on requirements
        if ResourceRequirement.GPU_REQUIRED in characteristics.resource_requirements:
            queue = 'gpu_priority'
        elif ResourceRequirement.GPU_PREFERRED in characteristics.resource_requirements:
            queue = 'gpu_priority' if best_worker.capability_score > 0.7 else 'gpu_fallback'
        else:
            queue = 'cpu_standard'
        
        return TaskRoutingDecision(
            task_id=task_id,
            selected_worker_id=best_worker.worker_id,
            selected_queue=queue,
            routing_reason=f"Best worker match with score: {best_worker.capability_score:.2f}",
            alternative_workers=[score.worker_id for score in worker_scores[1:4]],
            estimated_completion_time=best_worker.estimated_completion_time,
            confidence_score=best_worker.capability_score
        )
    
    # Content-specific classification methods
    async def _classify_audio_complexity(self, file_path: Path, file_size: int, 
                                       options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify audio processing complexity."""
        try:
            # Try to get audio duration
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            
            # Base processing time (transcription)
            base_time = duration / 60 * 0.1  # ~6 seconds per minute of audio
            
            # Add time for additional processing
            if options.get('enable_diarization'):
                base_time *= 2.0
            if options.get('enable_topic_segmentation'):
                base_time *= 1.5
            
            # Classify complexity
            if duration < 300:  # < 5 minutes
                complexity = TaskComplexity.LOW
            elif duration < 1800:  # < 30 minutes
                complexity = TaskComplexity.MEDIUM
            elif duration < 3600:  # < 1 hour
                complexity = TaskComplexity.HIGH
            else:
                complexity = TaskComplexity.VERY_HIGH
            
            return complexity, base_time
            
        except Exception:
            # Fallback based on file size
            estimated_duration = file_size / (1024 * 1024) * 60  # Rough estimate
            return TaskComplexity.MEDIUM, estimated_duration / 60 * 0.1
    
    async def _classify_video_complexity(self, file_path: Path, file_size: int,
                                       options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify video processing complexity."""
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            
            # Video processing is generally more intensive
            base_time = duration / 60 * 0.5  # ~30 seconds per minute of video
            
            if options.get('extract_thumbnails'):
                base_time *= 1.3
            
            # Classify based on duration and resolution
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            if width * height > 1920 * 1080:  # > 1080p
                base_time *= 2.0
                complexity = TaskComplexity.VERY_HIGH
            elif duration > 3600:  # > 1 hour
                complexity = TaskComplexity.VERY_HIGH
            elif duration > 1800:  # > 30 minutes
                complexity = TaskComplexity.HIGH
            elif duration > 300:  # > 5 minutes
                complexity = TaskComplexity.MEDIUM
            else:
                complexity = TaskComplexity.LOW
            
            return complexity, base_time
            
        except Exception:
            # Fallback based on file size
            estimated_duration = file_size / (1024 * 1024 * 10)  # Rough estimate
            return TaskComplexity.HIGH, estimated_duration
    
    async def _classify_document_complexity(self, file_path: Path, file_size: int,
                                          options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify document processing complexity."""
        # Estimate pages based on file size
        estimated_pages = max(1, file_size / (1024 * 100))  # ~100KB per page
        
        base_time = estimated_pages * 0.1  # ~6 seconds per page
        
        if options.get('use_docling'):
            base_time *= 2.0  # Docling is more thorough but slower
        
        if estimated_pages < 10:
            complexity = TaskComplexity.LOW
        elif estimated_pages < 50:
            complexity = TaskComplexity.MEDIUM
        elif estimated_pages < 200:
            complexity = TaskComplexity.HIGH
        else:
            complexity = TaskComplexity.VERY_HIGH
        
        return complexity, base_time
    
    async def _classify_image_complexity(self, file_path: Path, file_size: int,
                                       options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify image processing complexity."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                pixels = width * height
                
                if pixels > 4000 * 3000:  # > 12MP
                    complexity = TaskComplexity.HIGH
                    base_time = 2.0
                elif pixels > 2000 * 1500:  # > 3MP
                    complexity = TaskComplexity.MEDIUM
                    base_time = 1.0
                else:
                    complexity = TaskComplexity.LOW
                    base_time = 0.5
                
                return complexity, base_time
                
        except Exception:
            return TaskComplexity.LOW, 0.5
    
    async def _classify_web_complexity(self, file_path: Path, file_size: int,
                                     options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify web scraping complexity."""
        # Web scraping is generally fast but can vary
        return TaskComplexity.LOW, 1.0
    
    async def _classify_youtube_complexity(self, file_path: Path, file_size: int,
                                         options: Dict) -> Tuple[TaskComplexity, float]:
        """Classify YouTube processing complexity."""
        # YouTube involves download + processing
        return TaskComplexity.MEDIUM, 5.0
    
    async def _classify_default_complexity(self, file_path: Path, file_size: int,
                                         options: Dict) -> Tuple[TaskComplexity, float]:
        """Default complexity classification."""
        return TaskComplexity.MEDIUM, 2.0
    
    def _estimate_memory_requirement(self, content_type: str, file_size: int, options: Dict) -> float:
        """Estimate memory requirement in GB."""
        base_memory = {
            'audio': 0.5,
            'video': 2.0,
            'document': 1.0,
            'image': 0.5,
            'web': 0.2,
            'youtube': 1.0
        }.get(content_type, 0.5)
        
        # Scale with file size
        size_factor = min(file_size / (1024 * 1024 * 100), 5.0)  # Cap at 5x
        
        return base_memory * (1.0 + size_factor)
    
    def _estimate_disk_requirement(self, content_type: str, file_size: int, options: Dict) -> float:
        """Estimate disk space requirement in GB."""
        # Generally need 2-3x file size for temporary files
        return (file_size * 3) / (1024 * 1024 * 1024)
    
    async def _get_content_metadata(self, content_type: str, file_path: Path) -> Dict:
        """Get content-specific metadata."""
        metadata = {}
        
        try:
            if content_type in ['audio', 'video']:
                probe = ffmpeg.probe(str(file_path))
                duration = float(probe['format']['duration'])
                
                if content_type == 'audio':
                    metadata['audio_duration_seconds'] = duration
                else:
                    metadata['video_duration_seconds'] = duration
                    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                    width = video_stream.get('width', 0)
                    height = video_stream.get('height', 0)
                    metadata['video_resolution'] = f"{width}x{height}"
                    
            elif content_type == 'image':
                with Image.open(file_path) as img:
                    metadata['image_dimensions'] = img.size
                    
        except Exception as e:
            logger.debug("Could not extract content metadata", 
                        content_type=content_type, error=str(e))
        
        return metadata
```

## Testing Requirements

### Unit Tests
1. **Task Classification Tests**
   - Test complexity classification for each content type
   - Test resource requirement mapping
   - Test worker scoring algorithm
   - Test routing decision logic

2. **File Analysis Tests**
   - Test audio/video metadata extraction
   - Test image dimension analysis
   - Test document page estimation
   - Test memory/disk requirement calculation

### Integration Tests
1. **End-to-End Classification Tests**
   - Test complete classification workflow
   - Test worker selection with different scenarios
   - Test fallback routing when no suitable workers
   - Test load balancing across multiple workers

### Test Files to Create
- `tests/test_intelligent_task_classifier.py`
- `tests/test_task_characteristics.py`
- `tests/integration/test_task_routing_decisions.py`

## Dependencies
- **New**: `ffmpeg-python` for media file analysis
- **New**: `Pillow` for image analysis
- **Existing**: Worker registry from Task 2

## Success Criteria
1. Tasks are accurately classified by complexity and requirements
2. Worker selection considers capability, load, and availability
3. Routing decisions include confidence scores and alternatives
4. System handles edge cases (no workers, insufficient resources)
5. Performance is optimized for real-time classification
6. Fallback strategies work when optimal routing unavailable

## Next Steps
After completing this task:
1. Proceed to Task 6: Priority Queue Implementation
2. Test classification with various file types and sizes
3. Validate worker scoring and selection algorithms

---

**Dependencies**: Task 2 (Worker Registration), Task 4 (Worker Communication)
**Estimated Time**: 4-5 days
**Risk Level**: Medium (complex algorithm implementation)
