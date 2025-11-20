from abc import ABC, abstractmethod
from typing import Any, Dict, List
import structlog
import time

log = structlog.get_logger()

class PipelineStep(ABC):
    """Abstract base class for a step in the RAG pipeline."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the pipeline step. Reads/writes to context."""
        pass

class RAGPipeline:
    """Orchestrates the execution of a sequence of PipelineSteps."""
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    async def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        req_id = context.get("request_id", "unknown")
        pipeline_log = log.bind(pipeline_exec_id=req_id)
        
        total_start = time.perf_counter()
        
        for step in self.steps:
            step_start = time.perf_counter()
            try:
                # pipeline_log.debug(f"Starting step: {step.name}")
                context = await step.execute(context)
                duration = (time.perf_counter() - step_start) * 1000
                # pipeline_log.debug(f"Completed step: {step.name}", duration_ms=duration)
            except Exception as e:
                pipeline_log.error(f"Error in step {step.name}: {str(e)}")
                raise e
        
        total_duration = (time.perf_counter() - total_start) * 1000
        pipeline_log.info("Pipeline execution completed", total_duration_ms=total_duration)
        return context