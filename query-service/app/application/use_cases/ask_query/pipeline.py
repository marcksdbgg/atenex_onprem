from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import structlog

log = structlog.get_logger()

class PipelineStep(ABC):
    """
    Abstract base class for a step in the RAG pipeline.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the pipeline step.
        
        Args:
            context: A dictionary containing the current state of the pipeline execution.
                     Steps read from and write to this context.
                     
        Returns:
            The updated context dictionary.
        """
        pass

class RAGPipeline:
    """
    Orchestrates the execution of a sequence of PipelineSteps.
    """
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    async def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the pipeline steps in order.
        """
        context = initial_context.copy()
        pipeline_log = log.bind(pipeline_execution_id=context.get("request_id", "unknown"))
        
        for step in self.steps:
            try:
                # pipeline_log.info(f"Starting step: {step.name}")
                context = await step.execute(context)
                # pipeline_log.info(f"Completed step: {step.name}")
            except Exception as e:
                pipeline_log.error(f"Error in step {step.name}: {str(e)}")
                raise e
                
        return context
