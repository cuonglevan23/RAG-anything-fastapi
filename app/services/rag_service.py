import os
import uuid
import asyncio
from typing import Dict, Optional, Any
from pathlib import Path
from loguru import logger

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from app.core.config import settings
from app.models.schema import ProcessingStatus

class RAGService:
    def __init__(self):
        self.rag: Optional[RAGAnything] = None
        self.tasks: Dict[str, ProcessingStatus] = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize RAGAnything singleton"""
        if self.rag:
            return

        logger.info("Initializing RAGAnything service...")
        
        config = RAGAnythingConfig(
            working_dir=settings.RAG_STORAGE_DIR,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                settings.LLM_MODEL, prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=settings.OPENAI_API_KEY, **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=1536, max_token_size=8192,
            func=lambda texts: openai_embed(texts, model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY),
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )
        
        await self.rag._ensure_lightrag_initialized()
        logger.info("RAGAnything service initialized successfully.")

    async def finalize(self):
        """Cleanup on shutdown"""
        if self.rag:
            await self.rag.finalize_storages()
            logger.info("RAGAnything service finalized.")

    async def process_document(self, task_id: str, file_path: str, filename: str):
        """Background task for processing document"""
        try:
            async with self._lock:
                self.tasks[task_id].status = "parsing"
                self.tasks[task_id].message = "Parsing document..."
                self.tasks[task_id].percentage = 20.0

            # Process document
            # Note: process_document_complete is sync in RAGAnything, but we'll call it here
            # Ideally we'd wrap it in run_in_executor if it's very heavy
            await asyncio.to_thread(self.rag.process_document_complete, file_path=file_path)

            async with self._lock:
                self.tasks[task_id].status = "completed"
                self.tasks[task_id].message = "Processing completed successfully."
                self.tasks[task_id].percentage = 100.0
                
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            async with self._lock:
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].message = "Processing failed."
                self.tasks[task_id].error = str(e)

    def create_task(self, filename: str) -> str:
        """Create a new task and return its ID"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = ProcessingStatus(
            task_id=task_id,
            filename=filename,
            status="pending",
            message="Queued for processing",
            percentage=0.0
        )
        return task_id

    def get_task_status(self, task_id: str) -> Optional[ProcessingStatus]:
        """Get status of a specific task"""
        return self.tasks.get(task_id)

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """Execute RAG query"""
        if not self.rag:
            await self.initialize()
        return await self.rag.aquery(query=query, mode=mode)

rag_service = RAGService()
