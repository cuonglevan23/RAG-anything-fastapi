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
            # Step 1: Parsing
            async with self._lock:
                task = self.tasks[task_id]
                task.status = "parsing"
                task.message = "Parsing document structure..."
                task.percentage = 10.0
                task.logs.append("Starting engine: Mineru Parsing...")
            
            # Using RAGAnything internal methods to have more control
            await self.rag._ensure_lightrag_initialized()
            
            # Step 1.1: Parse document
            content_list, content_based_doc_id = await self.rag.parse_document(file_path)
            
            async with self._lock:
                task.percentage = 40.0
                task.logs.append(f"Parsing complete. Found {len(content_list)} content blocks.")
                task.status = "indexing"
                task.message = "Indexing text content into LightRAG..."

            # Step 2: Separate and Index Text
            from raganything.utils import separate_content, insert_text_content
            text_content, multimodal_items = separate_content(content_list)
            
            if text_content.strip():
                file_name = self.rag._get_file_reference(file_path)
                await insert_text_content(
                    self.rag.lightrag,
                    input=text_content,
                    file_paths=file_name,
                    ids=content_based_doc_id,
                )
                async with self._lock:
                    task.percentage = 70.0
                    task.logs.append("Text content successfully indexed.")

            # Step 3: Multimodal Processing
            if multimodal_items:
                async with self._lock:
                    task.message = f"Processing {len(multimodal_items)} multimodal items (images/tables)..."
                    task.logs.append(f"Detected {len(multimodal_items)} multimodal elements.")
                
                # Set context if available
                if hasattr(self.rag, "set_content_source_for_context"):
                    self.rag.set_content_source_for_context(content_list, self.rag.config.content_format)
                
                await self.rag._process_multimodal_content(multimodal_items, self.rag._get_file_reference(file_path), content_based_doc_id)
            else:
                await self.rag._mark_multimodal_processing_complete(content_based_doc_id)

            async with self._lock:
                task = self.tasks[task_id]
                task.status = "completed"
                task.message = "Processing completed successfully."
                task.percentage = 100.0
                task.logs.append("Finalized indexing. System ready.")
                
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            async with self._lock:
                task = self.tasks[task_id]
                task.status = "failed"
                task.message = "Processing failed."
                task.logs.append(f"ERROR: {str(e)}")
                task.error = str(e)

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
