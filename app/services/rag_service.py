import os
import uuid
import asyncio
from functools import partial
from typing import Dict, Optional, Any
from pathlib import Path
from loguru import logger

from app.services.vlm_parser import CustomOpenAIPipeline
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.rerank import cohere_rerank
# set_default_workspace: đảm bảo LightRAG dùng đúng namespace cho từng project
from lightrag.kg.shared_storage import set_default_workspace
from app.core.config import settings
from app.models.schema import ProcessingStatus
# test comment for commit
class RAGService:
    def __init__(self):
        self.instances: Dict[str, RAGAnything] = {}
        self.tasks: Dict[str, ProcessingStatus] = {}
        self._lock = asyncio.Lock()

    async def get_instance(self, project_id: str) -> RAGAnything:
        """Get or initialize RAGAnything instance for a project"""
        async with self._lock:
            if project_id in self.instances:
                return self.instances[project_id]

            logger.info(f"Initializing RAGAnything instance for project: {project_id}")
            
            project_dir = os.path.abspath(settings.BASE_RAG_DIR / project_id)
            os.makedirs(project_dir, exist_ok=True)
            
            # ✅ Scope LightRAG shared storage namespaces to THIS project
            # Without this, all projects share the same pipeline_status / namespace
            # causing cross-project data leakage when querying.
            set_default_workspace(project_id)
            logger.info(f"📰 Default workspace set to: {project_id}")
            
            logger.info(f"📁 PROJECT STORAGE PATH: {project_dir}")
            
            config = RAGAnythingConfig(
                working_dir=project_dir,
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

            # Custom extraction prompt: dạy LightRAG nhận diện "Điều X" là entity quan trọng
            # và giữ lại nội dung nguyên văn thay vì tóm tắt.
            LEGAL_ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that may contain Vietnamese legal content (laws, decrees, regulations),
identify all the entities and relationships needed to understand the document structure.

-Instructions-
1. ALWAYS treat each "Điều X" (Article/Clause number) as a PRIMARY ENTITY of type "legal_article".
2. Extract the COMPLETE, VERBATIM content of each Điều as its description—DO NOT summarize or truncate.
3. Identify relationships between Điều (e.g., "Điều 5 references Điều 12").
4. For other entities (organizations, concepts, terms), extract normally.
5. Use Vietnamese names exactly as they appear in the source text.

-Example-
Entity: Điều 12 | Type: legal_article | Description: <full verbatim text of Điều 12>
Entity: Thư viện chuyên ngành | Type: concept | Description: ...
Relationship: Điều 12 -> defines -> Thư viện chuyên ngành
"""

            # ============================================================
            # Cohere Rerank 3.5 Configuration
            # Để bật: set RERANK_ENABLE=true và COHERE_API_KEY trong .env
            # LightRAG khuyến nghị dùng mode="mix" khi bật reranker.
            # ============================================================
            rerank_func = None
            if settings.RERANK_ENABLE and settings.COHERE_API_KEY:
                rerank_func = partial(
                    cohere_rerank,
                    model=settings.RERANK_MODEL,              # rerank-v3.5 | rerank-multilingual-v3.0
                    api_key=settings.COHERE_API_KEY,          # Cohere API Key từ .env
                    base_url=settings.RERANK_BASE_URL,        # https://api.cohere.com/v2/rerank (mặc định)
                    enable_chunking=settings.RERANK_ENABLE_CHUNKING,      # True nếu tài liệu dài > 4096 token
                    max_tokens_per_doc=settings.RERANK_MAX_TOKENS_PER_DOC # Token limit mỗi chunk (mặc định 4096)
                )
                logger.info(f"✅ Cohere Reranker enabled: model={settings.RERANK_MODEL}")
            else:
                logger.info("⚠️  Reranker disabled. Set RERANK_ENABLE=true + COHERE_API_KEY in .env to enable.")
            # ============================================================

            # Xây dựng lightrag_kwargs động để thêm rerank_model_func có điều kiện
            lightrag_kwargs = {
                # ✅ KEY FIX: workspace isolates ALL LightRAG in-memory namespaces per project.
                # Without this, all projects default to workspace="" and share the same
                # text_chunks / entities_vdb / chunks_vdb / pipeline_status namespace,
                # causing cross-project query results even with different working_dirs.
                "workspace": project_id,
                "chunk_token_size": 512,          # Giảm từ 1200 → 600: mỗi Điều luật có chunk riêng
                "chunk_overlap_token_size": 128,   # Overlap nhỏ để giữ ngữ cảnh liên Điều
                "addon_params": {
                    "insert_batch_size": 5,
                    "language": "Vietnamese",
                    "entity_extract_max_gleaning": 2,
                }
            }
            if rerank_func:
                lightrag_kwargs["rerank_model_func"] = rerank_func  # Bật Cohere Reranker

            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                lightrag_kwargs=lightrag_kwargs,
            )
            
            await rag._ensure_lightrag_initialized()
            self.instances[project_id] = rag
            logger.info(f"Project '{project_id}' initialized successfully.")
            return rag

    async def list_projects(self) -> list[str]:
        """List all available project IDs based on storage directories"""
        if not settings.BASE_RAG_DIR.exists():
            return []
        return [d.name for d in settings.BASE_RAG_DIR.iterdir() if d.is_dir()]

    async def finalize(self):
        """Cleanup all instances on shutdown"""
        for project_id, rag in self.instances.items():
            await rag.finalize_storages()
            logger.info(f"Project '{project_id}' finalized.")

    async def process_document(self, task_id: str, file_path: str, filename: str, project_id: str):
        """Background task for processing document in a specific project"""
        try:
            rag = await self.get_instance(project_id)
            
            # Step 1: Parsing
            async with self._lock:
                task = self.tasks[task_id]
                task.status = "parsing"
                task.message = "Parsing document structure..."
                task.percentage = 10.0
                task.logs.append(f"Starting engine: Mineru Parsing for project {project_id}...")
            # Step 1.1: Parse document using VLM (GPT-4o)
            task.logs.append(f"Starting engine: OpenAIPipeline (VLM) for project {project_id}...")
            
            # Khởi tạo pipeline và process
            vlm_pipeline = CustomOpenAIPipeline(api_key=settings.OPENAI_API_KEY)
            
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            project_dir = os.path.abspath(settings.BASE_RAG_DIR / project_id)
            output_dir = project_dir # Save VLM outputs into the project dir
            
            parsed_md_path = os.path.join(output_dir, file_basename, "vlm", f"{file_basename}.md")
            
            if os.path.exists(parsed_md_path):
                task.logs.append(f"Found existing Markdown at {parsed_md_path}. Skipping VLM Parsing.")
            else:
                task.logs.append("No Markdown found, processing PDF via VLM to generate markdown...")
                # Chạy process_pdf trong thread pool để không block FastAPI event loop (tránh timeout)
                import asyncio
                parsed_md_path = await asyncio.to_thread(
                    vlm_pipeline.process_pdf, file_path, output_dir, file_basename
                )
                task.logs.append(f"VLM Parsing finished. Markdown created at: {parsed_md_path}")

            async with self._lock:
                task.percentage = 40.0
                task.status = "indexing"
                task.message = "Indexing markdown content into LightRAG..."

            # Step 2: Index Text Content Directly
            if os.path.exists(parsed_md_path):
                with open(parsed_md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()

                from lightrag.utils import compute_mdhash_id
                doc_id = compute_mdhash_id(md_text, prefix="doc-")
                file_name = os.path.basename(file_path)

                await rag.lightrag.ainsert(
                    input=md_text,
                    file_paths=file_name,
                    ids=doc_id
                )
                
                async with self._lock:
                    task.percentage = 70.0
                    task.logs.append("Markdown content successfully indexed into Knowledge Base.")
            else:
                raise FileNotFoundError(f"Markdown file not found after processing: {parsed_md_path}")
            
            # Note: Multimodal blocks (Tables/Equations) are already converted to text 
            # (HTML/LaTeX) within the markdown by VLM, so we don't need the legacy 
            # separate multimodal extraction step from Mineru outputs here.
            
            async with self._lock:
                task = self.tasks[task_id]
                task.status = "completed"
                task.message = "Processing completed successfully."
                task.percentage = 100.0
                task.logs.append("Finalized indexing. System ready.")
                
        except Exception as e:
            logger.error(f"Error processing document {filename} in {project_id}: {e}")
            async with self._lock:
                task = self.tasks[task_id]
                task.status = "failed"
                task.message = "Processing failed."
                task.logs.append(f"ERROR: {str(e)}")
                task.error = str(e)

    def create_task(self, filename: str, project_id: str) -> str:
        """Create a new task and return its ID"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = ProcessingStatus(
            task_id=task_id,
            filename=filename,
            project_id=project_id,
            status="pending",
            message="Queued for processing",
            percentage=0.0
        )
        return task_id

    def get_task_status(self, task_id: str) -> Optional[ProcessingStatus]:
        """Get status of a specific task"""
        return self.tasks.get(task_id)

    async def query(self, project_id: str, query: str, mode: str = "hybrid") -> str:
        """Execute RAG query within a specific project"""
        # Switch default workspace TRƯỚC khi query để tránh đọc nhầm project khác
        set_default_workspace(project_id)
        rag = await self.get_instance(project_id)
        
        # Khi reranker được bật, LightRAG khuyến nghị dùng mode='mix' để tối ưu hiệu suất
        # Người dùng có thể ghi đè bằng cách chọn mode khác trên Streamlit
        effective_mode = mode
        if settings.RERANK_ENABLE and mode == "hybrid":
            effective_mode = "mix"  # mix = hybrid + naive, tốt nhất khi có reranker
            logger.info(f"Reranker active: auto-upgrade mode hybrid → mix")

        # Mode 'local' và 'naive' phù hợp truy vấn nguyên văn (chunk-based)
        # Mode 'hybrid'/'mix' phù hợp câu hỏi tổng hợp (graph-based)
        return await rag.aquery(
            query=query, 
            mode=effective_mode, 
            top_k=100, 
            response_type="Structured List"
        )

    async def query_with_context(self, project_id: str, query: str, mode: str = "hybrid") -> dict:
        """
        Execute RAG query and return BOTH the final answer and the retrieved raw contexts.
        Used by the RAGAS evaluation tab.
        """
        # Switch default workspace TRƯỚC khi query để tránh đọc nhầm project khác
        set_default_workspace(project_id)
        rag = await self.get_instance(project_id)

        effective_mode = mode
        if settings.RERANK_ENABLE and mode == "hybrid":
            effective_mode = "mix"

        # Run the main answer query
        answer = await rag.aquery(
            query=query,
            mode=effective_mode,
            top_k=100,
            response_type="Structured List"
        )

        # Run naive query to get raw retrieved passages (contexts for RAGAS)
        # Naive mode bypasses Knowledge Graph and returns raw chunks directly
        try:
            raw_context = await rag.aquery(
                query=query,
                mode="naive",
                top_k=20,               # Fewer chunks, focus on most relevant
                response_type="Multiple Paragraphs"
            )
            # Split into individual passages for RAGAS (expects List[str])
            contexts = [p.strip() for p in raw_context.split("\n\n") if p.strip()]
            if not contexts:
                contexts = [raw_context]  # fallback: treat whole response as one context
        except Exception as e:
            logger.warning(f"Naive context retrieval failed: {e}. Using answer as context fallback.")
            contexts = [answer]

        return {
            "query":    query,
            "answer":   answer,
            "contexts": contexts,
        }

rag_service = RAGService()
