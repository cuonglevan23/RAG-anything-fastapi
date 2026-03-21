"""
Local Reranker — BAAI/bge-reranker-v2-m3
==========================================
Thay thế Cohere (mất phí) bằng model local miễn phí.

Model được chọn: BAAI/bge-reranker-v2-m3
- VRAM: ~560MB (fp16) — nhẹ, phù hợp hệ thống 16GB
- Đa ngôn ngữ: hỗ trợ tiếng Việt, Anh, Trung, ...
- Accuracy: tốt ngang Cohere rerank-v2 trên BEIR benchmark
- Giấy phép: Apache 2.0 — hoàn toàn miễn phí
- Tích hợp: FlagEmbedding hoặc sentence-transformers

Cài đặt:
    pip install FlagEmbedding

Hoặc dùng sentence-transformers (fallback):
    pip install sentence-transformers
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger


class LocalReranker:
    """
    Singleton wrapper cho BGE Reranker — load model 1 lần, reuse mãi.
    Thread-safe nhờ asyncio lock.
    """

    _instance: Optional["LocalReranker"] = None
    _lock = asyncio.Lock()

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._reranker = None
        self._backend = None  # "flag" hoặc "sentence_transformers"

    @classmethod
    async def get_instance(cls, model_name: str = "BAAI/bge-reranker-v2-m3") -> "LocalReranker":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_name=model_name)
                await cls._instance._load_model()
        return cls._instance

    async def _load_model(self):
        """Load model trong thread pool để không block event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        """Thử FlagEmbedding trước, fallback sang sentence-transformers."""
        # Thử FlagEmbedding (nhanh hơn, native BGE)
        try:
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            self._backend = "flag"
            logger.info(f"✅ Local Reranker loaded via FlagEmbedding: {self.model_name}")
            return
        except ImportError:
            logger.warning("FlagEmbedding not found, trying sentence-transformers...")
        except Exception as e:
            logger.warning(f"FlagEmbedding load failed: {e}, trying sentence-transformers...")

        # Fallback: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                self.model_name,
                max_length=512,
            )
            # fp16 nếu GPU available
            if self.use_fp16:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._reranker.model.half()
                except Exception:
                    pass
            self._backend = "sentence_transformers"
            logger.info(f"✅ Local Reranker loaded via sentence-transformers: {self.model_name}")
            return
        except ImportError:
            raise ImportError(
                "Cần cài ít nhất một trong hai:\n"
                "  pip install FlagEmbedding\n"
                "  pip install sentence-transformers"
            )

    def _compute_scores_sync(self, pairs: List[List[str]]) -> List[float]:
        """Tính scores đồng bộ — chạy trong thread pool."""
        if self._backend == "flag":
            scores = self._reranker.compute_score(pairs, normalize=True)
            # compute_score trả về float nếu 1 cặp, list nếu nhiều cặp
            if isinstance(scores, float):
                return [scores]
            return list(scores)
        elif self._backend == "sentence_transformers":
            scores = self._reranker.predict(pairs)
            import numpy as np
            # Áp dụng sigmoid để normalize về [0, 1]
            scores = 1 / (1 + np.exp(-scores))
            return scores.tolist()
        else:
            raise RuntimeError("Model chưa được load")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents và trả về format tương thích LightRAG.

        Returns:
            List[{"index": int, "relevance_score": float}] — sorted desc by score
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        # Chạy inference trong thread pool để không block asyncio event loop
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._compute_scores_sync, pairs)

        # Format theo LightRAG interface
        results = [
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ]

        # Sort desc
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results


# ============================================================
# Hàm public tương thích với LightRAG rerank_model_func interface
# Signature: async (query, documents, top_n=None, **kwargs) -> List[Dict]
# ============================================================

async def local_bge_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Drop-in replacement cho cohere_rerank — dùng BGE local model.

    Cách dùng trong rag_service.py:
        from app.services.local_reranker import local_bge_rerank
        from functools import partial

        rerank_func = partial(local_bge_rerank, model_name="BAAI/bge-reranker-v2-m3")
        lightrag_kwargs["rerank_model_func"] = rerank_func
    """
    reranker = await LocalReranker.get_instance(model_name=model_name)
    return await reranker.rerank(query=query, documents=documents, top_n=top_n)
