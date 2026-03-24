# Phần 7 — Reranking: Tại Sao & Cách Triển Khai Local

> ⏱ **5 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Sau khi LightRAG retrieve được top-K kết quả, chúng tôi thêm một bước nữa trước khi đưa vào LLM: Reranking. Tôi sẽ giải thích tại sao bước này quan trọng và tại sao chúng tôi chọn giải pháp local miễn phí thay vì Cohere."*

---

## Tại sao Vector Search chưa đủ?

*"Vector search dùng cosine similarity giữa embedding của query và embedding của chunk. Đây là phép đo GIÁN TIẾP về relevance."*

```
Query: "Thủ tục đăng ký thành lập thư viện tư nhân"

Embedding similarity top-5:
  Rank 1 (0.87): Chunk về "đăng ký thư viện tư nhân, điều 35" ← ĐÚNG
  Rank 2 (0.84): Chunk về "thành lập tổ chức văn hóa" ← SAI (false positive)
  Rank 3 (0.82): Chunk về "điều kiện cơ sở vật chất" ← có thể đúng
  Rank 4 (0.79): Chunk về "hồ sơ đăng ký" ← ĐÚNG
  Rank 5 (0.77): Chunk định nghĩa "thư viện tư nhân" ← ít liên quan
```

*"Vấn đề: Rank 2 rank cao hơn Rank 4 dù Rank 4 liên quan hơn. Embedding biểu diễn ngữ nghĩa TỔNG QUÁT, không đặc thù cho từng cặp (query, document)."*

---

## Cross-Encoder Reranking — Nguyên lý

*"Reranker giải quyết vấn đề này bằng cách xử lý CÙNG LÚC cả query và document."*

```
Bi-Encoder (embedding thông thường):
  Query → Encode → vector_q
  Doc   → Encode → vector_d
  Score = cosine(vector_q, vector_d)
  → Nhanh (O(1) per doc) nhưng kém chính xác

Cross-Encoder (reranker):
  (Query, Doc) → Mô hình transformer → Score trực tiếp
  → Chậm hơn O(n) per doc nhưng CHÍNH XÁC hơn nhiều
```

```
Sau reranking:
  Rank 1 (0.94): "đăng ký thư viện tư nhân, điều 35" ← ĐÃ LÊN ĐẦU
  Rank 2 (0.92): "hồ sơ đăng ký" ← ĐÚNG VỊ TRÍ
  Rank 3 (0.71): "điều kiện cơ sở vật chất" ← OK
  Rank 4 (0.32): "thành lập tổ chức văn hóa" ← ĐÃ HẠ XUỐNG (false positive lọc ra)
  [Rank 5 bị loại bỏ — score quá thấp]
```

---

## Lý do chọn BGE-Reranker-v2-m3

*"Ban đầu chúng tôi tích hợp Cohere Reranker — mạnh, multilingual, dễ dùng. Nhưng sau khi tính chi phí, không khả thi cho production."*

**Vấn đề của Cohere:**
```
Chi phí: $1 per 1 triệu tokens
Với 1000 query/ngày × 5 chunks × 200 tokens/chunk:
  = 1,000 × 5 × 200 = 1,000,000 tokens/ngày
  = $1/ngày = $30/tháng = $360/năm
  (Chưa tính chi phí indexing và LLM calls)
```

**Giải pháp: BGE-Reranker-v2-m3 (BAAI)**

| Tiêu chí | Cohere rerank-v3.5 | BGE-v2-m3 |
|---|---|---|
| Chi phí | $1/1M tokens | **$0** |
| VRAM | Cloud | **~560MB** |
| Tiếng Việt | ✅ Tốt | ✅ Tốt (multilingual) |
| Latency | ~200-500ms (network) | ~50-100ms (local) |
| Bảo mật | Data lên cloud | **On-premise** |
| License | Proprietary | **Apache 2.0** |

---

## Thiết kế Implementation

*"Chúng tôi thiết kế LocalReranker theo pattern Singleton — load model 1 lần khi server khởi động, reuse cho tất cả requests."*

```python
class LocalReranker:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance._load_model()  # Load 1 lần duy nhất
        return cls._instance

    async def rerank(self, query, documents, top_n=None):
        pairs = [[query, doc] for doc in documents]

        # Chạy trong thread pool → không block asyncio event loop
        scores = await loop.run_in_executor(None, self._compute_scores, pairs)

        return sorted([
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ], key=lambda x: -x["relevance_score"])
```

**Tại sao thread pool?** GPU inference là blocking operation. Nếu chạy trực tiếp trong asyncio event loop → block toàn bộ server khi đang rerank.

---

## Điểm mạnh và Điểm yếu

### ✅ Điểm mạnh

| | Mô tả |
|---|---|
| Miễn phí hoàn toàn | Không tốn API cost |
| On-premise | Dữ liệu không ra ngoài |
| Singleton | Load model 1 lần, nhanh sau đó |
| Thread-safe | asyncio lock + executor |
| Drop-in replacement | Cùng interface với Cohere |

### ❌ Điểm yếu

| | Mô tả | Giải pháp |
|---|---|---|
| Lần đầu chậm | Download model ~570MB | Pre-download khi deploy |
| VRAM thêm ~560MB | Trong budget 16GB | Giải phóng VRAM sau parse |
| Không bằng Cohere | Yếu hơn ~5-10% trên BEIR benchmark | Tốt đủ dùng với văn bản pháp lý |
