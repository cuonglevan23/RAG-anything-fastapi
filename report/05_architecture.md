# Phần 5 — Kiến Trúc Hệ Thống Đã Xây Dựng

> ⏱ **8 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Tôi đã trình bày LightRAG và RAG-Anything như là các thư viện nguồn mở. Bây giờ, tôi sẽ trình bày hệ thống thực tế mà chúng tôi đã xây dựng ở tầng ứng dụng — wrap 2 thư viện đó thành một sản phẩm hoàn chỉnh có thể dùng được."*

---

## Sơ đồ kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────────┐
│                       TẦNG GIAO DIỆN                          │
│                    Streamlit (port 8501)                       │
│                                                               │
│  Tab 1: Upload & Parse  | Tab 2: Chat  | Tab 3: RAGAS Eval   │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP REST API
                          │ (JSON payload)
┌─────────────────────────▼────────────────────────────────────┐
│                    TẦNG API                                    │
│                 FastAPI (port 8000)                            │
│                                                               │
│  POST /api/v1/rag/upload       → Nhận file, bắt đầu indexing  │
│  GET  /api/v1/rag/status/{id}  → Poll tiến trình              │
│  GET  /api/v1/rag/projects     → Danh sách workspace          │
│  POST /api/v1/rag/query        → Query, trả về answer          │
│  POST /api/v1/rag/query_eval   → Query + context (cho RAGAS)  │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                   TẦNG SERVICE                                 │
│              RAGService (rag_service.py)                       │
│                                                               │
│  - Quản lý instances (1 RAGAnything per workspace)            │
│  - Workspace isolation (workspace=project_id)                 │
│  - Vietnamese Legal Chunker                                   │
│  - Legal System Prompt injection                              │
│  - Local BGE Reranker                                         │
│  - Content sanitization (null bytes)                          │
└──────────┬──────────────────────────────────────┬────────────┘
           │                                      │
┌──────────▼──────────┐              ┌────────────▼───────────┐
│   RAGAnything        │              │  Storage Per Workspace  │
│                      │              │                         │
│  ├─ LightRAG Core    │              │  rag_storage/           │
│  │   ├─ GraphDB      │              │  └─ {project_id}/       │
│  │   ├─ VectorDB     │              │     ├─ {id}/            │
│  │   └─ KV Store     │              │     │   ├─ graph_*.graphml│
│  │                   │              │     │   ├─ entities_vdb/ │
│  └─ VLM Parser       │              │     │   ├─ text_chunks/  │
│      ├─ MinerU       │              │     │   └─ llm_cache/    │
│      └─ CustomVLM    │              │     └─ (isolate hoàn toàn)│
└─────────────────────┘              └────────────────────────┘
```

---

## Bug Nghiêm Trọng Đã Fix: Multi-Workspace Isolation

*"Đây là bug nguy hiểm nhất chúng tôi gặp trong quá trình phát triển. Nguy hiểm vì nó KHÔNG raise exception — hệ thống chạy bình thường nhưng trả kết quả SAI."*

**Vấn đề:**
```python
# Trong lightrag.py — source code gốc:
@dataclass
class LightRAGConfig:
    workspace: str = field(
        default_factory=lambda: os.getenv("WORKSPACE", "")
        # Nếu WORKSPACE không set → workspace = ""
        # "" được dùng làm key cho TẤT CẢ storage namespace
    )
```

**Hệ quả:**
```
User A upload "LuatThuVien.pdf" vào workspace "project-A"
  → LightRAG lưu với namespace key = ""

User B upload "BaoCaoTaiChinh.pdf" vào workspace "project-B"
  → LightRAG lưu với namespace key = "" (TRÙNG!)

User B query "doanh thu quý 1" trong "project-B"
  → Hệ thống tìm trong namespace "" → trả kết quả của "project-A"!
```

**Fix:**
```python
# rag_service.py — fix của chúng tôi:
lightrag_kwargs = {
    "workspace": project_id,  # ← force đúng namespace
    ...
}
```

*"Bài học: đọc source code, không tin default values của thư viện."*

---

## Luồng Xử Lý Upload — Bất đồng bộ hoàn toàn

*"Một tài liệu 100 trang có thể mất 5-15 phút để index. Nếu xử lý đồng bộ, người dùng phải chờ HTTP request trong 15 phút — không thể chấp nhận được. Chúng tôi giải quyết bằng async background task."*

```
[Client] POST /upload
           ↓
[API] Nhận file → lưu vào uploads/
      → Tạo task_id
      → fire asyncio.create_task(background_processing)
      → Trả về 202 Accepted + {"task_id": "abc-def"}  ← NGAY LẬP TỨC

[Background task - chạy độc lập]:
  1. VLM Parser → markdown
  2. LightRAG.ainsert_custom() → chunking → entity extraction → indexing
  3. Cập nhật status: "processing" → "completed" / "failed"

[Client] Mỗi 2 giây: GET /status/abc-def
  → {"status": "processing", "progress": "45/95 chunks", "message": "..."}
  → {"status": "completed"} ← UI hiển thị xong
```

---

## Query Flow — Tham số có thể tùy chỉnh

*"Người dùng có thể điều chỉnh các tham số query từ UI Streamlit."*

```python
# POST /api/v1/rag/query
{
    "query": "Điều 12 quy định gì?",
    "project_id": "luat-thu-vien",
    "mode": "hybrid",      # local | global | hybrid | naive | mix
    "top_k": 100,          # Số entities/chunks lấy ra
    "response_type": "Structured List"  # Loại output
}
```

---

## Điểm mạnh và Điểm yếu của Kiến trúc

### ✅ Điểm mạnh

| Điểm mạnh | Mô tả |
|---|---|
| Async hoàn toàn | Indexing không block API |
| Workspace isolation | Mỗi project hoàn toàn độc lập |
| Singleton instances | Không khởi tạo lại LightRAG mỗi request |
| Configurable query | UI có thể tuỳ chỉnh mode, top_k |
| Status tracking | Biết tiến trình chi tiết |

### ❌ Điểm yếu

| Điểm yếu | Mô tả | Giải pháp tiếp theo |
|---|---|---|
| Single instance | 1 server, không scale horizontally | Thêm Redis + distributed queue |
| In-memory instances | Server restart = khởi tạo lại tất cả | Persistent session store |
| JSON file storage | Không phù hợp production lớn | Thay bằng PostgreSQL + Qdrant |
| Không auth | Không có user authentication | Thêm JWT layer |
