# Phần 9 — Kết Quả, Bài Học & Roadmap

> ⏱ **5 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Trước khi vào demo, tôi muốn tổng kết lại hành trình của dự án — những gì chúng tôi học được và những gì vẫn còn phải làm."*

---

## Những vấn đề đã giải quyết — Theo thứ tự phát hiện

*"Thực tế phát triển không theo thứ tự đẹp như trong slide. Mỗi vấn đề được phát hiện khi test thực tế."*

| # | Vấn đề | Phát hiện khi nào | Fix | Impact |
|---|---|---|---|---|
| 1 | Workspace sharing bug | Test 2 tài liệu cùng lúc | `workspace=project_id` | 🔴 Critical |
| 2 | Điều cuối tài liệu không trả lời | Query "Điều 50" trong 54 điều | Legal boundary chunker | 🔴 Critical |
| 3 | Entity graph rời rạc | Nhìn vào GraphViz output | Custom entity_types | 🔴 Critical |
| 4 | Answer lan man | RAGAS relevancy = 0.45 | Legal system_prompt | 🟡 High |
| 5 | 400 Invalid JSON | Indexing file scan | `_sanitize()` | 🟡 High |
| 6 | Rerank tốn phí | Tính chi phí Cohere/tháng | BGE local model | 🟢 Medium |

---

## Bài học kỹ thuật quan trọng

### Bài học 1: Đọc source code — không tin documentation

*"Bug workspace isolation không được document ở đâu cả. Chúng tôi tìm ra bằng cách đọc trực tiếp `lightrag.py` dòng 158 — `workspace=""` default. Bài học: với thư viện mã nguồn mở, nếu có vấn đề kỳ lạ, đọc source code là nhanh nhất."*

### Bài học 2: Domain knowledge quyết định chất lượng

*"Kỹ sư AI giỏi chưa đủ. Cần hiểu domain — văn bản pháp lý có cấu trúc PHẦN > CHƯƠNG > ĐIỀU khác hoàn toàn với báo chí hay tiểu thuyết. Chunking strategy, entity types, system prompt — tất cả phải phản ánh cấu trúc domain."*

### Bài học 3: Đo lường trước, tối ưu sau

*"'Cảm giác tốt hơn' không đủ. RAGAS cho số liệu cụ thể: Answer Relevancy 0.45 → 0.78. Chúng tôi biết chính xác cải tiến nào có impact lớn nhất và tập trung vào đó."*

### Bài học 4: Test edge cases sớm

*"Null bytes trong tài liệu scan, điều luật ở trang cuối, multi-workspace — những trường hợp này không xuất hiện trong happy path testing. Phải test với dữ liệu thực ngay từ sớm."*

---

## Kiến trúc đầy đủ — Tổng kết các lớp

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                      │
│              Streamlit UI (3 tabs)                        │
├─────────────────────────────────────────────────────────┤
│                     API LAYER                            │
│              FastAPI (REST endpoints)                     │
├─────────────────────────────────────────────────────────┤
│                   SERVICE LAYER                          │
│  RAGService: workspace management, instance caching      │
├──────────────────────┬──────────────────────────────────┤
│   PARSING LAYER      │         INDEX LAYER               │
│  MinerU + GPT-4o     │  Legal Chunker + LightRAG         │
│  Context window      │  Entity extraction (legal types)  │
│  Sanitize content    │  Knowledge Graph + Vector DB      │
├──────────────────────┴──────────────────────────────────┤
│                   RETRIEVAL LAYER                        │
│  5 query modes + BGE Reranker + configurable top_k       │
├─────────────────────────────────────────────────────────┤
│                 EVALUATION LAYER                         │
│              RAGAS: 4 metrics tự động                     │
└─────────────────────────────────────────────────────────┘
```

---

## Roadmap — Những gì chưa làm được

### Ngắn hạn (1-2 tháng)
- [ ] **Authentication** — JWT login, phân quyền theo workspace
- [ ] **Batch upload** — upload nhiều file cùng lúc
- [ ] **Document management** — xóa, cập nhật tài liệu đã index
- [ ] **Export answers** — xuất lịch sử hội thoại + citations

### Trung hạn (3-6 tháng)
- [ ] **Local LLM** — thay GPT-4o bằng Qwen2.5-72B chạy on-premise
- [ ] **Local embedding** — thay OpenAI embedding bằng BGE-m3 local
- [ ] **Scale storage** — PostgreSQL + Qdrant cho millions documents
- [ ] **Streaming response** — trả lời token-by-token thay vì chờ đến cuối

### Dài hạn (6+ tháng)
- [ ] **Multi-document reasoning** — tra cứu cross-document phức tạp
- [ ] **Fine-tuned reranker** — train reranker đặc thù cho văn bản pháp lý VN
- [ ] **API Gateway** — exposing API cho third-party integration
