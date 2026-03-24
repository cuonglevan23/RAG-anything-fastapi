# Phần 10 — Demo Live & Q&A

> ⏱ **~16 phút** (3 phút script + 13 phút Q&A)

---

## 🎤 Lời dẫn vào Demo

*"Lý thuyết đủ rồi. Bây giờ tôi sẽ cho các bạn thấy hệ thống hoạt động thực tế. Tôi sẽ demo 3 tình huống: upload tài liệu, query với các mode khác nhau, và chạy RAGAS đánh giá tự động."*

---

## Checklist chuẩn bị (trước khi lên sân khấu)

- [ ] FastAPI đang chạy: `http://localhost:8000`
- [ ] Streamlit đang chạy: `http://localhost:8501`
- [ ] `LuatThuVien.pdf` sẵn trong thư mục Desktop
- [ ] Workspace "luat-demo" CHƯA tồn tại (để demo upload từ đầu)
- [ ] Bộ câu hỏi RAGAS `eval_questions.csv` sẵn sàng
- [ ] Màn hình: chia đôi slide (trái) + Streamlit (phải)

---

## Script Demo Tình Huống 1: Upload & Indexing (60 giây)

> *"Đây là Tab 1 — nơi người dùng upload tài liệu. Tôi tạo workspace mới tên 'luat-demo' và upload file Luật Thư Viện."*

```
[Thao tác]:
  Tab 1 → Nhập workspace: "luat-demo"
  → Nhấn "Create"
  → Upload: LuatThuVien.pdf
  → Nhấn "Process"

[Giải thích trong khi progress bar chạy]:
"Hệ thống đang thực hiện 2 công việc song song:
 Đầu tiên: VLM Parser phân tích từng trang — nhận diện text, bảng, ảnh
 Sau đó: LightRAG indexing — chunking theo ranh giới Điều luật,
          LLM extract entities, xây dựng Knowledge Graph

Với văn bản 100 trang, mất khoảng 8-12 phút. Tôi đã pre-index trước
để tiết kiệm thời gian demo."
```

---

## Script Demo Tình Huống 2: Query Comparison (90 giây)

*"Tôi sẽ chạy cùng câu hỏi với 2 mode khác nhau để thấy sự khác biệt rõ rệt."*

### Query 1 — Tra cứu cụ thể (mode: local)

```
[Thao tác]:
  Tab 2 → Chọn workspace "luat-demo"
  ⚙️ Query Settings:
    Mode: local
    Top-K: 100
    Response Type: Structured List

Query: "Điều 12 quy định về điều gì? Trích nguyên văn."

[Giải thích kết quả]:
"Mode 'local' tìm entity 'Điều 12' trực tiếp trong Knowledge Graph,
 lấy chunks được gắn với node đó, đưa vào LLM với yêu cầu trích nguyên văn.
 Kết quả: chính xác, có trích dẫn nguồn."
```

### Query 2 — Tổng quan (mode: hybrid)

```
[Thao tác]:
  Mode: hybrid
  Top-K: 200
  Response Type: Multiple Paragraphs

Query: "Luật Thư Viện quy định về xã hội hóa như thế nào?
        Liệt kê tất cả điều khoản liên quan."

[Giải thích kết quả]:
"Mode 'hybrid' chạy song song local search (tìm entity 'xã hội hóa')
 và global search (traverse cả community graph về chủ đề này).
 Kết quả: tổng hợp từ nhiều Điều — 6, 7, 14, 20... — mà local search đơn lẻ có thể bỏ sót."
```

---

## Script Demo Tình Huống 3: RAGAS Evaluation (30 giây)

```
[Thao tác]:
  Tab 3 → Upload eval_questions.csv (20 câu hỏi + ground truth)
  → Nhấn "Fetch RAG Answers" (tự động query 20 câu)
  → Nhấn "Run RAGAS Evaluation"
  → Xem kết quả 4 metrics

[Giải thích]:
"RAGAS dùng LLM để chấm điểm từng câu trả lời theo 4 tiêu chí.
 Chúng tôi dùng kết quả này để track chất lượng sau mỗi lần thay đổi cấu hình."
```

---

## Q&A — Câu hỏi thường gặp & Gợi ý trả lời

### "Chi phí vận hành hàng tháng là bao nhiêu?"

*"Phụ thuộc số lượng tài liệu và queries. Ước tính với 100 tài liệu/tháng, 500 queries/ngày:"*

| Thành phần | Chi phí |
|---|---|
| GPT-4o indexing | ~$3-5/tài liệu 100 trang |
| Embedding (text-embedding-3-small) | ~$0.02/tài liệu |
| GPT-4o query | ~$0.02-0.05/query |
| **BGE Reranker** | **$0** (local) |
| Server | Chi phí hạ tầng (on-premise) |

### "Tại sao không dùng model local hoàn toàn để giảm chi phí?"

*"Hoàn toàn có thể — chúng tôi thiết kế `llm_model_func` là interface, có thể swap bất kỳ LLM nào vào. Để thay GPT-4o bằng Qwen2.5-72B local chạy qua Ollama, chỉ cần thay hàm này. Nhưng cần GPU mạnh và model đủ lớn để extract entity tốt — khuyến nghị ≥ 70B params. Đây là roadmap của chúng tôi."*

### "Hệ thống có thể xử lý bao nhiêu tài liệu?"

*"Hiện tại giới hạn bởi disk và thời gian indexing. Test thực tế: 50 tài liệu PDF (100-200 trang mỗi file) hoạt động tốt. Scale lớn hơn cần thay storage backend: NanoVectorDB → Qdrant, JsonKV → PostgreSQL."*

### "Dữ liệu nhạy cảm có an toàn không?"

*"Với cấu hình hiện tại, text được gửi lên OpenAI API để embedding và entity extraction. Nếu yêu cầu on-premise hoàn toàn, có thể thay bằng: LLM local (Ollama/vLLM) + embedding local (BGE-m3) + VLM local (MiniCPM-V). Khi đó không có byte nào ra ngoài hệ thống."*

### "Sếp hỏi: cái này hơn gì ChatGPT Enterprise?"

*"ChatGPT Enterprise có thể upload document và query — đúng. Nhưng so sánh:"*

| | ChatGPT Enterprise | Hệ thống này |
|---|---|---|
| Knowledge Graph | ❌ Không | ✅ Traverse quan hệ điều khoản |
| On-premise | ❌ Phải ra cloud | ✅ Có thể 100% nội bộ |
| Custom domain | ❌ Prompt engineering hạn chế | ✅ Sâu từ chunking đến entity |
| Multi-workspace | ❌ Không isolate | ✅ Hoàn toàn cô lập |
| RAGAS eval | ❌ Không có | ✅ 4 metrics đo được |

---

## Lời kết

> *"Hệ thống RAG-Anything + LightRAG mà team xây dựng không phải chỉ là wrapper của một thư viện có sẵn. Chúng tôi đã phát hiện và giải quyết 6 vấn đề kỹ thuật nghiêm trọng, custom sâu cho domain pháp lý Việt Nam, và xây dựng hệ thống evaluation đo lường được chất lượng."*

> *"Điểm mạnh lớn nhất: Knowledge Graph giúp hiểu quan hệ giữa các điều khoản — thứ mà Vector Search đơn thuần không làm được. Điểm yếu lớn nhất hiện tại: chi phí indexing và phụ thuộc OpenAI API — roadmap sẽ giải quyết bằng local LLM."*

> *"Cảm ơn mọi người đã lắng nghe. Tôi sẵn sàng trả lời câu hỏi."*
