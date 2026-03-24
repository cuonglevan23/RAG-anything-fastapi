# Phần 2 — RAG Là Gì? Kiến Trúc và Hạn Chế

> ⏱ **8 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Trước khi đi vào hệ thống của chúng tôi, tôi cần giải thích RAG — Retrieval-Augmented Generation — vì đây là nền tảng của toàn bộ dự án."*

---

## RAG — Ý tưởng cốt lõi

*"Ý tưởng của RAG rất đơn giản và rất khéo léo. Thay vì hỏi một người ghi nhớ tất cả mọi thứ — vốn hay nhớ sai — RAG cho người đó MỞ SÁCH tra cứu trước khi trả lời."*

```
KHÔNG CÓ RAG:
  Câu hỏi → LLM (kiến thức hữu hạn, có thể bịa) → Câu trả lời

CÓ RAG:
  Câu hỏi → [Tìm kiếm tài liệu] → Lấy đoạn liên quan
                                          ↓
                             LLM (câu hỏi + context) → Câu trả lời có căn cứ
```

---

## 3 Giai đoạn của RAG

### Giai đoạn 1: INDEXING (làm một lần)

*"Đầu tiên, ta phải 'dạy' hệ thống biết tài liệu. Quá trình này gọi là indexing."*

```
Tài liệu gốc (PDF, DOCX...)
        ↓
   Chunking — chia nhỏ thành các đoạn
        ↓
   Embedding — chuyển text thành vector số học
   "Điều 12 quy định..." → [0.23, -0.45, 0.78, ...]
        ↓
   Lưu vào Vector Database
```

### Giai đoạn 2: RETRIEVAL (mỗi lần query)

*"Khi người dùng hỏi, hệ thống tìm những đoạn tài liệu liên quan nhất."*

```
"Điều 12 là gì?"
        ↓
   Embedding query → [0.21, -0.43, 0.76, ...]
        ↓
   Tính cosine similarity với toàn bộ chunks trong DB
        ↓
   Lấy top-K chunk có similarity cao nhất
```

### Giai đoạn 3: GENERATION (mỗi lần query)

```
Prompt = "Dựa trên context sau, trả lời câu hỏi:
          Context: [chunk 1] [chunk 2] [chunk 3]
          Câu hỏi: Điều 12 là gì?"
        ↓
   LLM tổng hợp → Câu trả lời có trích dẫn
```

---

## Ưu điểm của RAG so với LLM thuần túy

| Tiêu chí | LLM thuần túy | RAG |
|---|---|---|
| Nguồn thông tin | Kiến thức lúc train | Tài liệu thực tế của bạn |
| Hallucination | **Cao** | Thấp (bounded by context) |
| Cập nhật thông tin | Phải retrain ($$$) | Upload tài liệu mới |
| Trích dẫn nguồn | ❌ Không có | ✅ Biết từ file nào |
| Bảo mật dữ liệu | Ra cloud | Có thể on-premise |

---

## Hạn chế của RAG thông thường (Naive RAG)

*"Tuy nhiên, RAG thông thường vẫn có những hạn chế nghiêm trọng — đặc biệt với tài liệu pháp lý. Đây là điều quan trọng nhất tôi muốn các bạn hiểu trước khi tôi giới thiệu LightRAG."*

### Hạn chế 1 — Không hiểu quan hệ giữa các điều khoản

```
Câu hỏi: "Để hưởng chế độ X, cần đáp ứng điều kiện gì?"

Thông tin nằm rải rác:
  Điều 5:  "Đối tượng áp dụng chế độ X là..."
  Điều 12: "Điều kiện hưởng chế độ X, xem Điều 5 và Điều 18"
  Điều 18: "Điều kiện bổ sung: ..."

→ Naive RAG chỉ tìm chunk có "chế độ X" gần nhất
→ Bỏ qua chuỗi tham chiếu Điều 12 → Điều 5 + Điều 18
→ Câu trả lời THIẾU và SAI
```

### Hạn chế 2 — Chunk bị cắt giữa chừng

```
Token-based chunking (max 512 tokens):
  Chunk 7: "...khoản 3 Điều 11 quy định: thư viện phải có diện tích tối thiểu
            là 50m². Điều 12. Thư viện chuy"  ← BỊ CẮT
  Chunk 8: "ên ngành là thư viện có tài nguyên..."
  
→ Query "Điều 12": mỗi chunk chỉ có nửa nội dung → trả lời không đủ
```

### Hạn chế 3 — Entity extraction không có ngữ cảnh domain

```
Với default settings, LLM extract entity như:
  "Bộ Văn hóa" → organization ✅
  "Điều 12" → ???  ← KHÔNG ĐƯỢC NHẬN DIỆN là entity quan trọng
  "khoản 2 điểm a" → ???  ← BỊ BỎ QUA

→ Knowledge Graph thiếu các node quan trọng nhất của văn bản pháp lý
```

### Hạn chế 4 — Không xử lý được multimodal

```
PDF thực tế gồm:
  - Text ← Naive RAG xử lý được ✅
  - Bảng số liệu ← Naive RAG bỏ qua ❌
  - Sơ đồ tổ chức ← Naive RAG bỏ qua ❌
  - Công thức toán ← Naive RAG bỏ qua ❌
  - Tài liệu scan ← Naive RAG bỏ qua ❌
```

---

## Giải pháp

*"Mỗi hạn chế trên sẽ được giải quyết bởi một phần trong hệ thống của chúng tôi:"*

| Hạn chế | Giải pháp |
|---|---|
| Không hiểu quan hệ | **LightRAG GraphRAG** (Phần 3) |
| Chunk bị cắt | **Vietnamese Legal Chunker** (Phần 6) |
| Entity extraction sai | **Custom entity_types + system_prompt** (Phần 6) |
| Không xử lý multimodal | **RAG-Anything + VLM** (Phần 4) |
