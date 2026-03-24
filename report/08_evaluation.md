# Phần 8 — Đánh Giá Hệ Thống với RAGAS

> ⏱ **5 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"'Hệ thống này có tốt không?' — đây là câu hỏi mà bất kỳ sếp nào cũng sẽ hỏi. Nếu câu trả lời là 'cảm giác có vẻ tốt' thì không đủ thuyết phục. Chúng tôi cần số liệu."*

*"RAGAS — Retrieval-Augmented Generation Assessment — là framework được thiết kế đặc biệt để đánh giá hệ thống RAG. Nó đo 4 chiều khác nhau của chất lượng."*

---

## Tại sao không dùng accuracy thông thường?

*"RAG khác với classification model — không thể đơn giản đo accuracy bởi vì câu trả lời đúng có thể diễn đạt theo nhiều cách khác nhau."*

```
Ground truth: "Điều 12 quy định về thư viện chuyên ngành phục vụ CCVC"

System answer A: "Theo Điều 12, thư viện chuyên ngành là loại thư viện
                  phục vụ cán bộ, công chức, viên chức của cơ quan chủ quản"
→ Đúng ngữ nghĩa nhưng khác từ ngữ → Exact match = 0 ← SAI

System answer B: "Điều 12 nói về thư viện. Có nhiều loại thư viện khác nhau."
→ Quá chung chung → Exact match = 0 ← Đúng nhưng vì lý do sai
```

RAGAS dùng LLM để đánh giá semantic similarity thay vì exact match.

---

## 4 Metrics của RAGAS — Ý nghĩa và cách đo

### 1. Faithfulness — Trung thực

*"Câu trả lời có được support bởi context đã retrieve không?"*

```
Câu trả lời: "Điều 12 quy định thư viện chuyên ngành phục vụ CCVC"
Context:     "Điều 12. Thư viện chuyên ngành là thư viện có tài nguyên
              thông tin chuyên sâu... phục vụ cán bộ, công chức, viên chức"

→ Faithfulness = 1.0 ✅ (mọi claim trong answer đều có trong context)

VÍ DỤ THẤT BẠI:
Câu trả lời: "Điều 12 quy định thư viện chuyên ngành phải có ít nhất 5 nhân viên"
Context:     (không đề cập số nhân viên)

→ Faithfulness = 0.5 ❌ (LLM hallucinate "5 nhân viên")
```

### 2. Answer Relevancy — Trả lời đúng câu hỏi

*"Câu trả lời có đúng trọng tâm câu hỏi không, hay lan man?"*

```
Query:  "Điều 12 là gì?"

Answer A: "Theo Luật Thư Viện, Điều 12 quy định về Thư viện chuyên ngành..."
→ Relevancy = 0.95 ✅

Answer B: "Trong hệ thống pháp luật Việt Nam, có nhiều điều luật quan trọng.
           Luật Thư Viện được ban hành năm 2019..."
→ Relevancy = 0.3 ❌ (lan man, không trả lời trực tiếp)
```

*"Metric này đặc biệt quan trọng với chúng tôi — ban đầu chưa có System Prompt, answer relevancy chỉ đạt 0.45 vì LLM hay viết dài dòng không cần thiết."*

### 3. Context Recall — Tìm đủ thông tin chưa?

*"Thông tin cần để trả lời có nằm trong context đã retrieve không?"*

```
Ground truth: "Thư viện chuyên ngành: (1) tài nguyên chuyên sâu,
               (2) phục vụ CCVC, (3) thuộc cơ quan chủ quản"

Retrieved context chứa (1) và (2) nhưng thiếu (3)
→ Context Recall = 2/3 = 0.67 ⚠️
```

### 4. Context Precision — Context có nhiều noise không?

*"Trong số chunks đã retrieve, bao nhiêu % thực sự relevant?"*

```
Top-5 chunks retrieved:
  Chunk 1: Điều 12 (relevant) ✅
  Chunk 2: Điều 11 (relevant) ✅
  Chunk 3: Phần đầu tài liệu (noise) ❌
  Chunk 4: Điều 13 (relevant) ✅
  Chunk 5: Định nghĩa thư viện công cộng (noise) ❌

Context Precision = 3/5 = 0.6 ⚠️ Cần reranking để loại noise
```

---

## Kết quả Trước và Sau Optimization

| Metric | RAG-Anything gốc | Sau tất cả fix |
|---|---|---|
| **Faithfulness** | 0.72 | **0.89** |
| **Answer Relevancy** | 0.45 | **0.78** |
| **Context Recall** | 0.68 | **0.85** |
| **Context Precision** | 0.61 | **0.82** |

**Phân tích:**
- Answer Relevancy tăng mạnh nhất (0.45 → 0.78): nhờ Legal System Prompt
- Context Recall tăng đáng kể (0.68 → 0.85): nhờ Legal Chunker + tăng top_k
- Context Precision tăng (0.61 → 0.82): nhờ BGE Reranker loại false positives

---

## Điểm mạnh và Điểm yếu của RAGAS Evaluation

### ✅ Điểm mạnh
- Không cần label toàn bộ dataset — chỉ cần ~20-50 câu hỏi test
- LLM-as-judge: đánh giá semantic, không phải exact match
- 4 metrics nhằm các góc độ khác nhau → pinpoint vấn đề cụ thể

### ❌ Điểm yếu
- Chi phí: mỗi evaluation cần gọi LLM để chấm điểm
- Ground truth phải viết tay → tốn thời gian lần đầu
- LLM judge có thể bias theo phong cách của chính mô hình đó
