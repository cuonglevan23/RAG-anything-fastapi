# Phần 3 — LightRAG: GraphRAG Engine

> ⏱ **10 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Phần trước tôi đã trình bày vấn đề của Naive RAG — đặc biệt với văn bản pháp lý. Bây giờ tôi sẽ giới thiệu giải pháp cốt lõi: LightRAG."*

---

## LightRAG là gì?

*"LightRAG là một framework RAG được phát triển bởi nhóm nghiên cứu tại HKUST (Đại học Khoa học và Công nghệ Hồng Kông), công bố năm 2024. Ý tưởng cốt lõi là kết hợp Knowledge Graph với Vector Search — giải pháp mà họ gọi là GraphRAG."*

*"Nhưng tại sao cần graph? Hãy để tôi giải thích."*

---

## Tại sao cần Knowledge Graph?

*"Trong văn bản pháp lý, thông tin không phải là những đoạn văn độc lập — mà là một MẠNG LƯỚI quan hệ."*

```
Cấu trúc quan hệ thực tế trong Luật Thư Viện:

  Điều 5 (Chính sách nhà nước)
      │ ← được cụ thể hóa bởi
  Điều 12 (Thư viện chuyên ngành)
      │ ← tham chiếu thêm
  Điều 18 (Điều kiện hoạt động)
      │ ← dẫn chiếu về
  Điều 35 (Hồ sơ đăng ký)
```

*"Naive RAG tìm 'Điều 12' như một đoạn văn đơn lẻ. LightRAG biết 'Điều 5 → Điều 12 → Điều 18 → Điều 35' là một chuỗi quan hệ và traverse toàn bộ chuỗi đó."*

---

## Kiến trúc LightRAG — Hai tầng song song

### Tầng 1: Knowledge Graph (Neo4j-style, lưu bằng GraphML)

```
Khi indexing:
  Chunk text → LLM → Extract entities + relations

  Entities:
    "Điều 12"           → node (type: dieu)
    "Thư viện chuyên ngành" → node (type: khai_niem)
    "Bộ VHTT&DL"        → node (type: to_chuc)

  Relations:
    "Điều 12" ──[định nghĩa]──► "Thư viện chuyên ngành"
    "Điều 12" ──[quản lý bởi]──► "Bộ VHTT&DL"
    "Điều 12" ──[tham chiếu]──► "Điều 5"
```

### Tầng 2: Vector Database (NanoVectorDB)

```
Chunk text → Embedding model → Vector [0.23, -0.45, ...]
                                       ↓
                             Lưu trong Vector DB

Khi query:
  Câu hỏi → Embedding → Tìm chunk nearest neighbor
```

---

## 5 Query Modes — Lựa chọn chiến lược

*"LightRAG cung cấp 5 chế độ query khác nhau. Đây là điểm mạnh lớn so với RAG thông thường."*

### `local` — Cho câu hỏi cụ thể về entity

```
Flow:
  Query → extract keywords → entity_vdb search
  → tìm entity nodes → traverse local subgraph
  → lấy text chunks của nodes → LLM generate

Tốt nhất cho: "Điều 12 quy định gì?", "Thư viện chuyên ngành là gì?"
```

### `global` — Cho câu hỏi tổng quan

```
Flow:
  Query → graph community detection → summarize communities
  → global context → LLM generate

Tốt nhất cho: "Chính sách xã hội hóa thư viện trong toàn bộ luật thế nào?"
```

### `hybrid` — Kết hợp local + global

```
Flow: Chạy song song local + global → merge contexts → LLM

Tốt nhất cho: Hầu hết câu hỏi thực tế
```

### `naive` — Thuần vector search

```
Flow: Query → vector search → top-K chunks → LLM

Nhanh nhất, phù hợp khi câu hỏi đơn giản
```

### `mix` — hybrid + naive + reranker

```
Flow: hybrid + naive → merge → BGE reranker → LLM

Tốt nhất khi đã cài reranker
```

---

## Quá trình Indexing chi tiết

*"Đây là nội dung bên trong của LightRAG khi bạn upload tài liệu — quan trọng để hiểu vì sao indexing mất thời gian."*

```
Document
    ↓
Chunking (400 tokens, overlap 200)
    ↓
Với mỗi chunk: [song song]
  ├── Embedding → Vector DB
  └── LLM call: "Extract entities and relations from this chunk"
        ↓
        Response: 
          ENTITY | Điều 12 | dieu | Quy định về thư viện chuyên ngành...
          ENTITY | Thư viện chuyên ngành | khai_niem | ...
          RELATION | Điều 12 | định nghĩa | Thư viện chuyên ngành | ...
        ↓
        Nếu kết quả không đủ → Gleaning (retry 2 lần) với prompt mạnh hơn
    ↓
Merge entities và relations vào Knowledge Graph
    ↓
Index xong ✅
```

*"Mỗi chunk cần 1 LLM call — 95 chunks = 95 LLM calls. Đây là chi phí và thời gian của indexing."*

---

## Điểm mạnh và Điểm yếu

### ✅ Điểm mạnh

| Điểm mạnh | Mô tả |
|---|---|
| Hiểu quan hệ | Traverse graph, không chỉ similarity |
| 5 query modes | Linh hoạt theo loại câu hỏi |
| Multi-workspace | Cô lập dữ liệu giữa project |
| LLM cache | Không gọi LLM 2 lần cho cùng chunk |
| Open-source | Apache 2.0, có thể tùy biến sâu |

### ❌ Điểm yếu (gốc)

| Điểm yếu | Mô tả | Chúng tôi fix |
|---|---|---|
| Chỉ xử lý text | Không biết ảnh, bảng | ✅ RAG-Anything |
| Default English | System prompt, entity types generic | ✅ Custom domain |
| workspace="" bug | Tất cả project dùng chung namespace | ✅ Pass `workspace=project_id` |
| Chunking generic | Cắt giữa Điều luật | ✅ Legal boundary chunker |
| Không rerank | Sau retrieve không sort lại | ✅ BGE local reranker |
