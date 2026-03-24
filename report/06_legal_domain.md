# Phần 6 — Custom Domain: 5 Vấn Đề RAG-Anything Gốc Chưa Xử Lý

> ⏱ **8 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Đây là phần quan trọng nhất của bài thuyết trình — những gì chúng tôi đã tự làm, không phải những gì thư viện gốc cung cấp."*

*"Khi chúng tôi chạy RAG-Anything gốc với văn bản luật tiếng Việt, kết quả rất tệ. Tôi sẽ trình bày từng vấn đề cụ thể và cách chúng tôi giải quyết."*

---

## Vấn đề 1 — `system_prompt=None`: LLM không biết đang làm gì

### Bằng chứng vấn đề

*"RAG-Anything gốc gọi LLM như thế này:"*

```python
# Trong RAG-Anything gốc:
def llm_model_func(prompt, system_prompt=None, ...):
    return openai_complete(model, prompt, system_prompt=system_prompt, ...)
    # system_prompt=None → LLM chỉ nhận được prompt kỹ thuật của LightRAG
    # Không có hướng dẫn nào về ngữ cảnh pháp lý Việt Nam
```

**Hệ quả cụ thể:**
```
Prompt LightRAG gửi cho LLM để extract entity:
  "Extract entities from: 'Điều 12. Thư viện chuyên ngành là thư viện 
  có tài nguyên thông tin chuyên sâu...'"

LLM không có system prompt → Extract generic:
  Entity: "thư viện" → type: organization
  Entity: "tài nguyên thông tin" → type: concept
  ← KHÔNG nhận ra "Điều 12" là entity quan trọng nhất!
```

### Fix của chúng tôi

```python
LEGAL_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.
Nhiệm vụ: xử lý luật, nghị định, thông tư, quyết định của Nhà nước Việt Nam.

Quy tắc bắt buộc:
1. Nhận diện cấu trúc phân cấp: PHẦN > CHƯƠNG > MỤC > ĐIỀU > KHOẢN > ĐIỂM
2. Mỗi "Điều X" là đơn vị pháp lý độc lập — TRÍCH DẪN NGUYÊN VĂN
3. Quan hệ tham chiếu giữa các điều luật rất quan trọng
4. Giữ nguyên số điều, khoản, điểm chính xác (Điều 12 khoản 2 điểm a)
5. Trả lời bằng tiếng Việt, dùng thuật ngữ pháp lý chính xác"""

def llm_model_func(prompt, system_prompt=None, ...):
    # Nếu LightRAG không truyền system_prompt → inject legal context
    effective_system = system_prompt if system_prompt is not None else LEGAL_SYSTEM_PROMPT
    return openai_complete(model, prompt, system_prompt=effective_system, ...)
```

**Kết quả sau fix:** LLM giờ extract đúng:
```
Entity: "Điều 12" → type: dieu ✅
Entity: "Thư viện chuyên ngành" → type: khai_niem ✅
Relation: "Điều 12" → [định nghĩa] → "Thư viện chuyên ngành" ✅
```

---

## Vấn đề 2 — Chunking cắt giữa Điều luật

### Bằng chứng vấn đề

*"Token-based chunking không biết 'Điều 12' là một đơn vị logic quan trọng."*

```
Nội dung gốc (300 tokens mỗi Điều):
  ...Điều 11. [290 tokens nội dung]
  Điều 12. Thư viện chuyên ngành...
  [310 tokens tiếp theo]...
  Điều 13. [...]

Chunking token-based (max 400 tokens):
  Chunk 5: "...Điều 11 [290 tokens] Điều 12. Thư viện" ← CẮT GIỮA ĐIỀU 12
  Chunk 6: "chuyên ngành là thư viện có tài nguyên..." ← NỬA SAU ĐIỀU 12

→ Query "Điều 12 quy định gì?" 
→ Retrieve được 2 chunk nhưng mỗi cái thiếu một nửa
→ Answer: "Tôi không có đủ thông tin" ← SAI
```

### Fix: Vietnamese Legal Chunker

```python
LEGAL_BOUNDARY = re.compile(
    r'(?=\n(?:PHẦN|CHƯƠNG|MỤC|ĐIỀU|Phần|Chương|Mục|Điều)\s+[\dIVXivx]+[\.:]?\s)',
    re.UNICODE
)

def vietnamese_legal_chunker(tokenizer, content, ...):
    # BƯỚC 1: Split tại boundary pháp lý TRƯỚC
    parts = LEGAL_BOUNDARY.split(content)
    
    for part in parts:
        # BƯỚC 2: Nếu part vừa trong 1 chunk → giữ nguyên
        if len(tokens) <= chunk_token_size:
            results.append({"content": part, ...})
        
        # BƯỚC 3: Nếu Điều quá dài → split token + overlap
        else:
            for start in range(0, len(tokens), chunk_size - overlap):
                results.append({"content": decode(tokens[start:start+chunk_size])})
    
    # BƯỚC 4: Fallback nếu không tìm thấy boundary
    if not results:
        # token split thông thường → tương thích mọi loại tài liệu
```

**Kết quả sau fix:**
```
Chunk A: "Điều 11. [nội dung đầy đủ]"  ← TRỌN VẸN
Chunk B: "Điều 12. Thư viện chuyên ngành là thư viện có tài nguyên..."  ← TRỌN VẸN
Chunk C: "Điều 13. [...]"

→ Query "Điều 12 quy định gì?"
→ Retrieve Chunk B → answer đầy đủ ✅
```

---

## Vấn đề 3 — Entity Types generic không phù hợp

### Bằng chứng vấn đề

```python
# LightRAG default — dùng cho English general documents:
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

# → "Điều 12" không vừa vào bất kỳ type nào
# → LLM lúng túng → đôi khi label là "event", đôi khi bỏ qua
# → Knowledge Graph: thiếu hàng trăm node pháp lý quan trọng
```

### Fix: Legal Entity Types

```python
"entity_types": [
    "dieu",       # Điều X — đơn vị pháp lý cơ bản ← QUAN TRỌNG NHẤT
    "khoan",      # Khoản trong Điều → "khoản 2 Điều 12"
    "muc",        # Mục trong Chương → "Mục 1: Thư viện công cộng"
    "chuong",     # Chương trong Luật → "Chương III: Hoạt động thư viện"
    "phan",       # Phần (cấp cao nhất)
    "to_chuc",    # Tổ chức, cơ quan nhà nước → "Bộ VHTT&DL"
    "khai_niem",  # Khái niệm pháp lý → "Thư viện chuyên ngành"
    "hanh_vi",    # Hành vi được/không được làm
    "doi_tuong",  # Đối tượng áp dụng → "cán bộ công chức"
    "chinh_sach", # Chính sách → "xã hội hóa thư viện"
]
```

---

## Vấn đề 4 — Null Bytes trong PDF: Lỗi 400

### Bằng chứng vấn đề

```
ERROR: OpenAI API Call Failed, Error code: 400
"We could not parse the JSON body of your request"
```

*"Lỗi này xuất hiện ở chunk thứ 9/95. Tất cả 8 chunk trước đó bình thường. Tại sao?"*

**Root cause:** PDF scan dùng OCR kém chất lượng tạo ra **null bytes** (`\x00`) trong text. JSON không cho phép null bytes trong string — OpenAI API nhận JSON body không hợp lệ → từ chối với 400.

### Fix: _sanitize function

```python
def _sanitize(text: str) -> str:
    # B1: Re-encode UTF-8 → loại bỏ byte sequence không hợp lệ
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    
    # B2: Xóa null bytes và control characters (giữ \n, \t)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or (ord(ch) >= 32 and ch != "\x7f")
    )
    return text

# Áp dụng ở 2 chỗ:
# 1. Chunker — trước khi tokenize
part = _sanitize(part)

# 2. llm_model_func — trước khi gửi API
prompt = _sanitize(prompt)
```

---

## Vấn đề 5 — Entity Extract Max Gleaning

*"RAG-Anything gốc mặc định `entity_extract_max_gleaning=1` — chỉ retry 1 lần nếu LLM extract không đủ entity. Với văn bản pháp lý dày đặc thông tin:"*

```
Chunk chứa "Điều 12. Thư viện chuyên ngành. Điều 13. Thư viện chuyên dụng.
            Điều 14. Điều kiện thành lập. Điều 15. Hồ sơ đăng ký."

Lần 1: LLM extract được Điều 12, Điều 13
Gleaning lần 2: "Bạn đã bỏ qua Điều 14 và Điều 15, hãy extract thêm"
  → LLM extract Điều 14, Điều 15

→ Tăng từ 2 entity → 4 entity nhờ gleaning
```

**Fix:** Tăng `entity_extract_max_gleaning: 2` trong addon_params.

---

## Tóm tắt: RAG-Anything gốc vs Phiên bản của chúng tôi

| Vấn đề | RAG-Anything gốc | Phiên bản custom |
|---|---|---|
| System prompt | ❌ `None` | ✅ Legal Vietnamese |
| Chunking | ❌ Token-only | ✅ Legal boundary-first |
| Entity types | ❌ Generic English | ✅ 10 Vietnamese legal types |
| Null bytes | ❌ Crash với 400 | ✅ Sanitize tự động |
| Gleaning | ⚠️ 1 lần | ✅ 2 lần |
| Workspace | ❌ Shared namespace | ✅ Per-project isolation |
| Reranking | ❌ Không có | ✅ BGE local free |
