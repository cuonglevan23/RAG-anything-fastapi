# Phần 4 — RAG-Anything: Multimodal & Parsing Pipeline

> ⏱ **10 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"LightRAG giải quyết vấn đề hiểu quan hệ giữa các điều khoản. Nhưng còn một vấn đề khác: tài liệu thực tế không phải chỉ là text. Và đây là nơi RAG-Anything đóng vai trò."*

---

## Vấn đề: Tài liệu thực tế phức tạp hơn ta nghĩ

*"Hãy nhìn vào một tài liệu pháp lý điển hình:"*

```
Trang 1: Text thông thường → ✅ LightRAG gốc xử lý được
Trang 5: Bảng so sánh 4 loại thư viện → ❌ LightRAG gốc BỎ QUA
Trang 12: Sơ đồ tổ chức hệ thống thư viện → ❌ BỎ QUA
Trang 20: Công thức tính diện tích tối thiểu → ❌ BỎ QUA
Trang 30: Tài liệu SCAN (không có text layer) → ❌ KHÔNG ĐỌC ĐƯỢC
```

*"Nếu bạn hỏi 'Diện tích tối thiểu của phòng đọc là bao nhiêu?' — câu trả lời nằm trong công thức trên trang 20. LightRAG gốc không bao giờ đọc được trang đó."*

---

## RAG-Anything — Giải pháp đa phương thức

*"RAG-Anything là extension của LightRAG, thêm một lớp PARSING ở giữa: trước khi đưa content vào LightRAG, nó xử lý mọi loại file và chuyển tất cả thành text có cấu trúc."*

```
┌─────────────────────────────────────────────────────┐
│                   RAG-Anything                       │
│                                                      │
│  Input: bất kỳ file nào                              │
│  PDF / DOCX / PNG / JPG / Excel / PPT / MD           │
│                     ↓                                │
│           PARSING PIPELINE                           │
│  ┌────────────────────────────────────────┐         │
│  │ Step 1: Layout Detection (MinerU)       │         │
│  │   → Xác định: text / image / table     │         │
│  │                                        │         │
│  │ Step 2: Multimodal Processing (GPT-4o) │         │
│  │   → Image → text description           │         │
│  │   → Table → structured text            │         │
│  │   → Equation → LaTeX → text            │         │
│  │   → Scan → OCR → text                  │         │
│  └───────────────────┬────────────────────┘         │
│                       ↓                              │
│               Unified Markdown                       │
│                       ↓                              │
│              LightRAG Indexing                        │
└─────────────────────────────────────────────────────┘
```

---

## Parsing Pipeline — Từng bước kỹ thuật

### Bước 1: Layout Detection với MinerU

*"MinerU là một thư viện phân tích layout tài liệu. Nó dùng model học máy để nhận diện từng vùng trên mỗi trang."*

```python
# MinerU output mỗi trang:
[
    {"type": "text",     "bbox": [50, 100, 550, 200], "content": "Điều 12..."},
    {"type": "image",    "bbox": [50, 210, 550, 400], "image_path": "p5_img1.png"},
    {"type": "table",    "bbox": [50, 410, 550, 600], "cells": [...]},
    {"type": "equation", "bbox": [50, 610, 300, 680], "latex": "S = 0.5 \\times n"},
]
```

### Bước 2: Context Window — Thiết kế quan trọng

*"Đây là một quyết định thiết kế quan trọng. Khi gặp một hình ảnh, làm sao GPT-4o biết hình đó mô tả gì nếu không có ngữ cảnh?"*

```python
RAGAnythingConfig(
    context_window = 2,        # Lấy 2 trang xung quanh
    context_mode   = "page",   # Mode: page hoặc chunk
    max_context_tokens = 2000, # Token limit cho context
)

# Kết quả:
# Thay vì chỉ gửi ảnh: GPT-4o gửi:
#   "Trang 4: [text về thư viện chuyên ngành]
#    Trang 5: [ảnh này]   ← đang describe
#    Trang 6: [text tiếp theo về điều kiện]"
```

### Bước 3: Multimodal Processing

**Xử lý ảnh:**
```
Prompt → GPT-4o Vision:
  "Context xung quanh: Đây là sơ đồ trong Chương III về tổ chức thư viện.
   Mô tả chi tiết hình ảnh, tập trung vào quan hệ tổ chức và cấu trúc phân cấp."

Output:
  "Sơ đồ tổ chức hệ thống thư viện Việt Nam theo 3 cấp:
   - Cấp quốc gia: Thư viện Quốc gia Việt Nam
   - Cấp tỉnh: Thư viện tỉnh/thành phố (63 đơn vị)
   - Cấp cơ sở: Thư viện huyện, xã, trường học"
```

**Xử lý bảng:**
```
Input: Bảng HTML/PDF với 4 cột, 6 hàng

Output:
  "Bảng so sánh các loại thư viện theo Luật Thư Viện 2019:
   | Loại thư viện | Điều kiện thành lập | Cơ quan quản lý | Nguồn vốn |
   |---|---|---|---|
   | Công cộng | Điều 25-28 | UBND các cấp | Ngân sách nhà nước |
   ..."
```

---

## Custom VLM Parser — Những gì chúng tôi thêm

*"Ngoài MinerU, chúng tôi đã phát triển custom pipeline riêng — `CustomOpenAIPipeline` — được thiết kế cho tài liệu pháp lý Việt Nam scan chất lượng thấp."*

```
MinerU (gốc):
  Dùng model học máy cho layout detection
  Ưu điểm: Nhanh, không tốn API
  Nhược điểm: Kém với tài liệu scan, font chữ đặc biệt

CustomOpenAIPipeline (chúng tôi thêm):
  Dùng GPT-4o Vision để đọc toàn trang như con người
  Ưu điểm: Rất chính xác, hiểu ngữ cảnh pháp lý
  Nhược điểm: Tốn API (~70 calls cho 1 tài liệu 100 trang)
```

---

## Điểm mạnh và Điểm yếu của Parsing

### ✅ Điểm mạnh

| Điểm mạnh | Mô tả |
|---|---|
| Xử lý mọi loại file | PDF, DOCX, PNG, scan |
| Context-aware | Biết ảnh/bảng nằm trong ngữ cảnh gì |
| Chất lượng cao | GPT-4o Vision cho kết quả rất tốt |
| Fallback mechanism | Nếu MinerU lỗi → custom pipeline |

### ❌ Điểm yếu

| Điểm yếu | Mô tả | Mitigation |
|---|---|---|
| Chi phí API cao | ~70 GPT-4o calls/tài liệu 100 trang | Dùng gpt-4o-mini cho ảnh đơn giản |
| Chậm | 5-15 phút cho 1 tài liệu 100 trang | Background processing, async |
| Phụ thuộc internet | Phải gọi OpenAI API | Custom local VLM (roadmap) |
| VRAM cao | MinerU cần GPU 8-10GB | Tắt MinerU, dùng custom pipeline CPU |
