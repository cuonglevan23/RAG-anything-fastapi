# Phần 1 — Mở Đầu: Vấn Đề Thực Tế

> ⏱ **5 phút**

---

## 🎤 Lời thuyết minh chi tiết

---

*"Xin chào toàn thể mọi người. Hôm nay tôi sẽ trình bày về một dự án mà team phát triển trong thời gian vừa qua — một hệ thống hỏi đáp thông minh dựa trên tài liệu nội bộ, được xây dựng trên nền tảng LightRAG và RAG-Anything."*

*"Để bắt đầu, hãy để tôi vẽ cho các bạn một bức tranh thực tế."*

---

## Bức tranh thực tế

*"Hãy tưởng tượng bạn là một cán bộ pháp chế tại một cơ quan nhà nước. Trên máy tính bạn có 200 file PDF: luật, nghị định, thông tư, quyết định — mỗi file từ 50 đến 300 trang. Một ngày, lãnh đạo hỏi: 'Điều kiện để thành lập thư viện tư nhân là gì? Có cần vốn tối thiểu không? Hồ sơ nộp ở đâu?' — Bạn mất bao lâu để trả lời?"*

*"Với cách truyền thống: mở từng file PDF, Ctrl+F tìm từ khóa, đọc từng trang, tổng hợp thủ công — mất 30 phút đến 2 tiếng, và vẫn có thể bỏ sót điều khoản nào đó ở trang 180 của một file khác."*

*"Đây là bài toán thực tế mà hàng nghìn tổ chức đang gặp hàng ngày. Và đây là lý do tại sao chúng tôi xây dựng hệ thống này."*

---

## Tại sao không dùng ChatGPT thông thường?

*"Câu hỏi đầu tiên mà mọi người thường đặt ra: 'Dùng ChatGPT không được sao?' — Câu trả lời ngắn gọn là: không. Và đây là lý do cụ thể."*

**Vấn đề 1 — Hallucination (Bịa đặt):**

```
Người dùng: "Điều 12 Luật Thư Viện số 46/2019/QH14 quy định gì?"

ChatGPT: "Điều 12 quy định về việc thư viện phải nộp báo cáo hàng năm
          cho Bộ Văn hóa, Thể thao và Du lịch..." ← BỊA HOÀN TOÀN
```

*"LLM được train trên dữ liệu internet tổng hợp. Nó KHÔNG có luật nội bộ của công ty bạn, và quan trọng hơn: khi không biết, nó có xu hướng BỊA ĐẶT với độ tự tin cao — đây gọi là hallucination."*

**Vấn đề 2 — Không có tài liệu nội bộ:**
```
ChatGPT không biết:
  - Quy trình nội bộ của công ty
  - Hợp đồng chưa công bố
  - Báo cáo nội bộ
  - Văn bản pháp lý mới nhất (cutoff date)
```

**Vấn đề 3 — Không trích dẫn nguồn:**
*"Trong môi trường pháp lý, 'tôi tin rằng' là không chấp nhận được. Cần: 'Theo Điều 12 văn bản số X, trang Y'."*

---

## Yêu cầu của hệ thống

*"Vậy hệ thống chúng tôi cần xây dựng phải đáp ứng những tiêu chí sau:"*

| Yêu cầu | Mô tả |
|---|---|
| **Chính xác** | Câu trả lời phải đúng với nội dung tài liệu, không bịa |
| **Có nguồn** | Trích dẫn cụ thể: file nào, điều nào |
| **Đa dạng tài liệu** | PDF scan, bảng biểu, ảnh, không chỉ text thuần |
| **Đa ngôn ngữ** | Tiếng Việt, UTF-8 đặc thù, dấu thanh điệu |
| **Hiểu quan hệ** | "Điều 5 tham chiếu Điều 12" — cần traverse không chỉ tìm kiếm |
| **Riêng tư** | Dữ liệu nhạy cảm không được ra ngoài (on-premise capable) |

---

## Agenda hôm nay — Roadmap trình bày

*"Tôi sẽ dẫn các bạn qua 9 chủ đề theo thứ tự từ nền tảng đến chi tiết kỹ thuật:"*

1. **RAG là gì?** — Giải pháp cốt lõi
2. **LightRAG** — Engine GraphRAG chúng tôi chọn và tại sao
3. **RAG-Anything** — Xử lý đa phương thức
4. **Kiến trúc hệ thống** — Cấu trúc toàn bộ
5. **Custom domain tiếng Việt** — Những gì gốc chưa làm được và cách fix
6. **Reranking** — Tăng chính xác với model local miễn phí
7. **Đánh giá RAGAS** — Đo chất lượng bằng số liệu
8. **Kết quả & Bài học**
9. **Demo live**
