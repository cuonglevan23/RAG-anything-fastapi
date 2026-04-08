# Hướng dẫn cài đặt Backend RAG-Anything

Dự án này là một dịch vụ backend chuyên sâu. Dưới đây là 2 cách để thiết lập và chạy hệ thống Backend API (sử dụng **FastAPI**), bỏ qua quá trình khởi chạy giao diện frontend/Streamlit không cần thiết.

Bạn có thể chạy trực tiếp trên **Môi trường Python (Local)** hoặc thông qua **Docker**.

---

## Cách 1: Thiết lập môi trường Python trực tiếp (Local)

**Yêu cầu hệ thống:**
- Python 3.10 trở lên.
- Cài đặt các thư viện hệ thống (System dependencies) đáp ứng tiến trình xử lý tài liệu file cứng.
  - **Trên Linux/Ubuntu:**
    ```bash
    sudo apt-get update
    sudo apt-get install libreoffice tesseract-ocr tesseract-ocr-vie poppler-utils ffmpeg libsm6 libxext6
    ```
  - **Trên macOS:** *Có thể cài qua Homebrew*
    ```bash
    brew install --cask libreoffice
    brew install tesseract tesseract-lang poppler ffmpeg
    ```

**Các bước thực hiện:**

**Bước 1: Clone kho lưu trữ (Repository)**
```bash
git clone <your-repo-link>
cd RAG-Anything
```

**Bước 2: Cài đặt công cụ quản lý gói `uv`** *(Khuyến nghị)*
Dự án sử dụng `uv` giúp quản lý và cài đặt dependency tốc độ cao.
```bash
# Trên Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(Hoặc có thể cài đặt thông qua pip: `pip install uv`)*

**Bước 3: Cài đặt Dependencies Python**
Tải và cài đặt tất cả các gói dựa trên cấu hình dự án:
```bash
uv sync --all-extras
```

**Bước 4: Cấu hình biến môi trường**
Tạo file cấu hình `.env` dựa trên file mẫu:
```bash
cp env.example .env
```
Mở file `.env` bằng trình chỉnh sửa và thiết lập khóa API cho `OPENAI_API_KEY` cũng như một loạt những cấu hình khác nếu cần.

**Bước 5: Khởi chạy Backend API**
Khởi động hệ thống server FastAPI:
```bash
uv run python app/main.py
```
*(Lệnh này gọi chạy uvicorn nội bộ phục vụ app trên cổng 8000)*

API Backend của ứng dụng giờ đây chạy tại: **http://localhost:8000**

---

## Cách 2: Thiết lập bằng Docker (Khuyến nghị)

Triển khai thông qua Docker sẽ đóng gói đồng bộ toàn bộ môi trường và giúp bạn tối giản công sức cài đặt các dependency cho máy tính cá nhân.

**Yêu cầu thiết yếu:**
- Máy tính của bạn đã được cài đặt sẵn [Docker](https://docs.docker.com/get-docker/).

**Các bước thực hiện:**

**Bước 1: Clone dự án**
```bash
git clone <your-repo-link>
cd RAG-Anything
```

**Bước 2: Chuẩn bị biến môi trường**
Tạo file `.env` kèm theo khóa bí mật của bạn:
```bash
cp env.example .env
```
Nhớ thêm khóa `OPENAI_API_KEY` của bạn vào nhé.

**Bước 3: Build Docker Image**
Xây dựng Docker image `rag-anything` từ `Dockerfile`:
```bash
docker build -t rag-anything .
```
*(Quá trình này tốn một vài phút giúp tải trước hệ thống LibreOffice, Tesseract OCR...).*

**Bước 4: Chạy Docker Container của Backend API**
Vì mặc định hệ thống chạy Streamlit, bạn cần gọi ghi đè lại lệnh khởi động của image để hệ thống phục vụ Backend FastAPI tại cổng `8000`:
```bash
docker run -p 8000:8000 --env-file .env rag-anything uv run python app/main.py
```

Sau khi chạy thành công, máy chủ Backend sẽ phục vụ tại: **http://localhost:8000**

---

## Hướng dẫn sử dụng các Endpoint API (bằng Terminal `curl`)

Sau khi Backend đã hoạt động ở `http://localhost:8000`, bạn có thể kiểm tra danh sách đầy đủ các API tại: **http://localhost:8000/docs** (Swagger UI).

Dưới đây là một số API chính và cách gọi chúng bằng lệnh `curl`:

### 1. Lấy danh sách Workspace / Project
Liệt kê tất cả các project (workspace) đang có sẵn trong kho dữ liệu.
```bash
curl -X GET "http://localhost:8000/api/v1/rag/projects" \
  -H "accept: application/json"
```

### 2. Upload file vào một Project cụ thể
Upload từng file (hỗ trợ .pdf, .txt, .docx, .png, etc.) lên hệ thống để RAG nhận diện. Quá trình xử lý file sẽ chạy ngầm.
```bash
curl -X POST "http://localhost:8000/api/v1/rag/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "project_id=my_workspace" \
  -F "file=@/path/to/your/document.pdf"
```
*(Kết quả trả về sẽ cấp cho bạn một `task_id` để theo dõi tiến trình)*

### 3. Upload nhiều file cùng lúc (Batch Upload)
Trường hợp bạn cần upload cùng lúc nhiều file vào chung một namespace project.
```bash
curl -X POST "http://localhost:8000/api/v1/rag/upload_batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "project_id=my_workspace" \
  -F "files=@/path/to/your/doc1.pdf" \
  -F "files=@/path/to/your/doc2.pdf"
```

### 4. Kiểm tra trạng thái xử lý File
Vì quá trình bóc tách dữ liệu chạy ngầm, bạn dùng `task_id` nhận được ở bước upload để kiểm tra xem file đã sẵn sàng phục vụ RAG chưa.
```bash
curl -X GET "http://localhost:8000/api/v1/rag/status/<thay-task-id-vao-day>" \
  -H "accept: application/json"
```

### 5. Truy vấn và Đặt câu hỏi (RAG Query)
Truy vấn thông tin dựa trên dữ liệu bạn đã upload. Tham số `mode` có thể là `hybrid` hoặc `naive`.
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my_workspace",
    "query": "Tóm tắt định hướng của tài liệu file này là gì?",
    "mode": "hybrid",
    "top_k": 5
  }'
```