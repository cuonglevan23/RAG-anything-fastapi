# 1. Clone repo
git clone <your-repo-link>
cd RAG-Anything

# 2. Tạo file .env
cp env.example .env
# Thêm OPENAI_API_KEY vào file .env

# 3. Build và chạy
docker build -t rag-anything .
docker run -p 8501:8501 --env-file .env rag-anything