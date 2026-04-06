# 🐳 Dockerfile for RAG-Anything
# Optimized for Vietnamese Multimodal Processing

FROM python:3.10-slim

# 1. Install System Dependencies (LibreOffice, Tesseract, PDF tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    libreoffice-java-common \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv for faster dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Set Working Directory
WORKDIR /app

# 4. Copy Dependency Files
COPY pyproject.toml uv.lock ./

# 5. Install Python Dependencies
RUN uv sync --all-extras

# 6. Copy Application Code
COPY . .

# 7. Create persistent storage directories
RUN mkdir -p uploads output rag_storage

# 8. Environment Variables
ENV HOST=0.0.0.0
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 9. Expose Port
EXPOSE 8501

# 10. Entrypoint (Run Streamlit by default)
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py"]
