# -*- coding: utf-8 -*-
"""rag-anything_fixed_vlm.ipynb

Tích hợp CustomOpenAIPipeline (VLM) vào RAGAnything và sửa lỗi rope_type / MinerU conflict.
"""

# Basic installation
!pip
install
raganything
!pip
install
'raganything[all]'
!pip
install
nest_asyncio
!pip
install - -upgrade
pip
!pip
install
"mineru[all]>=2.7.0"
!pip
install - U
transformers
!pip
install
openai

import os
import io
import json
import base64
import time
import typing
import asyncio
import sys
import numpy as np
from PIL import Image
from loguru import logger
from openai import OpenAI
import transformers
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig

# === FIX: Transformers rope_scaling conflict patch ===
# Lỗi: "Found conflicts between 'rope_type=default' and 'type=mrope'"
# Patch này sẽ loại bỏ field 'type' cũ nếu 'rope_type' đã tồn tại trong config khi load model.
# === FIX: Transformers rope_scaling conflict patch ===
# Patch này sửa lỗi "Found conflicts between 'rope_type=default' and 'type=mrope'"
# Bằng cách lưu lại bản gốc vào một thuộc tính ẩn của class, tránh bị đè lặp (RecursionError).

if not hasattr(AutoConfig, "_original_from_pretrained"):
    AutoConfig._original_from_pretrained = AutoConfig.from_pretrained


def robust_patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    # Luôn gọi thông qua bản gốc đã lưu
    config = cls._original_from_pretrained(pretrained_model_name_or_path, **kwargs)
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if "rope_type" in config.rope_scaling and "type" in config.rope_scaling:
            logger.info("Patching rope_scaling: removing legacy 'type' field.")
            del config.rope_scaling["type"]
    return config


AutoConfig.from_pretrained = classmethod(robust_patched_from_pretrained)
logger.info("AutoConfig.from_pretrained has been patched (Robust mode).")

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# === MinerU & VLM Utilities ===
try:
    from mineru_vl_utils import MinerUClient
except ImportError:
    logger.error("mineru_vl_utils not found. Please install it to use MinerU2.5.")
    MinerUClient = None

from mineru.utils.enum_class import CategoryId, ModelPath, BlockType, MakeMode, ImageType
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.pdf_image_tools import load_images_from_pdf, images_bytes_to_pdf_bytes
from mineru.backend.vlm.model_output_to_middle_json import result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox

# For Colab
try:
    from google.colab import userdata
except ImportError:
    userdata = None


# === OpenAI Usage Tracker ===
class OpenAIUsageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "total_latency": 0.0
        }

    def record_usage(self, response, latency: float):
        self.stats["call_count"] += 1
        self.stats["total_latency"] += latency
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.stats["prompt_tokens"] += usage.prompt_tokens
            self.stats["completion_tokens"] += usage.completion_tokens
            self.stats["total_tokens"] += usage.total_tokens

    def log_summary(self):
        call_count = self.stats['call_count']
        total_latency = self.stats['total_latency']
        avg_latency = total_latency / call_count if call_count > 0 else 0
        logger.info("=" * 40)
        logger.info("       OPENAI USAGE SUMMARY       ")
        logger.info("=" * 40)
        logger.info(f"Total API Calls    : {call_count}")
        logger.info(f"Total Tokens       : {self.stats['total_tokens']}")
        logger.info("=" * 40)


# === Custom VLM Parsing Pipeline ===
class CustomOpenAIPipeline:
    def __init__(self, output_dir: str, api_key: str):
        self.output_dir = output_dir
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError("OpenAI API Key is missing.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = "gpt-4o"
        self.usage_tracker = OpenAIUsageTracker()

        if MinerUClient is None:
            raise ImportError("mineru_vl_utils is required for MinerU2.5 layout analysis.")

        logger.info(f"Initializing MinerU2.5 (Qwen2VL)...")
        model_id = "opendatalab/MinerU2.5-2509-1.2B"

        try:
            # Dùng AutoConfig đã được patch ở trên
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if not hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = 32768

            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, config=config, torch_dtype="auto", device_map="auto"
            )
            self.qwen_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self.mineru_client = MinerUClient(
                backend="transformers", model=self.qwen_model, processor=self.qwen_processor
            )
            logger.info("MinerU2.5 Layout model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MinerU2.5: {e}")
            raise e

    def _pil_image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_openai(self, image: Image.Image, prompt: str) -> typing.Tuple[str, float]:
        base64_image = self._pil_image_to_base64(image)
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=2048
            )
            latency = time.time() - start_time
            self.usage_tracker.record_usage(response, latency)
            return response.choices[0].message.content.strip(), latency
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "", 0.0

    def _map_type_to_prompt(self, block_type: str) -> str:
        block_type = block_type.lower()
        if block_type == "table":
            return "Recognize the table in this image and output it in HTML format. Ensure the structure is correct."
        elif block_type == "equation":
            return r"Recognize the formula in this image and output it in LaTeX format. Wrap the formula in \[ and \]."
        return "Recognize the text in this image. Output the raw OCR results."

    def _process_block(self, image: Image.Image, block_info: dict) -> dict:
        img_w, img_h = image.size
        if 'bbox' not in block_info: return None
        bbox = block_info['bbox']
        if len(bbox) != 4: return None
        x1, y1, x2, y2 = bbox
        if all(0.0 <= c <= 1.01 for c in [x1, y1, x2, y2]):
            x1 *= img_w;
            y1 *= img_h;
            x2 *= img_w;
            y2 *= img_h
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))
        crop = image.crop((x1, y1, x2, y2))
        block_type = block_info.get('type', 'text')
        prompt = self._map_type_to_prompt(block_type)
        content = ""
        if prompt:
            content, _ = self._call_openai(crop, prompt)
            if content.startswith("```"):
                lines = content.split('\n')
                if len(lines) > 2: content = "\n".join(lines[1:-1]).strip()
        return {"type": block_type, "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h], "content": content,
                "angle": 0}

    def process_pdf(self, input_path: str) -> str:
        """Processes PDF and returns the path to the generated .md file"""
        self.usage_tracker.reset()
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        with open(input_path, "rb") as f:
            pdf_bytes = f.read()

        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        pil_images = [img["img_pil"] for img in images_list]
        all_page_blocks = []

        for i, image in enumerate(pil_images):
            logger.info(f"VLM Parsing Page {i + 1}/{len(pil_images)}")
            try:
                layout_blocks = self.mineru_client.two_step_extract(image)
            except Exception as e:
                logger.error(f"MinerU layout analysis failed: {e}")
                layout_blocks = []

            if layout_blocks and isinstance(layout_blocks[0], dict):
                layout_blocks.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

            page_blocks = []
            for block in layout_blocks:
                processed = self._process_block(image, block)
                if processed: page_blocks.append(processed)
            all_page_blocks.append(page_blocks)

        local_md_dir = os.path.join(self.output_dir, file_name, "vlm")
        local_image_dir = os.path.join(local_md_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # ===== Generate Middle JSON =====
        middle_json = result_to_middle_json(all_page_blocks, images_list, pdf_doc, image_writer)
        pdf_info = middle_json["pdf_info"]

        md_writer.write_string(
            f"{file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4)
        )
        md_writer.write_string(
            f"{file_name}_model.json",
            json.dumps(all_page_blocks, ensure_ascii=False, indent=4)
        )
        md_writer.write(f"{file_name}_origin.pdf", pdf_bytes)

        # ===== Markdown =====
        md_content = union_make(pdf_info, MakeMode.MM_MD, os.path.basename(local_image_dir))

        md_file_path = os.path.join(local_md_dir, f"{file_name}.md")
        md_writer.write_string(f"{file_name}.md", md_content)

        # ===== Content list =====
        content_list = union_make(
            pdf_info,
            MakeMode.CONTENT_LIST,
            os.path.basename(local_image_dir)
        )
        md_writer.write_string(
            f"{file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4)
        )

        # ===== Draw layout bbox PDF =====
        try:
            draw_layout_bbox(
                pdf_info,
                pdf_bytes,
                local_md_dir,
                f"{file_name}_layout.pdf"
            )
        except Exception as e:
            logger.warning(f"Failed to draw layout PDF: {e}")

        self.usage_tracker.log_summary()
        return md_file_path


# === Main RAG Execution ===
async def main():
    # 1. API Configuration
    if userdata:
        api_key = userdata.get("OPENAI_API_KEY")
    else:
        api_key = "YOUR_OPENAI_API_KEY_HERE"

    input_pdf = "/content/46.signed-2_origin.pdf"  # File nguồn của bạn
    output_workspace = "./output"
    file_name = os.path.splitext(os.path.basename(input_pdf))[0]
    parsed_md_path = os.path.join(output_workspace, file_name, "vlm", f"{file_name}.md")

    # 2. VLM Parsing Stage (Cờ để skip nếu đã có kết quả)
    if os.path.exists(parsed_md_path):
        logger.info(f"Đã tìm thấy file Markdown: {parsed_md_path}. Bỏ qua bước VLM Parsing để tiết kiệm token.")
    else:
        logger.info("Stage 1: Parsing PDF using CustomOpenAIPipeline (with Transformers Patch)...")
        vlm_pipeline = CustomOpenAIPipeline(output_dir=output_workspace, api_key=api_key)
        parsed_md_path = vlm_pipeline.process_pdf(input_pdf)
        logger.info(f"VLM Parsing finished. Markdown created at: {parsed_md_path}")

    # 3. RAGAnything setup
    working_dir = "./rag_storage"
    config = RAGAnythingConfig(
        working_dir=working_dir,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini", prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=api_key, **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=1536, max_token_size=8192,
        func=lambda texts: openai_embed(texts, model="text-embedding-3-small", api_key=api_key),
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # 4. Stage 2: Indexing with Bypassing internal MinerU
    # Kiểm tra xem folder rag_storage đã có dữ liệu chưa (thường kiểm tra file kv_store_full_text_llm_response_cache.json hoặc tương tự)
    # Ở đây dùng cách đơn giản: nếu đã tồn tại kv_store thì coi như đã index
    if os.path.exists(os.path.join(working_dir, "kv_store_full_text_llm_response_cache.json")):
        logger.info(f"Đã thấy dữ liệu trong {working_dir}. Chuyển thẳng sang bước Query.")
    else:
        logger.info("Stage 2: Indexing parsed content into RAG system...")
        if os.path.exists(parsed_md_path):
            with open(parsed_md_path, "r", encoding="utf-8") as f:
                md_text = f.read()

            # Đảm bảo instance lightrag đã được khởi tạo
            await rag._ensure_lightrag_initialized()

            # Thêm trực tiếp vào Knowledge Base
            await rag.lightrag.ainsert(md_text)
            logger.info("Successfully indexed parsed Markdown.")
        else:
            logger.error(f"Markdown file not found: {parsed_md_path}")
            return

    # 5. Query
    query = "Liệt kê toàn bộ các điều luật có trong văn bản theo danh sách đầy đủ."
    logger.info(f"Stage 3: Querying: {query}")
    text_result = await rag.aquery(query, mode="hybrid", top_k=100, response_type="Structured List")
    print("\n" + "=" * 50)
    print("QUERY RESULT:")
    print("=" * 50)
    print(text_result)


if __name__ == "__main__":
    if nest_asyncio:
        nest_asyncio.apply()
    asyncio.run(main())
