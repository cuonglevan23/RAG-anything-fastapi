import os
import io
import json
import base64
import time
import typing
from PIL import Image
from loguru import logger
from openai import OpenAI
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig
from app.core.config import settings

# Patch transfromers rope conflict
if not hasattr(AutoConfig, "_original_from_pretrained"):
    AutoConfig._original_from_pretrained = AutoConfig.from_pretrained

def robust_patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    config = cls._original_from_pretrained(pretrained_model_name_or_path, **kwargs)
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if "rope_type" in config.rope_scaling and "type" in config.rope_scaling:
            logger.info("Patching rope_scaling: removing legacy 'type' field.")
            del config.rope_scaling["type"]
    return config

AutoConfig.from_pretrained = classmethod(robust_patched_from_pretrained)

# MinerU & VLM Utilities
try:
    from mineru_vl_utils import MinerUClient
except ImportError:
    MinerUClient = None
    logger.warning("mineru_vl_utils not found. VLM Pipeline won't work without it.")

from mineru.utils.enum_class import MakeMode, ImageType
from mineru.utils.pdf_image_tools import load_images_from_pdf
from mineru.backend.vlm.model_output_to_middle_json import result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox

import asyncio
from openai import AsyncOpenAI

class OpenAIUsageTracker:
    def __init__(self):
        self.reset()
        self._lock = asyncio.Lock()

    def reset(self):
        self.stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "total_latency": 0.0
        }

    async def record_usage(self, response, latency: float):
        async with self._lock:
            self.stats["call_count"] += 1
            self.stats["total_latency"] += latency
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.stats["prompt_tokens"] += usage.prompt_tokens
                self.stats["completion_tokens"] += usage.completion_tokens
                self.stats["total_tokens"] += usage.total_tokens

    def log_summary(self):
        call_count = self.stats['call_count']
        logger.info(f"OpenAI Usage: {call_count} calls, {self.stats['total_tokens']} tokens.")


class CustomOpenAIPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError("OpenAI API Key is missing.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model_name = settings.LLM_MODEL if hasattr(settings, "LLM_MODEL") else "gpt-4o"
        self.usage_tracker = OpenAIUsageTracker()
        self.api_semaphore = asyncio.Semaphore(10)  # Giới hạn 10 calls song song

        if MinerUClient is None:
            raise ImportError("mineru_vl_utils is required for MinerU2.5 layout analysis.")

        logger.info(f"Initializing MinerU2.5 (Qwen2VL)...")
        model_id = "opendatalab/MinerU2.5-2509-1.2B"

        try:
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
            raise

    def _pil_image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def _call_openai(self, image: Image.Image, prompt: str) -> typing.Tuple[str, float]:
        async with self.api_semaphore:
            base64_image = self._pil_image_to_base64(image)
            start_time = time.time()
            try:
                response = await self.client.chat.completions.create(
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
                    max_tokens=2048,
                    timeout=60.0 # Tránh treo vĩnh viễn
                )
                latency = time.time() - start_time
                await self.usage_tracker.record_usage(response, latency)
                return response.choices[0].message.content.strip(), latency
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                return "", 0.0

    def _map_type_to_prompt(self, block_type: str) -> str:
        block_type = block_type.lower()
        if block_type == "table":
            return (
                "You are an expert OCR model specializing in Vietnamese administrative and medical documents.\n"
                "Carefully read the table in this image and output it in HTML <table> format.\n\n"
                "CRITICAL RULES — follow exactly:\n"
                "1. Read each cell INDEPENDENTLY. Do NOT infer a cell's value from neighboring cells.\n"
                "2. Checkbox columns ('Có'/'Không', 'Có'/'Không', 'Yes'/'No'):\n"
                "   - A ✓, ✗, X, or filled mark means that specific column is selected.\n"
                "   - An EMPTY cell means NOT selected — output empty <td></td>.\n"
                "   - NEVER move a checkmark from one column to its neighbor.\n"
                "3. Preserve the exact column count and alignment from the image.\n"
                "4. For merged cells, use colspan/rowspan in HTML.\n"
                "5. Output ONLY the HTML table. No explanation, no markdown fences.\n\n"
                "Example of correct checkbox output:\n"
                "<tr><td>Tăng huyết áp</td><td>✓</td><td></td></tr>  <!-- Có=✓, Không=empty -->\n"
                "<tr><td>Bệnh tim mạch</td><td></td><td>✓</td></tr>  <!-- Có=empty, Không=✓ -->"
            )
        elif block_type == "equation":
            return r"Recognize the formula in this image and output it in LaTeX format. Wrap the formula in \[ and \]."
        return (
            "You are an expert OCR model for Vietnamese documents.\n"
            "Transcribe ALL text exactly as it appears. Preserve line breaks.\n"
            "Do not translate, interpret, or summarize. Output raw OCR text only."
        )


    async def _process_block(self, image: Image.Image, block_info: dict) -> dict:
        img_w, img_h = image.size
        if 'bbox' not in block_info: return None
        bbox = block_info['bbox']
        if len(bbox) != 4: return None
        x1, y1, x2, y2 = bbox
        if all(0.0 <= c <= 1.01 for c in [x1, y1, x2, y2]):
            x1 *= img_w; y1 *= img_h; x2 *= img_w; y2 *= img_h
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))
        crop = image.crop((x1, y1, x2, y2))
        block_type = block_info.get('type', 'text')
        prompt = self._map_type_to_prompt(block_type)
        content = ""
        if prompt:
            content, _ = await self._call_openai(crop, prompt)
            if content.startswith("```"):
                lines = content.split('\n')
                if len(lines) > 2: content = "\n".join(lines[1:-1]).strip()
        return {"type": block_type, "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h], "content": content, "angle": 0}

    async def process_pdf(self, input_path: str, output_dir: str, file_name: str) -> str:
        """Processes PDF and returns the path to the generated .md file"""
        self.usage_tracker.reset()
        
        with open(input_path, "rb") as f:
            pdf_bytes = f.read()

        # Render images (Blocking CPU task, but fast enough)
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        pil_images = [img["img_pil"] for img in images_list]
        all_page_blocks = []

        for i, image in enumerate(pil_images):
            start_page = time.time()
            logger.info(f"VLM Parsing Page {i + 1}/{len(pil_images)}")
            
            # Step 1: Layout Detection (GPU - Run in thread to avoid blocking event loop)
            try:
                # Chạy MinerU (CPU/GPU bound) in thread pool
                layout_blocks = await asyncio.to_thread(self.mineru_client.two_step_extract, image)
            except Exception as e:
                logger.error(f"MinerU layout analysis failed: {e}")
                layout_blocks = []

            if layout_blocks and isinstance(layout_blocks[0], dict):
                layout_blocks.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

            # Step 2: Content Extraction (API - Concurrent)
            tasks = [self._process_block(image, block) for block in layout_blocks]
            page_blocks_results = await asyncio.gather(*tasks)
            page_blocks = [b for b in page_blocks_results if b is not None]
            
            all_page_blocks.append(page_blocks)
            logger.info(f"Page {i+1} processed in {time.time()-start_page:.1f}s")

        local_md_dir = os.path.join(output_dir, file_name, "vlm")
        local_image_dir = os.path.join(local_md_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # Middle JSON creation
        middle_json = result_to_middle_json(all_page_blocks, images_list, pdf_doc, image_writer)
        pdf_info = middle_json["pdf_info"]

        md_writer.write_string(f"{file_name}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=4))
        md_writer.write_string(f"{file_name}_model.json", json.dumps(all_page_blocks, ensure_ascii=False, indent=4))
        md_writer.write(f"{file_name}_origin.pdf", pdf_bytes)

        md_content = union_make(pdf_info, MakeMode.MM_MD, os.path.basename(local_image_dir))
        md_file_path = os.path.join(local_md_dir, f"{file_name}.md")
        md_writer.write_string(f"{file_name}.md", md_content)

        try:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
        except Exception as e:
            logger.warning(f"Failed to draw layout PDF: {e}")

        self.usage_tracker.log_summary()
        return md_file_path



# ── Singleton: load MinerU2.5 MỘT LẦN DUY NHẤT khi server khởi động ──────────
import threading
_vlm_pipeline_instance: "CustomOpenAIPipeline | None" = None
_vlm_pipeline_lock = threading.Lock()

def get_vlm_pipeline() -> "CustomOpenAIPipeline":
    """
    Trả về singleton instance của CustomOpenAIPipeline.
    Thread-safe: dùng threading.Lock để tránh double-init race condition.
    Lần đầu gọi: load MinerU2.5 Qwen2VL vào GPU (~30s đầu tiên).
    Các lần sau: trả về ngay lập tức, không reload.
    """
    global _vlm_pipeline_instance
    if _vlm_pipeline_instance is not None:
        return _vlm_pipeline_instance          # fast path — không cần lock
    with _vlm_pipeline_lock:                   # chỉ 1 thread vào đây cùng lúc
        if _vlm_pipeline_instance is None:     # double-checked locking
            logger.info("Initializing VLM Pipeline singleton (first call)...")
            _vlm_pipeline_instance = CustomOpenAIPipeline(api_key=settings.OPENAI_API_KEY)
            logger.info("VLM Pipeline singleton ready — will be reused for all files.")
    return _vlm_pipeline_instance
