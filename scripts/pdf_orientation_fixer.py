"""
pdf_orientation_fixer.py (v5-RANKING)
─────────────────────────────────────────────────────────────────────
Optimized PDF Orientation Fixer using GPT-4o Visual Grid Ranking.
Created for RAG-Anything high-performance pipeline.
─────────────────────────────────────────────────────────────────────
"""

import os, sys, shutil, argparse, base64, io, time
from pathlib import Path
from PIL import Image
from loguru import logger
import fitz # PyMuPDF

try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai -q")
    from openai import OpenAI

def get_image_base64(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_img.thumbnail((1024, 1024))
    pil_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_orientation_grid(page: "fitz.Page") -> Image.Image:
    pix = page.get_pixmap(dpi=150)
    img_orig = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img90  = img_orig.rotate(-90, expand=True)
    img180 = img_orig.rotate(-180, expand=True)
    img270 = img_orig.rotate(-270, expand=True)
    max_w = max(img_orig.width, img90.width)
    max_h = max(img_orig.height, img90.height)
    grid = Image.new('RGB', (max_w * 2, max_h * 2), (255, 255, 255))
    grid.paste(img_orig, (0, 0))
    grid.paste(img90,    (max_w, 0))
    grid.paste(img180,   (0, max_h))
    grid.paste(img270,   (max_w, max_h))
    grid.thumbnail((1024, 1024))
    return grid

def detect_orientation_ranking(client: OpenAI, page: "fitz.Page") -> int:
    grid_img = create_orientation_grid(page)
    b64_img = get_image_base64(grid_img)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Grid order: 1:Top-Left, 2:Top-Right, 3:Bottom-Left, 4:Bottom-Right. Identify which version is UPRIGHT portrait Vietnamese text. Return ONLY the index (1, 2, 3, or 4)."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            max_tokens=5,
            temperature=0.0
        )
        ans = response.choices[0].message.content.strip()
        idx_map = {"1": 0, "2": 90, "3": 180, "4": 270}
        import re
        match = re.search(r'[1-4]', ans)
        if match: return idx_map[match.group()]
    except Exception as e:
        logger.error(f"Ranking Error: {e}")
    return 0

def fix_one_pdf(client: OpenAI, src: str, dst: str) -> int:
    doc = fitz.open(src)
    rotated = 0
    for i, page in enumerate(doc):
        angle = detect_orientation_ranking(client, page)
        if angle != 0:
            new_rot = (page.rotation + angle) % 360
            page.set_rotation(new_rot)
            logger.info(f"Page {i+1}: Picked #{angle//90 + 1} -> result {new_rot}°")
            rotated += 1
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    doc.save(dst, garbage=3, deflate=True, clean=True)
    doc.close()
    return rotated

def process_folder(api_key: str, input_dir: str, output_dir: str):
    client = OpenAI(api_key=api_key)
    src = Path(input_dir)
    dst = Path(output_dir)
    pdfs = sorted(src.glob("*.pdf")) + sorted(src.glob("*.PDF"))
    logger.info(f"🚀 Orientation Fixer: {len(pdfs)} files")
    for i, pdf in enumerate(pdfs, 1):
        out = dst / pdf.name
        try:
            n = fix_one_pdf(client, str(pdf), str(out))
            res = "✅ FIXED" if n > 0 else "⏭️ OK"
            logger.info(f"[{i}/{len(pdfs)}] {res} | {pdf.name} ({n} pages)")
        except Exception as e:
            logger.error(f"[{i}/{len(pdfs)}] ❌ FAILED | {pdf.name}: {e}")
            shutil.copy2(pdf, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()
    if not args.key: sys.exit(1)
    process_folder(args.key, args.input, args.output)
