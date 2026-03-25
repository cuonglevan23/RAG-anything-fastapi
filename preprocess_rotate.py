"""
preprocess_rotate.py
─────────────────────────────────────────────────────────────────────
Tiền xử lý PDF: xoay các trang landscape → portrait.

Cách dùng trong Colab (cell):
─────────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(["python", "/content/RAG-anything-fastapi/preprocess_rotate.py",
                "--input", "/content/scan",
                "--output", "/content/scan_fix"])

Hoặc chỉnh INPUT_DIR / OUTPUT_DIR rồi chạy thẳng:
  python preprocess_rotate.py
─────────────────────────────────────────────────────────────────────
"""

import os, sys, shutil, argparse
from pathlib import Path
from loguru import logger

# ── Cài nhanh nếu chưa có ──────────────────────────────────────────
try:
    import fitz
except ImportError:
    os.system("pip install pymupdf -q")
    import fitz

try:
    from PIL import Image
except ImportError:
    os.system("pip install pillow -q")
    from PIL import Image


# ══════════════════════════════════════════════════════════════════
#  MODEL-BASED ROTATION (Tesseract OSD)
# ══════════════════════════════════════════════════════════════════

def detect_orientation_osd(page: "fitz.Page") -> int:
    """
    Dùng Tesseract OSD (Orientation and Script Detection) để tìm góc xoay.
    Trả về góc cần xoay thêm (0, 90, 180, 270).
    """
    try:
        import pytesseract
        # Render trang ở độ phân giải thấp (72 DPI) để OSD chạy nhanh
        pix = page.get_pixmap(dpi=72)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Chạy OSD
        osd = pytesseract.image_to_osd(img)
        # Output mẫu: "Orientation: 90\nRotate: 270\n..."
        for line in osd.splitlines():
            if "Rotate:" in line:
                return int(line.split(":")[-1].strip())
    except Exception as e:
        # Nếu không có tesseract hoặc lỗi, fallback về 0 (không xoay)
        return 0
    return 0


def fix_one_pdf(src: str, dst: str, use_model: bool = True) -> int:
    """
    Xoay tất cả trang của PDF về đúng chiều dựa trên model OSD hoặc heuristic.
    """
    doc     = fitz.open(src)
    rotated = 0

    for i, page in enumerate(doc):
        angle = 0
        if use_model:
            angle = detect_orientation_osd(page)

        # Nếu model không tìm thấy (angle=0) nhưng page đang nằm ngang (landscape)
        # thì vẫn có thể dùng heuristic làm cứu cánh cuối cùng
        if angle == 0:
            w, h = page.rect.width, page.rect.height
            curr_rot = page.rotation
            if curr_rot in (90, 270): w, h = h, w
            if w > h * 1.15: # Landscape
                angle = 270 # Mặc định xoay 90 độ CW

        if angle != 0:
            # Set rotation tuyệt đối trong PDF
            # page.rotation là góc hiện tại, angle là góc cần xoay để về 0
            new_rot = (page.rotation + angle) % 360
            page.set_rotation(new_rot)
            logger.info(f"  Page {i+1}: Orientation model suggests +{angle}° → set_rotation({new_rot})")
            rotated += 1

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    doc.save(dst, garbage=4, deflate=True)
    doc.close()
    return rotated


def process_folder(input_dir: str, output_dir: str, use_model: bool = True):
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    pdfs   = sorted(src.glob("*.pdf")) + sorted(src.glob("*.PDF"))
    others = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() != ".pdf"]

    logger.info(f"📂 Input : {src}  ({len(pdfs)} PDFs)")
    logger.info(f"📂 Output: {dst}")
    logger.info(f"🧠 Mode  : {'Tesseract OSD (Model)' if use_model else 'Heuristic Only'}")
    logger.info("─" * 60)

    total_rotated = 0
    for i, pdf in enumerate(pdfs, 1):
        out = dst / pdf.name
        logger.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            n = fix_one_pdf(str(pdf), str(out), use_model=use_model)
            total_rotated += n
            logger.info(f"  ✅ {n} page(s) fixed")
        except Exception as e:
            logger.error(f"  ❌ Error: {e} — copying original")
            shutil.copy2(pdf, out)

    for f in others:
        shutil.copy2(f, dst / f.name)

    logger.info("─" * 60)
    logger.info(f"✅ Preprocessing Done. {total_rotated} pages fixed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Folder PDF gốc")
    parser.add_argument("--output", required=True, help="Folder PDF đầu ra")
    parser.add_argument("--no-model", action="store_true", help="Tắt model OSD, chỉ dùng heuristic")
    args = parser.parse_args()

    process_folder(args.input, args.output, use_model=not args.no_model)
