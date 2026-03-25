"""
preprocess_rotate.py
─────────────────────────────────────────────────────────────────────
Tiền xử lý folder PDF/DOCX: tự động phát hiện và xoay về portrait.

Cách dùng trong Colab:
  !pip install pymupdf pytesseract pillow -q
  !apt-get install -y tesseract-ocr tesseract-ocr-vie -qq
  !python preprocess_rotate.py --input /content/docs --output /content/docs_fixed

Nếu không có Tesseract (dùng heuristic nhanh hơn):
  !python preprocess_rotate.py --input /content/docs --output /content/docs_fixed --no-ocr
─────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from loguru import logger

# ── Cài đặt thư viện cần thiết ──────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Cần cài PyMuPDF: pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Cần cài Pillow: pip install pillow")
    sys.exit(1)


def detect_rotation_tesseract(pil_img: Image.Image) -> int:
    """
    Dùng Tesseract OSD để phát hiện góc xoay.
    Trả về góc cần rotate (0, 90, 180, 270).
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(pil_img, config="--psm 0 -c min_characters_to_try=5")
        for line in osd.splitlines():
            if "Rotate:" in line:
                angle = int(line.split(":")[-1].strip())
                return angle
    except Exception as e:
        logger.warning(f"Tesseract OSD failed: {e} → dùng heuristic")
    return -1  # không phát hiện được


def detect_rotation_heuristic(pil_img: Image.Image) -> int:
    """
    Heuristic đơn giản: nếu ảnh nằm ngang (width > height * 1.3) → xoay 90° CW.
    Phù hợp cho tài liệu hành chính/pháp lý Việt Nam (luôn portrait).
    """
    w, h = pil_img.size
    if w > h * 1.3:
        return 90   # landscape → cần xoay 90° CW
    if h > w * 1.3:
        return 0    # đã là portrait
    return 0        # gần vuông → giữ nguyên


def fix_pdf_rotation(input_path: str, output_path: str, use_ocr: bool = True) -> dict:
    """
    Xử lý 1 file PDF: detect rotation từng trang → rotate → lưu ra output_path.
    Trả về dict thống kê.
    """
    doc = fitz.open(input_path)
    stats = {"pages": len(doc), "rotated": 0, "unchanged": 0}

    for page_num, page in enumerate(doc):
        # 1. Kiểm tra rotation trong PDF metadata trước
        meta_rot = page.rotation
        if meta_rot != 0:
            # Đã có rotation metadata → normalize về 0
            page.set_rotation(0)
            # Và rotate content ngược lại
            page.set_rotation(meta_rot)
            page.set_rotation(0)
            stats["rotated"] += 1
            logger.info(f"  Page {page_num+1}: PDF metadata rotation={meta_rot}° → fixed")
            continue

        # 2. Render trang thành ảnh để phân tích
        mat  = fitz.Matrix(1.5, 1.5)  # scale 1.5x để Tesseract chính xác hơn
        clip = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", [clip.width, clip.height], clip.samples)

        # 3. Detect rotation
        angle = -1
        if use_ocr:
            angle = detect_rotation_tesseract(pil_img)

        if angle == -1:  # fallback heuristic
            angle = detect_rotation_heuristic(pil_img)

        if angle in (90, 180, 270):
            # PyMuPDF: set_rotation dùng góc clockwise
            page.set_rotation(angle)
            stats["rotated"] += 1
            logger.info(f"  Page {page_num+1}: detected {angle}° rotation → corrected")
        else:
            stats["unchanged"] += 1

    # Lưu file đã sửa
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return stats


def process_folder(input_dir: str, output_dir: str, use_ocr: bool = True):
    """
    Xử lý toàn bộ folder: PDF → rotate → lưu vào output_dir.
    Non-PDF files được copy sang output_dir không thay đổi.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.iterdir())
    pdf_files   = [f for f in files if f.suffix.lower() == ".pdf"]
    other_files = [f for f in files if f.suffix.lower() != ".pdf" and f.is_file()]

    logger.info(f"Found {len(pdf_files)} PDFs + {len(other_files)} other files in {input_dir}")

    total_rotated = 0
    total_pages   = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        out_file = output_path / pdf_file.name
        logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        try:
            stats = fix_pdf_rotation(str(pdf_file), str(out_file), use_ocr=use_ocr)
            total_pages   += stats["pages"]
            total_rotated += stats["rotated"]
            logger.info(
                f"  ✅ Done: {stats['pages']} pages, "
                f"{stats['rotated']} rotated, {stats['unchanged']} unchanged"
            )
        except Exception as e:
            logger.error(f"  ❌ Failed: {e} — copying original")
            shutil.copy2(str(pdf_file), str(out_file))

    # Copy file không phải PDF nguyên vẹn
    for f in other_files:
        shutil.copy2(str(f), str(output_path / f.name))
        logger.info(f"Copied (non-PDF): {f.name}")

    logger.info("─" * 60)
    logger.info(f"✅ DONE: {len(pdf_files)} PDFs processed")
    logger.info(f"   Total pages: {total_pages}")
    logger.info(f"   Pages rotated/fixed: {total_rotated}")
    logger.info(f"   Output folder: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-rotate PDF preprocessing tool")
    parser.add_argument("--input",  required=True, help="Folder chứa PDF gốc")
    parser.add_argument("--output", required=True, help="Folder lưu PDF đã sửa")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Dùng heuristic thay vì Tesseract OSD (nhanh hơn)")
    args = parser.parse_args()

    use_ocr = not args.no_ocr
    if use_ocr:
        logger.info("Mode: Tesseract OSD + heuristic fallback")
    else:
        logger.info("Mode: Heuristic only (nhanh, không cần Tesseract)")

    process_folder(args.input, args.output, use_ocr=use_ocr)
