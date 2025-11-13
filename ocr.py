import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Alias for easier exception handling
TesseractError = pytesseract.TesseractError

# Try to use tqdm for a nice progress bar (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ---------------------------------------
# Tesseract setup (edit path if needed)
# ---------------------------------------
# If Tesseract is not in your PATH on Windows, uncomment and set the correct path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default config values (can be overridden by CLI args)
DEFAULT_LANG = "jpn"              # change via --lang (e.g. jpn, kor, eng+jpn)
DEFAULT_MIN_CONFIDENCE = 30.0     # was 60.0 -> more tolerant for manga
DEFAULT_SUFFIX = "_clean"
DEFAULT_PSM = 6                   # default page segmentation mode

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# Minimum text box size to avoid tiny noisy boxes
MIN_BOX_WIDTH = 5
MIN_BOX_HEIGHT = 5


def build_text_mask(
    image: np.ndarray,
    lang: str = DEFAULT_LANG,
    min_conf: float = DEFAULT_MIN_CONFIDENCE,
    psm: int = DEFAULT_PSM,
) -> np.ndarray:
    """
    Build a binary mask around detected text regions using Tesseract.
    Returns a single-channel mask (uint8) with white = text regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight denoising and normalization to help Tesseract
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.equalizeHist(gray)

    try:
        # Tesseract OCR: get bounding boxes for each text element
        data = pytesseract.image_to_data(
            gray,
            lang=lang,
            config=(
                f"--oem 3 "          # Default engine
                f"--psm {psm} "      # Page segmentation mode
                "-c preserve_interword_spaces=1"
            ),
            output_type=Output.DICT,
        )
    except TesseractError as e:
        raise RuntimeError(f"Tesseract failed: {e}") from e

    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    n_boxes = len(data["level"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf_str = data["conf"][i]

        if not text:
            continue

        # Parse confidence and filter low confidence boxes
        try:
            conf = float(conf_str)
        except ValueError:
            continue

        # Tesseract uses -1 for "no confidence"; skip those
        if conf <= 0:
            continue

        if conf < min_conf:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w_box = int(data["width"][i])
        h_box = int(data["height"][i])

        # Skip very tiny boxes (usually noise)
        if w_box < MIN_BOX_WIDTH or h_box < MIN_BOX_HEIGHT:
            continue

        # Add some padding around the detected text box
        pad_x = int(w_box * 0.15)
        pad_y = int(h_box * 0.15)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w_box + pad_x, w)
        y2 = min(y + h_box + pad_y, h)

        mask[y1:y2, x1:x2] = 255

    # Slightly dilate & close the mask to merge nearby regions and cover edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def clean_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply inpainting + light smoothing to text regions defined by the mask.
    This tries to "whiten" (clean) the text area using the background.
    """
    # Inpaint the text regions using surrounding pixels (similar to background fill)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Light smoothing to blend the repaired regions
    smoothed = cv2.GaussianBlur(inpainted, (3, 3), 0)

    # Only apply smoothing inside masked regions to avoid blurring the whole page
    result = inpainted.copy()
    result[mask > 0] = smoothed[mask > 0]

    return result


def process_image_file(
    img_path: Path,
    lang: str,
    min_conf: float,
    psm: int,
    inplace: bool,
    suffix: str,
    verbose: bool = True,
) -> bool:
    """
    Process a single image:
    - build text mask
    - inpaint + smooth
    - save cleaned image

    Returns True if a cleaned image was saved, False otherwise.
    """
    if verbose:
        print(f"[+] Processing: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        if verbose:
            print("    [!] Failed to read image, skipping.")
        return False

    try:
        mask = build_text_mask(image, lang=lang, min_conf=min_conf, psm=psm)
    except RuntimeError as e:
        if verbose:
            print(f"    [!] OCR error on {img_path.name}: {e}")
        return False

    if cv2.countNonZero(mask) == 0:
        if verbose:
            print("    [i] No text detected (mask is empty), skipping.")
        return False

    cleaned = clean_image(image, mask)

    if inplace:
        out_path = img_path
    else:
        out_path = img_path.with_name(img_path.stem + suffix + img_path.suffix)

    cv2.imwrite(str(out_path), cleaned)

    if verbose:
        print(f"    [✓] Saved: {out_path.name}")

    return True


def collect_images(folder_path: Path, recursive: bool):
    """
    Collect all image files from a folder.
    If recursive is True, also search subfolders.
    """
    if recursive:
        glob_pattern = "**/*"
    else:
        glob_pattern = "*"

    images = [
        p for p in folder_path.glob(glob_pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    images.sort()
    return images


def process_folder(
    folder_path: Path,
    lang: str,
    min_conf: float,
    psm: int,
    inplace: bool,
    suffix: str,
    recursive: bool,
):
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"[!] Folder not found or not a directory: {folder_path}")
        sys.exit(1)

    images = collect_images(folder_path, recursive=recursive)

    if not images:
        print("[!] No images found in the folder.")
        return

    print(f"[i] Found {len(images)} image(s) in {folder_path}")
    print(f"[i] Language: {lang}, Min confidence: {min_conf}, PSM: {psm}, In-place: {inplace}")

    cleaned_count = 0

    if TQDM_AVAILABLE:
        iterator = tqdm(images, desc="Cleaning images", unit="img")
    else:
        iterator = images

    for img_path in iterator:
        saved = process_image_file(
            img_path,
            lang=lang,
            min_conf=min_conf,
            psm=psm,
            inplace=inplace,
            suffix=suffix,
            verbose=not TQDM_AVAILABLE,
        )
        if saved:
            cleaned_count += 1

    print(f"[i] Done. Cleaned {cleaned_count}/{len(images)} image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple manga/scan cleaner using Tesseract OCR masks + OpenCV inpainting. "
            "It detects text, builds a mask around it, and fills it with background color."
        )
    )

    parser.add_argument(
        "folder",
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help="Tesseract language code(s), e.g. 'eng', 'jpn', 'kor', 'eng+jpn'. Default: %(default)s",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum OCR confidence to accept a text box. Default: %(default)s",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=DEFAULT_PSM,
        help="Tesseract page segmentation mode (0–13, usually 6 or 7 works for manga). Default: %(default)s",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original images instead of creating *_clean copies.",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help="Suffix for cleaned files (ignored if --inplace). Default: '%(default)s'",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images in subfolders recursively.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    folder_path = Path(args.folder)

    print(f"[i] Input folder: {folder_path.resolve()}")

    process_folder(
        folder_path=folder_path,
        lang=args.lang,
        min_conf=args.min_conf,
        psm=args.psm,
        inplace=args.inplace,
        suffix=args.suffix,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
