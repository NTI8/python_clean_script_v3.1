# python_clean_script_v3.1
````markdown
# Manga OCR Cleaner (Tesseract + OpenCV)

Small Python tool to auto-clean text from manga / scanned pages using Tesseract OCR masks + OpenCV inpainting.

---

## 1. Requirements

- **OS**: Windows 10/11  
- **Python**: 3.9+  
- **Python packages** (install once):

  ```bash
  pip install opencv-python numpy pytesseract tqdm
````

* **Tesseract OCR for Windows**:

  * Install from the official Windows installer (UB Mannheim build is recommended).

  * Default install path (used by this project):

    ```text
    C:\Program Files\Tesseract-OCR\
    ```

  * Make sure `tesseract.exe` is in your `PATH`, **or** set this line in `ocr.py`:

    ```python
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

---

## 2. Language data (`.traineddata` files)

All language files must be placed in:

```text
C:\Program Files\Tesseract-OCR\tessdata\
```

* Already included by default:

  * `eng.traineddata` (English)
  * `osd.traineddata` (orientation/script detection)

* For **Japanese** support (recommended for manga):

  * Download from `tessdata_best` (raw URLs) and save to `tessdata`:

    * `jpn.traineddata`
    * `jpn_vert.traineddata`
  * After copying, check:

    ```bash
    tesseract --list-langs
    ```

    You should see at least:

    ```text
    eng
    jpn
    jpn_vert
    osd
    ```

---

## 3. Files in this project

* `ocr.py` – main script:

  * Scans a folder of images
  * Runs Tesseract to detect text
  * Builds a mask around text
  * Uses OpenCV inpainting to fill text with background
  * Optionally smooths masked regions
  * Saves cleaned images

Supported image types: `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`.

---

## 4. Basic usage

From a terminal (PowerShell / CMD), go to the script folder, for example:

```bash
cd "C:\Users\abdoh\Downloads\testScript"
```

### English only

```bash
python ocr.py "C:\Users\abdoh\Downloads\11C"
```

### Japanese (horizontal manga)

```bash
python ocr.py "C:\Users\abdoh\Downloads\11C" --lang jpn --min-conf 20 --psm 6
```

### Japanese vertical text (if needed)

```bash
python ocr.py "C:\Users\abdoh\Downloads\11C" --lang jpn_vert --min-conf 20 --psm 6
```

---

## 5. Useful options (CLI flags)

All flags are **optional**:

* `--lang`
  Tesseract language codes. Examples:

  * `--lang eng`
  * `--lang jpn`
  * `--lang eng+jpn`

* `--min-conf`
  Minimum OCR confidence to accept a text box (default: `30.0`).
  Lower values = more boxes, higher = stricter.

  ```bash
  --min-conf 20
  ```

* `--psm`
  Page segmentation mode (Tesseract). Common values:

  * `6` – assume a block of text (default)
  * `7` – single text line
  * `11` – sparse text

  Example:

  ```bash
  --psm 6
  ```

* `--inplace`
  Overwrite original images **instead of** creating `*_clean` copies:

  ```bash
  --inplace
  ```

* `--suffix`
  Suffix for cleaned files (ignored if `--inplace` is used). Default: `_clean`.

  ```bash
  --suffix "_clean"
  ```

* `--recursive`
  Process all images in subfolders as well:

  ```bash
  --recursive
  ```

---

## 6. What the script does (summary)

For each image in the target folder:

1. Convert to grayscale and lightly denoise.
2. Run Tesseract to detect text boxes.
3. Build a binary mask around text (with small padding).
4. Dilate/merge mask regions to cover full bubbles/letters.
5. Use OpenCV **inpaint** to fill masked areas from surrounding pixels (background).
6. Apply a light blur only inside the masked zones.
7. Save the cleaned image (either as `<name>_clean.ext` or overwriting original if `--inplace`).

That’s it – drop your manga pages in a folder, run the command, and you get cleaned pages ready for typesetting.

```
::contentReference[oaicite:0]{index=0}
```
