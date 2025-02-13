import os
import re
import json
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import pytesseract

# Define project directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_FOLDER = os.path.join(BASE_DIR, "images")
RAW_IMAGES_FOLDER = os.path.join(BASE_DIR, "raw_image")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

# Ensure the folders exist
for folder in [IMAGES_FOLDER, RAW_IMAGES_FOLDER, DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def preprocess_image_from_array(image):
    """
    Crop the region of interest from the image.

    Crop Coordinates:
      x = 69, y = 125, width = 1402, height = 235
    If the image is too small for these coordinates, return None.

    NOTE: We do NOT convert to grayscale so that color is preserved.
    """
    x, y, w, h = 69, 125, 1402, 235
    height, width = image.shape[:2]
    if width < x + w or height < y + h:
        print(f"Image dimensions ({width}x{height}) are smaller than the required crop region ({x+w}x{y+h}).")
        return None

    # Keep the original color
    cropped = image[y:y + h, x:x + w]
    return cropped


def extract_text_and_data_from_image(image):
    """
    Run OCR on the processed image using Tesseract.
    Returns both the plain text and the word-level OCR data dictionary (unused in parse_text, but kept for reference).
    """
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return text, ocr_data


def parse_text(text, ocr_data, processed):
    """
    Parses the OCR text to extract report fields.

    We also force the net profit percentage to match the sign of the USDT net profit.
    If the net profit in USDT is negative, we prepend '-' to the net profit percentage if missing.
    """
    print("DEBUG OCR TEXT:\n", text)
    data = {}

    # Split OCR text into non-empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1) Try to capture the strategy name from a line containing "Deep Backtesting"
    for line in lines:
        if "Deep Backtesting" in line:
            left_part = line.split("Deep Backtesting")[0].strip()
            left_part = re.sub(r"[©)@]+$", "", left_part).strip()
            data["strategy_name"] = left_part
            break
    if "strategy_name" not in data:
        data["strategy_name"] = "Unknown Strategy"

    # 2) Parse the date range (e.g. "2024-01-01 — 2025-01-01")
    for line in lines:
        m = re.search(r"(\d{4}-\d{2}-\d{2})\s*[-–—]+\s*(\d{4}-\d{2}-\d{2})", line)
        if m:
            data["test_period"] = {"start_date": m.group(1), "end_date": m.group(2)}
            break
    if "test_period" not in data:
        data["test_period"] = {"start_date": "", "end_date": ""}

    # 3) Regex to capture the numeric fields (allowing for negative values)
    data_pattern = re.compile(
        r"^\s*(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+"
        r"(-?[0-9,\.]+)\s+(-?[0-9\.]+%)\s+"
        r"(-?[0-9\.]+)\s+"
        r"(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+"
        r"(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+(\d+)",
        re.IGNORECASE
    )

    # Look for a line that matches the data pattern
    for line in lines:
        match = data_pattern.search(line)
        if match:
            # Group(1) => Net Profit (USDT)    e.g. "-41,101.63"
            # Group(2) => Net Profit (%)       e.g. "41.10%"
            net_profit_usdt_str = match.group(1).strip()
            net_profit_pct_str = match.group(2).strip()

            # If the USDT portion is negative but the percentage is not, prepend minus sign
            if net_profit_usdt_str.startswith('-') and not net_profit_pct_str.startswith('-'):
                net_profit_pct_str = '-' + net_profit_pct_str

            data["net_profit"] = net_profit_pct_str

            # Group(3) => total closed trades
            try:
                data["total_closed_trades"] = int(match.group(3).replace(",", ""))
            except ValueError:
                data["total_closed_trades"] = 0

            # Group(4) => percent profitable
            data["percent_profitable"] = match.group(4)

            # Group(5) => profit factor
            try:
                data["profit_factor"] = float(match.group(5))
            except ValueError:
                data["profit_factor"] = 0.0

            # Group(7) => max drawdown (percentage)
            data["max_drawdown"] = match.group(7)

            # Group(9) => avg trade (percentage)
            data["avg_trade"] = match.group(9)

            # Group(10) => avg bars in trade
            try:
                data["avg_bars_in_trade"] = int(match.group(10))
            except ValueError:
                data["avg_bars_in_trade"] = 0

            break  # Stop after finding the first matching line

    return data


def extract_data():
    """
    1. Load all raw images from the RAW_IMAGES_FOLDER.
    2. Clear both the RAW_IMAGES_FOLDER and the processed images folder (IMAGES_FOLDER).
    3. For each uploaded image:
         - Crop it (keep color).
         - Save the processed image into IMAGES_FOLDER.
         - Run OCR and parse the data.
         - Extract the coin/chart name from the filename (e.g., "BTCUSDT").
    4. Assemble the parsed data into a JSON file and save it in the DATA_FOLDER.
    """
    raw_image_files = [
        os.path.join(RAW_IMAGES_FOLDER, f)
        for f in os.listdir(RAW_IMAGES_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    if not raw_image_files:
        messagebox.showerror("Error", "No images found in the raw images folder.")
        return

    # Read images into memory
    original_images = {}
    for file_path in raw_image_files:
        image = cv2.imread(file_path)
        if image is not None:
            original_images[os.path.basename(file_path)] = image
        else:
            print(f"Error reading image: {file_path}")

    # Clear out raw images folder
    for file_path in raw_image_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

    # Clear out processed images folder
    processed_files = [
        os.path.join(IMAGES_FOLDER, f)
        for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    for file_path in processed_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

    results = []
    strategy_info = {}
    first_image = True

    for filename, image in original_images.items():
        processed = preprocess_image_from_array(image)
        if processed is None or processed.size == 0:
            print(f"Skipping image {filename} due to invalid crop dimensions.")
            continue

        processed_path = os.path.join(IMAGES_FOLDER, filename)
        if not cv2.imwrite(processed_path, processed):
            print(f"Failed to write processed image {processed_path}.")
            continue

        ocr_text, ocr_data = extract_text_and_data_from_image(processed)
        parsed = parse_text(ocr_text, ocr_data, processed)

        # Store strategy info from the first valid image
        if first_image:
            strategy_info["strategy_name"] = parsed.get("strategy_name", "Unknown Strategy")
            strategy_info["test_period"] = parsed.get("test_period", {"start_date": "", "end_date": ""})
            first_image = False

        # Extract the coin/chart name from the filename (e.g. "BTCUSDT")
        coin_match = re.search(r"([A-Z0-9]+USDT)", filename, re.IGNORECASE)
        if coin_match:
            chart_name = coin_match.group(1).upper()
        else:
            chart_name = os.path.splitext(filename)[0]

        result = {
            "chart": chart_name,
            "net_profit": parsed.get("net_profit", ""),
            "total_closed_trades": parsed.get("total_closed_trades", 0),
            "percent_profitable": parsed.get("percent_profitable", ""),
            "profit_factor": parsed.get("profit_factor", 0.0),
            "max_drawdown": parsed.get("max_drawdown", ""),
            "avg_trade": parsed.get("avg_trade", ""),
            "avg_bars_in_trade": parsed.get("avg_bars_in_trade", 0)
        }
        results.append(result)

    final_data = {
        "strategy_name": strategy_info.get("strategy_name", "Unknown Strategy"),
        "test_period": strategy_info.get("test_period", {"start_date": "", "end_date": ""}),
        "results": results
    }

    # Build a safe file name for JSON output
    safe_strategy_name = re.sub(r'[\\/*?:"<>|]', "", final_data["strategy_name"])
    if not safe_strategy_name.strip():
        safe_strategy_name = "Unknown_Strategy"
    output_filename = f"{safe_strategy_name.replace(' ', '_')}.json"
    output_path = os.path.join(DATA_FOLDER, output_filename)

    try:
        with open(output_path, "w") as outfile:
            json.dump(final_data, outfile, indent=4)
        messagebox.showinfo("Success", f"Data extraction complete.\nJSON saved to:\n{output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to write JSON file:\n{e}")


def upload_images():
    """
    Open a file dialog for the user to select one or more images.
    The selected images are copied into the RAW_IMAGES_FOLDER.
    """
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_paths:
        return

    for file_path in file_paths:
        try:
            shutil.copy(file_path, RAW_IMAGES_FOLDER)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy {file_path}:\n{e}")
            return

    messagebox.showinfo("Images Uploaded", f"{len(file_paths)} image(s) uploaded successfully.")


def create_gui():
    """
    Create and run the tkinter GUI with an Upload Images button and an Extract Data button.
    """
    root = tk.Tk()
    root.title("TradingView Report Data Extraction")
    root.geometry("400x200")
    root.resizable(False, False)

    upload_button = tk.Button(root, text="Upload Images", width=20, height=2, command=upload_images)
    upload_button.pack(pady=20)

    extract_button = tk.Button(root, text="Extract Data", width=20, height=2, command=extract_data)
    extract_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
