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
    Crop the region of interest from the image without converting it to grayscale.

    Crop Coordinates:
      x = 69, y = 125, width = 1402, height = 235
    If the image is too small for these coordinates, return None.
    """
    x, y, w, h = 69, 125, 1402, 235
    height, width = image.shape[:2]
    if width < x + w or height < y + h:
        print(f"Image dimensions ({width}x{height}) are smaller than the required crop region ({x+w}x{y+h}).")
        return None

    cropped = image[y:y + h, x:x + w]
    return cropped


def extract_text_from_image(image):
    """
    Run OCR on the processed image using Tesseract with a custom config.
    """
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


def parse_text(text):
    r"""
    Parses the OCR text to extract report fields.

    Expected OCR output (example):

      WMA + SMI + Donchian Stop (Fresh) ©) @) Deep Backtesting
      Be Y
      2024-01-01 — 2025-01-01 (cenerate report}
      Overview Performance Summary _List of Trades _ Properties
      Net Profit Total Closed Trades Percent Profitable Profit Factor Max Drawdown Avg Trade Avg # Bars in Trades
      -44,993.00 USDT -44.99% 2,498 49.40% 0.892 61,514.63 USDT 58.10%  -18.01 USDT 0.04% 15

    The regex below is designed to capture from the data line:
      Group 1: Net Profit (USDT amount) – ignored
      Group 2: Net Profit (percentage)
      Group 3: Total Closed Trades
      Group 4: Percent Profitable
      Group 5: Profit Factor
      Group 6: Max Drawdown (USDT amount) – ignored
      Group 7: Max Drawdown (percentage)
      Group 8: Avg Trade (USDT amount) – ignored
      Group 9: Avg Trade (percentage)
      Group 10: Avg Bars in Trade

    Note: The pattern now allows for negative values (using -?).
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

    # 3) Define the regex to capture the numeric fields (allowing for negative values)
    data_pattern = re.compile(
        r"^\s*(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+(-?[0-9,\.]+)\s+(-?[0-9\.]+%)\s+(-?[0-9\.]+)\s+(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+(-?[0-9,\.]+)\s*USDT\s*(-?[0-9\.]+%)\s+(\d+)",
        re.IGNORECASE
    )

    # Look for a line that matches the data pattern
    for line in lines:
        match = data_pattern.search(line)
        if match:
            # Assign the extracted fields
            data["net_profit"] = match.group(2)  # Net Profit percentage
            try:
                data["total_closed_trades"] = int(match.group(3).replace(",", ""))
            except ValueError:
                data["total_closed_trades"] = 0
            data["percent_profitable"] = match.group(4)
            try:
                data["profit_factor"] = float(match.group(5))
            except ValueError:
                data["profit_factor"] = 0.0
            data["max_drawdown"] = match.group(7)
            data["avg_trade"] = match.group(9)
            try:
                data["avg_bars_in_trade"] = int(match.group(10))
            except ValueError:
                data["avg_bars_in_trade"] = 0
            break

    return data


def extract_data():
    """
    1. Load all raw images from the RAW_IMAGES_FOLDER.
    2. Clear both the RAW_IMAGES_FOLDER and the processed images folder (IMAGES_FOLDER) to remove any previous files.
    3. For each uploaded image:
         - Process it (crop the region of interest).
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

    original_images = {}
    for file_path in raw_image_files:
        image = cv2.imread(file_path)
        if image is not None:
            original_images[os.path.basename(file_path)] = image
        else:
            print(f"Error reading image: {file_path}")

    for file_path in raw_image_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

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

        ocr_text = extract_text_from_image(processed)
        parsed = parse_text(ocr_text)

        if first_image:
            strategy_info["strategy_name"] = parsed.get("strategy_name", "Unknown Strategy")
            strategy_info["test_period"] = parsed.get("test_period", {"start_date": "", "end_date": ""})
            first_image = False

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
