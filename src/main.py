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
    Convert the image to grayscale and crop the region of interest.
    
    Crop Coordinates:
      x = 69, y = 125, width = 1402, height = 235
    If the image is too small for these coordinates, return None.
    """
    x, y, w, h = 69, 125, 1402, 235
    height, width = image.shape[:2]
    if width < x + w or height < y + h:
        print(f"Image dimensions ({width}x{height}) are smaller than the required crop region ({x+w}x{y+h}).")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optional thresholding for better OCR:
    # gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    cropped = gray[y:y + h, x:x + w]
    return cropped


def extract_text_from_image(image):
    """
    Run OCR on the processed image using Tesseract with custom config.
    """
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


def parse_text(text):
    """
    Because your OCR shows something like:

      (header line)
      Net Profit Total Closed Trades Percent Profitable Profit Factor Max Drawdown Avg Trade Avg # Bars in Trades

      (data line)
      8,431.45 USDT 8.43% 2,381 50.40% 1.02 26,194.08 USDT 25.53% 3.54 USDT 0.07% 16

    We need a line-by-line approach:
      1) Possibly parse strategy name from "WMA + SMI + Donchian Stop (Fresh)" line
      2) Parse the date range "2024-01-01 — 2025-01-01"
      3) Find the line that starts with digits (the data line) and use a single big regex to extract each field in order.

    The big regex pattern for the data line (example):
      ^\s*([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+([0-9,\.]+)\s+([0-9\.]+%)\s+([0-9\.]+)\s+([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+(\d+)

    That captures:
      group(1) -> 8,431.45
      group(2) -> 8.43%
      group(3) -> 2,381
      group(4) -> 50.40%
      group(5) -> 1.02
      group(6) -> 26,194.08
      group(7) -> 25.53%
      group(8) -> 3.54
      group(9) -> 0.07%
      group(10) -> 16

    But you only need:
      net_profit (only in %) -> group(2)
      total_closed_trades -> group(3)
      percent_profitable -> group(4)
      profit_factor -> group(5)
      max_drawdown (only in %) -> group(7)
      avg_trade (only in %) -> group(9)
      avg_bars_in_trade -> group(10)
    """

    # Debug: Print the raw OCR text
    print("DEBUG OCR TEXT:\n", text)

    data = {}

    # Split into lines (strip out empty lines)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1) Try to parse the "WMA + SMI + Donchian Stop (Fresh)" as the strategy name
    #    For example, if the line ends with "Deep Backtesting"
    #    We'll look for something containing "Donchian Stop" near "Deep Backtesting"
    #    If your text is always consistent, you can do simpler patterns.
    for line in lines:
        # Example: "WMA + SMI + Donchian Stop (Fresh) ©) @) Deep Backtesting"
        # We'll capture everything up to "Deep Backtesting"
        if "Deep Backtesting" in line:
            # A quick attempt: split by "Deep Backtesting" and keep left part
            left_part = line.split("Deep Backtesting")[0].strip()
            # Remove trailing ©) @) if needed
            left_part = re.sub(r"[©)@]+$", "", left_part).strip()
            data["strategy_name"] = left_part
            break

    # If we didn't find a strategy name above, default to "Unknown Strategy"
    if "strategy_name" not in data:
        data["strategy_name"] = "Unknown Strategy"

    # 2) Parse the date range "2024-01-01 — 2025-01-01"
    #    Some OCR may read the dash differently (— or -).
    for line in lines:
        m = re.search(r"(\d{4}-\d{2}-\d{2})\s*[-–—]+\s*(\d{4}-\d{2}-\d{2})", line)
        if m:
            data["test_period"] = {
                "start_date": m.group(1),
                "end_date": m.group(2)
            }
            break

    # If not found, set an empty date range
    if "test_period" not in data:
        data["test_period"] = {"start_date": "", "end_date": ""}

    # 3) Parse the main data line (which typically starts with a digit, e.g. "8,431.45 USDT 8.43% ...")
    #    We'll define a big regex for that line
    data_pattern = re.compile(
        r"^\s*([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+([0-9,\.]+)\s+([0-9\.]+%)\s+([0-9\.]+)\s+([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+([0-9,\.]+)\s*USDT\s*([0-9\.]+%)\s+(\d+)",
        re.IGNORECASE
    )

    for line in lines:
        match = data_pattern.search(line)
        if match:
            # net_profit_pct
            data["net_profit"] = match.group(2)  # e.g. "8.43%"
            # total_closed_trades
            data["total_closed_trades"] = int(match.group(3).replace(",", ""))  # e.g. "2,381" -> 2381
            # percent_profitable
            data["percent_profitable"] = match.group(4)  # e.g. "50.40%"
            # profit_factor
            try:
                data["profit_factor"] = float(match.group(5))
            except ValueError:
                data["profit_factor"] = 0.0
            # max_drawdown_pct
            data["max_drawdown"] = match.group(7)  # e.g. "25.53%"
            # avg_trade_pct
            data["avg_trade"] = match.group(9)  # e.g. "0.07%"
            # avg_bars_in_trade
            data["avg_bars_in_trade"] = int(match.group(10))  # e.g. "16"
            break  # Stop once we've parsed the data line

    return data


def extract_data():
    """
    1. Load all raw images from the RAW_IMAGES_FOLDER.
    2. Clear both the RAW_IMAGES_FOLDER and the processed images folder (IMAGES_FOLDER) to remove any previous files.
    3. For each uploaded image:
         - Process it (convert to grayscale and crop).
         - Save the processed image into IMAGES_FOLDER.
         - Run OCR and parse the data.
         - Extract the coin/chart name from the filename (e.g., "BTCUSDT").
    4. Assemble the parsed data into a JSON file and save it in the DATA_FOLDER.
    """
    # Get raw image files
    raw_image_files = [
        os.path.join(RAW_IMAGES_FOLDER, f)
        for f in os.listdir(RAW_IMAGES_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    if not raw_image_files:
        messagebox.showerror("Error", "No images found in the raw images folder.")
        return

    # Load raw images into memory
    original_images = {}
    for file_path in raw_image_files:
        image = cv2.imread(file_path)
        if image is not None:
            original_images[os.path.basename(file_path)] = image
        else:
            print(f"Error reading image: {file_path}")

    # Clear the raw images folder
    for file_path in raw_image_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

    # Clear the processed images folder (IMAGES_FOLDER)
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

    # Process each raw image from memory
    for filename, image in original_images.items():
        processed = preprocess_image_from_array(image)
        if processed is None or processed.size == 0:
            print(f"Skipping image {filename} due to invalid crop dimensions.")
            continue

        # Save the processed image into IMAGES_FOLDER
        processed_path = os.path.join(IMAGES_FOLDER, filename)
        if not cv2.imwrite(processed_path, processed):
            print(f"Failed to write processed image {processed_path}.")
            continue

        # OCR on the processed image
        ocr_text = extract_text_from_image(processed)
        parsed = parse_text(ocr_text)

        # For the first image, set overall strategy_name & test_period
        if first_image:
            strategy_info["strategy_name"] = parsed.get("strategy_name", "Unknown Strategy")
            strategy_info["test_period"] = parsed.get("test_period", {"start_date": "", "end_date": ""})
            first_image = False

        # Extract coin/chart name from the filename (e.g., "BTCUSDT")
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

    # Assemble the final JSON structure
    final_data = {
        "strategy_name": strategy_info.get("strategy_name", "Unknown Strategy"),
        "test_period": strategy_info.get("test_period", {"start_date": "", "end_date": ""}),
        "results": results
    }
    # Create a safe filename
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
