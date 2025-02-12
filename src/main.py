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
# Processed images folder (for cropped/grayscale images)
IMAGES_FOLDER = os.path.join(BASE_DIR, "images")
# Raw images folder (for uploaded images)
RAW_IMAGES_FOLDER = os.path.join(BASE_DIR, "raw_image")
# JSON output folder
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
    cropped = gray[y:y + h, x:x + w]
    return cropped


def extract_text_from_image(image):
    """
    Run OCR on the processed image using Tesseract.
    """
    text = pytesseract.image_to_string(image)
    return text


def parse_text(text):
    """
    Parse the OCR text using regex to extract report fields.
    
    Expected fields:
      - Strategy Name
      - Test Period (start and end date)
      - Net Profit (percentage)
      - Total Closed Trades (integer)
      - Percent Profitable (percentage)
      - Profit Factor (float)
      - Max Drawdown (percentage)
      - Avg Trade (percentage)
      - Avg Bars in Trade (integer)
    """
    data = {}

    strategy_match = re.search(r"Strategy(?: Name)?:\s*(.+)", text, re.IGNORECASE)
    if strategy_match:
        data["strategy_name"] = strategy_match.group(1).strip()

    test_period_match = re.search(
        r"Test Period:\s*(\d{4}-\d{2}-\d{2})\s*[-–]\s*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE
    )
    if test_period_match:
        data["test_period"] = {
            "start_date": test_period_match.group(1),
            "end_date": test_period_match.group(2)
        }

    net_profit_match = re.search(r"Net Profit[:\s]+([-+]?\d+(?:\.\d+)?%)", text, re.IGNORECASE)
    if net_profit_match:
        data["net_profit"] = net_profit_match.group(1).strip()

    closed_trades_match = re.search(r"Total Closed Trades[:\s]+(\d+)", text, re.IGNORECASE)
    if closed_trades_match:
        data["total_closed_trades"] = int(closed_trades_match.group(1).strip())

    percent_profitable_match = re.search(r"Percent Profitable[:\s]+([-+]?\d+(?:\.\d+)?%)", text, re.IGNORECASE)
    if percent_profitable_match:
        data["percent_profitable"] = percent_profitable_match.group(1).strip()

    profit_factor_match = re.search(r"Profit Factor[:\s]+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if profit_factor_match:
        try:
            data["profit_factor"] = float(profit_factor_match.group(1).strip())
        except ValueError:
            data["profit_factor"] = 0.0

    max_drawdown_match = re.search(r"Max Drawdown[:\s]+([-+]?\d+(?:\.\d+)?%)", text, re.IGNORECASE)
    if max_drawdown_match:
        data["max_drawdown"] = max_drawdown_match.group(1).strip()

    avg_trade_match = re.search(r"Avg Trade[:\s]+([-+]?\d+(?:\.\d+)?%)", text, re.IGNORECASE)
    if avg_trade_match:
        data["avg_trade"] = avg_trade_match.group(1).strip()

    avg_bars_match = re.search(r"Avg Bars in Trade[:\s]+(\d+)", text, re.IGNORECASE)
    if avg_bars_match:
        data["avg_bars_in_trade"] = int(avg_bars_match.group(1).strip())

    return data


def extract_data():
    """
    1. Load all raw images from the RAW_IMAGES_FOLDER.
    2. Clear both the RAW_IMAGES_FOLDER and the processed images folder (IMAGES_FOLDER) to remove any previous files.
    3. For each uploaded image:
         - Process it (convert to grayscale and crop).
         - Save the processed image into IMAGES_FOLDER.
         - Run OCR and parse the data.
         - Extract the coin/chart name from the filename.
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

        if first_image:
            strategy_info["strategy_name"] = parsed.get("strategy_name", "Unknown Strategy")
            strategy_info["test_period"] = parsed.get("test_period", {"start_date": "", "end_date": ""})
            first_image = False

        # Extract coin/chart name from the filename (e.g., "BTCUSDT")
        coin_match = re.search(r"([A-Z0-9]+USDT)", filename)
        if coin_match:
            chart_name = coin_match.group(1)
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
    safe_strategy_name = re.sub(r'[\\/*?:"<>|]', "", final_data["strategy_name"])
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
