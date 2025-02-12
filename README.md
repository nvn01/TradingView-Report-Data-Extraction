# TradingView Report Data Extraction

This project is a **TradingView Report Data Extraction** tool that allows users to upload trading strategy reports as images, process them using **OCR (Tesseract)**, extract relevant trading data, and save the extracted data in JSON format.

## Features

- 📂 **Upload multiple images** containing TradingView strategy reports.
- 🎨 **Preprocess images** by converting them to grayscale and cropping the relevant section.
- 🔍 **Extract text using OCR** powered by Tesseract.
- 📊 **Parse trading metrics** from the extracted text.
- 💾 **Store extracted data in JSON** format for further analysis.

## Project Structure

```
project-root/
│── docs/
│   └── CONTEXT.MD
│── src/
│   └── main.py
│── poetry.lock
│── pyproject.toml
│── README.md
│── raw_image
│── data
│── images
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies Using Poetry

```bash
poetry install
```

### 3. Ensure Tesseract-OCR is Installed

- **Windows:** [Download here](http://github.com/tesseract-ocr/tesseract/releases/tag/5.5.0)
- **Linux (Ubuntu/Debian):**

```bash
sudo apt install tesseract-ocr
```

- **macOS:**

```bash
brew install tesseract
```

### 4. Run the Application

```bash
poetry run python src/main.py
```

## Usage

### 1. Launch the GUI

- Click the **"Upload Images"** button and select images of TradingView reports.
- Click **"Extract Data"** to process the images and generate structured trading data in JSON format.

### 2. Supported Trading Metrics

- Strategy Name
- Test Period (Start & End Date)
- Net Profit (%)
- Total Closed Trades
- Percent Profitable (%)
- Profit Factor
- Max Drawdown (%)
- Avg Trade (%)
- Avg Bars in Trade

### 3. Output Data Structure (JSON Example)

```json
{
  "strategy_name": "RSI Divergence",
  "test_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-02-01"
  },
  "results": [
    {
      "chart": "BTCUSDT",
      "net_profit": "+15.2%",
      "total_closed_trades": 35,
      "percent_profitable": "60%",
      "profit_factor": 1.8,
      "max_drawdown": "-5.3%",
      "avg_trade": "+0.5%",
      "avg_bars_in_trade": 10
    }
  ]
}
```

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- Tesseract-OCR (`pytesseract`)
- Tkinter (GUI)
- Poetry (for dependency management)

## License

This project is licensed under the **MIT License**.

---
