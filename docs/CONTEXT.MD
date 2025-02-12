TradingView Report Data Extraction - Technical Documentation

Overview

The TradingView Report Data Extraction App automates the process of reading strategy reports from TradingView screenshots. The app uses OCR (Optical Character Recognition) to extract critical trading metrics from strategy backtests

Tech Stack

OCR Engine: Tesseract OCR (Offline, Open-Source)

Preprocessing: OpenCV for image enhancement

tkinter for gui

Storage folder name data: JSON file for extracted results

Features

1. Image Processing & Preprocessing

Convert images to grayscale

Apply thresholding and noise reduction

Crop specific areas for targeted OCR processing (the dimensions is Crop Coordinates: x=69, y=125, width=1402, height=235)

2. OCR Text Extraction

Use Tesseract OCR to extract text from predefined sections

Parse relevant numerical data from reports

3. Data Parsing & Structuring

Apply regex-based extraction to capture:

Name strategy

Strategy Tested Date Range

Net Profit (only in percentage)

Total Closed Trades

Percent Profitable

Profit Factor

Max Drawdown (only in percentage)

Avg Trade (only in percentage)

Avg # Bars in Trades

Store extracted data in a structured format (JSON)

4. Execution

made tkinter gui for user to upload image
give it upload image button to upload the image (the image will be more than one like arround 10 images)
give it extract button to start the process
after press extract button, clear images folder

so the image i upload is one strategy but with different coins or chart
you can see the name coin on the image file name

1. Image Input

User uploads or selects a screenshot from a directory

2. Preprocessing

Convert image to grayscale

Apply thresholding and noise removal

Crop specific data sections for better OCR accuracy and save in image folder
(the dimensions is Crop Coordinates: x=69, y=125, width=1402, height=235)

3. OCR & Text Extraction

Run Tesseract OCR on processed sections by read every file on image folder

Extract text and apply regex to fetch numerical values

4. Data Formatting & Output

Convert extracted data into structured JSON format

save into data folder give it name as the strategy name

here are the template json example

{
"strategy_name": "WMA + SMI + Donchian Stop (Fresh)",
"test_period": {
"start_date": "2024-01-01",
"end_date": "2025-01-01"
},
"results": [
{
"chart": "BONKUSDT",
"net_profit": "-2.74%",
"total_closed_trades": 2373,
"percent_profitable": "51.50%",
"profit_factor": 0.998,
"max_drawdown": "58.24%",
"avg_trade": "0.19%",
"avg_bars_in_trade": 16
},
{
"chart": "BTCUSDT",
"net_profit": "9.91%",
"total_closed_trades": 10143,
"percent_profitable": "35.71%",
"profit_factor": 1.032,
"max_drawdown": "9.19%",
"avg_trade": "0.01%",
"avg_bars_in_trade": 34
},
{
"chart": "ETHUSDT",
"net_profit": "4.50%",
"total_closed_trades": 5321,
"percent_profitable": "42.90%",
"profit_factor": 1.150,
"max_drawdown": "8.50%",
"avg_trade": "0.05%",
"avg_bars_in_trade": 22
}
]
}
