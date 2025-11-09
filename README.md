Multi-Stock GRU Prediction Demo (TensorFlow.js)

This project demonstrates a fully client-side stock movement prediction model built with TensorFlow.js, running entirely in the browser and deployable via GitHub Pages.
It allows users to upload a CSV file with market data for multiple S&P 500 stocks, train a GRU-based model, and visualize per-stock prediction accuracy interactively.

🧩 Project Overview

The app reads local CSV market data with columns:
Date, Symbol, Open, Close — for 10 S&P 500 stocks across daily observations.

It then:

Normalizes features per stock (Open, Close) using Min–Max scaling.

Builds time-series sequences:

Input: 12-day sliding window of [Open, Close] for all 10 stocks → shape (12, 20).

Output: 3-day-ahead binary movement labels (1 = up, 0 = down) for each stock → 10 × 3 = 30 outputs.

Trains a stacked GRU model in-browser using WebGL acceleration.

Evaluates performance per stock, ranks them by accuracy, and visualizes correct/wrong predictions.

🧠 Model Architecture
Input & Output

Input shape: (12, 20)
→ 12 days × (10 stocks × 2 features)

Output shape: (30)
→ (10 stocks × 3 forecast days)

Network Layers
