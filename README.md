# 📈 Multi-Stock GRU Prediction Demo (TensorFlow.js)

This project demonstrates a **fully client-side stock prediction model** built with **TensorFlow.js**.  
It runs entirely in the browser and can be deployed via **GitHub Pages**, requiring no backend or server.

Users can upload a CSV file containing market data for multiple S&P 500 stocks, train a GRU-based model, and visualize prediction accuracy interactively.

---

## 🧩 Overview

The app reads a local CSV file with columns:

for 10 S&P 500 stocks across daily observations.

### Pipeline Summary:
1. **Normalization** – Each stock’s `Open` and `Close` values are scaled using Min–Max normalization.  
2. **Sequence Preparation** – A 12-day sliding window is built for each stock:
   - **Input:** Last 12 days × (10 stocks × 2 features) → shape `(12, 20)`
   - **Output:** 3-day-ahead binary “up/down” labels → shape `(30)`
3. **Model Training** – A stacked GRU model learns these relationships directly in the browser using WebGL.
4. **Evaluation & Visualization** – Displays accuracy ranking and per-stock prediction timelines.

---

## 🧠 Model Architecture

| Layer | Type | Parameters | Purpose |
|-------|------|-------------|----------|
| 1 | GRU | units=48, returnSequences=True, dropout=0.1 | Capture sequential price dynamics |
| 2 | GRU | units=24, returnSequences=False, dropout=0.1 | Encode time-based dependencies |
| 3 | Dense | units=30, activation='sigmoid' | Predict up/down for 10 stocks × 3 days |

### Model Specs
- **Input shape:** `(12, 20)`  
- **Output shape:** `(30)`  
- **Loss:** Binary Crossentropy  
- **Metric:** Binary Accuracy  
- **Optimizer:** Adam (`learningRate = 0.0015`)

---

## 🏋️ Training Process

The model trains **entirely in the browser** using TensorFlow.js with GPU acceleration (WebGL).  
Training progress (epoch, loss, accuracy, validation results) is displayed in real time.

### Features:
- **Adaptive Early Stopping:**  
  - Training stops automatically when validation accuracy no longer improves for several epochs.  
  - Ensures efficient convergence without overfitting.  
  - The model won’t always run all epochs — it stops early when good performance is reached.
- **Batch size:** 64 (auto-adjusts if memory is limited)  
- **Validation split:** 20% of the dataset

---

## 🔮 Prediction & Evaluation

After training:
- The model predicts 3-day-ahead binary movements (up/down) for all stocks in the test set.
- For each stock:
  - Accuracy is computed over all predictions and forecast horizons.
  - The decision threshold (sigmoid output cutoff) is **automatically optimized** per output from 0.2–0.8 to maximize accuracy.

### Visualization Components
- **Stock Accuracy Ranking:**  
  Sorted horizontal bar chart of accuracy per stock.
- **Prediction Timeline:**  
  Line chart showing correct (blue) vs. wrong (red) predictions over time.

---

## 📊 Visualization

Interactive plots built with **Chart.js**:
- Real-time progress bar for training.
- Accuracy ranking chart.
- Correct/wrong prediction timeline for the best-performing stock.

All graphs update dynamically when training or predictions are complete.

---

## ⚙️ Deployment

This app is **100% client-side**.  
No data leaves the user’s browser.

### To run locally:
1. Open `index.html` in your browser.  
2. Upload your CSV file.  
3. Click **Train Model**, then **Run Prediction**.

### To deploy on GitHub Pages:
1. Push this repository to GitHub.
2. Enable **Pages** in the repository settings (root directory).
3. Visit your GitHub Pages URL — the app will run directly in the browser.

---

## 🧾 Summary

| Component | Description |
|------------|--------------|
| Framework | TensorFlow.js |
| Architecture | 2-layer GRU → Dense(30, sigmoid) |
| Input | 12 days × 10 stocks × 2 features |
| Output | 3-day binary up/down predictions per stock |
| Loss / Metric | Binary Crossentropy / Binary Accuracy |
| Training | Client-side with early stopping |
| Visualization | Chart.js (accuracy + prediction timeline) |
| Deployment | GitHub Pages / local browser |

---

## ✅ Key Highlights

- 💻 **Runs fully in-browser** – no Python or backend needed.  
- ⚡ **Fast and responsive** – WebGL acceleration for real-time learning.  
- 🧠 **Adaptive training** – Stops early when validation results stabilize.  
- 📊 **Interactive UI** – Visual feedback for every step of the process.  
- 🔒 **Privacy-preserving** – User data never leaves the local environment.

---

## 🧠 Future Improvements

- Add technical indicators (e.g., RSI, MACD)
