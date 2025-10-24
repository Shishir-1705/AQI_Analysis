Air Quality Index (AQI) Analysis & Forecasting — Indian Cities 🇮🇳

📊 Overview
This project performs an in-depth **Air Quality Index (AQI) analysis and forecasting** across major Indian cities.  
It uncovers **pollution patterns, seasonal variations, spatial trends**, and predicts **future pollution levels** using statistical and machine learning models.  
The analysis is based on multi-year datasets collected from verified sources like the **Central Pollution Control Board (CPCB)**.

---

🧠 Objectives
- Analyze **historical air quality data** for key pollutants (PM2.5, PM10, NO₂, SO₂, etc.).
- Identify **city-wise and seasonal pollution trends**.
- Visualize **spatial pollution hotspots** using latitude–longitude heatmaps.
- Build and compare **time series forecasting models** (ARIMA & Prophet) for pollution prediction.
- Generate **data-driven insights** and **policy recommendations** for cleaner urban environments.

---

⚙️ Key Features
- 🧹 **Data Cleaning & Preprocessing** — Handles missing values, validates data types, and fills gaps using median or interpolation.  
- 📈 **Exploratory Data Analysis (EDA)** — KDE plots, boxplots, heatmaps, pie charts, and regression visualizations.  
- 🔥 **City-Level Pollution Comparison** — Highlights pollution variability among major Indian cities.  
- 🕒 **Time-Series Forecasting** — Predicts AQI trends for the next 6 months using:
  - ARIMA model (Auto-regressive Integrated Moving Average)
  - Prophet model (Facebook’s forecasting library)
- 📊 **Model Performance Evaluation** — Compares models using RMSE & MAE metrics.  
- 💬 **Insights & Recommendations** — Automatically generated policy-level takeaways based on forecast outcomes.  

---

📦 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Forecasting | statsmodels (ARIMA), prophet |
| Evaluation | scikit-learn (RMSE, MAE) |

---

🗺️ Visual Highlights
- Density Plot of Pollutant Averages  
- Region-wise Pollution Comparison  
- City-level Boxplots (Top Polluted Cities)  
- Spatial Heatmaps (Latitude vs Longitude)  
- Pollution Type Distribution Pie Chart  
- Regression: Pollutant Min vs Max  
- Forecast Plots for Delhi, Mumbai, Kolkata, Bengaluru  

All output visuals and forecast CSVs are stored in the **`outputs/`** folder.

---

🚀 Results & Insights
- Cities like **Delhi** and **Kolkata** show recurring winter spikes in pollution.  
- Forecasts reveal an expected **rise of 10–20%** AQI levels during festive and winter months.  
- The **Prophet model** provided smoother long-term trends, while **ARIMA** captured short-term fluctuations better.  
- Policy suggestions include stricter emission control, green zone development, and early-warning systems.

---

🧩 Repository Structure

OUTLIER_THRESHOLD = 1.5  # Modify IQR multiplier
COLOR_PALETTE = 'coolwarm'  # Change visualization colors
TOP_CITIES = 5  # Number of cities to analyze

