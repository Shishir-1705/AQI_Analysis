# AQI Analysis and Forecasting Script
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Optional forecasting libraries - may need to be installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    # pmdarima provides auto_arima
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Config / Parameters
# ---------------------------
CSV_PATH = 'air_quality_data.csv'  # your file
FORECAST_HORIZON_DAYS = 180       # ~ 6 months
OUTPUT_DIR = 'outputs'
TOP_N_CITIES = 4                  # forecast for top N cities by record count
DATE_COL_CANDIDATES = ['date', 'timestamp', 'Date', 'DATE']

# create outputs folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load and initial cleaning
# ---------------------------
df = pd.read_csv(CSV_PATH)
print("Initial dataframe shape:", df.shape)
print(df.info())
print("Missing values:\n", df.isnull().sum())

# Ensure pollutant columns exist
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    else:
        # If pollutant_avg missing, try to compute from min/max
        print(f"Warning: {col} not found in dataset.")

# Identify date column
date_col = None
for c in DATE_COL_CANDIDATES:
    if c in df.columns:
        date_col = c
        break
if date_col is None:
    # try to find a datetime-like column automatically
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            date_col = c
            break

if date_col is None:
    raise ValueError("No date column found. Please include a column named 'date' or 'timestamp' or similar.")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])  # drop rows with invalid dates
df = df.sort_values(by=date_col).reset_index(drop=True)

# Ensure city and pollutant_avg exist
if 'city' not in df.columns:
    raise ValueError("No 'city' column found in the dataset.")
if 'pollutant_avg' not in df.columns:
    # attempt to create pollutant_avg from pollutant_min/max
    if 'pollutant_min' in df.columns and 'pollutant_max' in df.columns:
        df['pollutant_avg'] = df[['pollutant_min', 'pollutant_max']].mean(axis=1)
        print("Created 'pollutant_avg' from min/max.")
    else:
        raise ValueError("No 'pollutant_avg' and cannot derive it from min/max. Please provide pollutant_avg.")

# ---------------------------
# Existing EDA (kept from your script, simplified)
# ---------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 5))
sns.kdeplot(df['pollutant_avg'], fill=True)
plt.title('Density Plot of Pollutant Averages')
plt.xlabel('Average Pollutant Level')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kde_pollutant_avg.png'))
plt.close()

if 'region' in df.columns:
    region_avg = df.groupby('region')['pollutant_avg'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=region_avg.values, y=region_avg.index)
    plt.title('Average Pollution Levels by Region')
    plt.xlabel('Pollution Level')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'region_avg_pollution.png'))
    plt.close()

# City boxplot (top cities)
plt.figure(figsize=(12, 6))
top_cities = df['city'].value_counts().nlargest(5).index
subset = df[df['city'].isin(top_cities)]
sns.boxplot(x='city', y='pollutant_max', data=subset)
plt.title('Pollutant Max Values in Top Cities')
plt.xlabel('City')
plt.ylabel('Max Pollutant Level')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_top_cities_pollutant_max.png'))
plt.close()

# Heatmap via binned lat/long if available
if 'latitude' in df.columns and 'longitude' in df.columns:
    df['lat_bin'] = pd.cut(df['latitude'], bins=10)
    df['long_bin'] = pd.cut(df['longitude'], bins=10)
    grid = df.groupby(['lat_bin', 'long_bin'])['pollutant_avg'].mean().unstack()
    plt.figure(figsize=(10,8))
    sns.heatmap(grid, cmap='Spectral', linewidths=0.5)
    plt.title('Spatial Heatmap of Average Pollution (Binned)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_heatmap_binned.png'))
    plt.close()

# Pie chart for pollutant types if available
if 'pollutant_id' in df.columns:
    pollutant_counts = df['pollutant_id'].value_counts()
    plt.figure(figsize=(8,8))
    plt.pie(pollutant_counts, labels=pollutant_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Pollutant Types Monitored')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pollutant_type_share.png'))
    plt.close()

# Min vs Max scatter
if {'pollutant_min', 'pollutant_max'}.issubset(df.columns):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='pollutant_min', y='pollutant_max', hue='pollutant_avg', alpha=0.6)
    plt.title('Pollution Level Variation (Min vs Max)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'min_vs_max_scatter.png'))
    plt.close()

# Regression between pollutant_min and pollutant_max (if available)
if 'pollutant_min' in df.columns and 'pollutant_max' in df.columns:
    plt.figure(figsize=(10,6))
    sns.regplot(x='pollutant_min', y='pollutant_max', data=df, scatter_kws={'alpha':0.3})
    X = df[['pollutant_min']].fillna(0)
    y = df['pollutant_max'].fillna(0)
    lin = LinearRegression().fit(X, y)
    r_sq = lin.score(X, y)
    plt.text(0.05, 0.95, f'R² = {r_sq:.2f}\ny = {lin.coef_[0]:.2f}x + {lin.intercept_:.2f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'reg_min_vs_max.png'))
    plt.close()

# ---------------------------
# Time-series preparation
# ---------------------------
# We'll build a city-date aggregated series (daily mean)
df_ts = df[[date_col, 'city', 'pollutant_avg']].copy()
df_ts = df_ts.rename(columns={date_col: 'ds', 'pollutant_avg': 'y'})
df_ts['ds'] = pd.to_datetime(df_ts['ds'])
# aggregate to daily mean per city
city_daily = df_ts.groupby(['city', pd.Grouper(key='ds', freq='D')])['y'].mean().reset_index()

# choose top cities (by record count)
top_cities_for_forecast = city_daily['city'].value_counts().nlargest(TOP_N_CITIES).index.tolist()
print("Forecast will be generated for cities:", top_cities_for_forecast)

# Utility: Train/test split for time series (last N days as test)
def train_test_split_ts(series, test_days=30):
    series = series.dropna()
    if len(series) < test_days + 10:
        test_days = max(1, int(len(series) * 0.2))
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    return train, test

# Forecasting function returns forecast dataframe and metrics
def forecast_city_arima(train_series, test_series, horizon_days):
    """
    Try to use pmdarima.auto_arima if available, otherwise fit a simple ARIMA(2,1,2)
    Return pandas Series of forecast (index = date)
    """
    try:
        if PMDARIMA_AVAILABLE:
            print("Using pmdarima.auto_arima for ARIMA order selection.")
            arima_model = pm.auto_arima(train_series, seasonal=False, error_action='ignore', suppress_warnings=True)
            order = arima_model.order
            model = ARIMA(train_series, order=order).fit()
        else:
            # fallback
            print("pmdarima not available. Using ARIMA(2,1,2) fallback.")
            model = ARIMA(train_series, order=(2,1,2)).fit()
    except Exception as e:
        print("ARIMA fit failed, trying SARIMAX simple fallback:", str(e))
        model = SARIMAX(train_series, order=(1,1,1)).fit(disp=False)
    # forecast for test + horizon if needed
    start = test_series.index[0]
    end_for_test = test_series.index[-1]
    # Forecast covering both test range and future horizon
    forecast_index = pd.date_range(start=start, periods=len(test_series) + horizon_days, freq='D')
    forecast = model.predict(start=forecast_index[0], end=forecast_index[-1])
    forecast = pd.Series(forecast, index=forecast_index)
    # compute metrics on the portion aligned with test_series
    pred_test = forecast.loc[test_series.index]
    rmse = mean_squared_error(test_series.values, pred_test.values, squared=False)
    mae = mean_absolute_error(test_series.values, pred_test.values)
    return forecast, rmse, mae, model

def forecast_city_prophet(train_df, test_df, horizon_days):
    """
    train_df and test_df are DataFrames with 'ds' and 'y' columns and ds as datetime index
    Returns forecast pd.Series starting from test start for length test + horizon
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet library not available.")
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(train_df.reset_index().rename(columns={'index':'ds'}))  # train_df index = ds
    future = m.make_future_dataframe(periods=len(test_df) + horizon_days, freq='D')
    forecast = m.predict(future)
    forecast_series = pd.Series(forecast['yhat'].values, index=pd.to_datetime(forecast['ds']))
    pred_test = forecast_series.loc[test_df.index]
    rmse = mean_squared_error(test_df['y'].values, pred_test.values, squared=False)
    mae = mean_absolute_error(test_df['y'].values, pred_test.values)
    return forecast_series, rmse, mae, m

# Container for summary results
summary_rows = []

# Run forecasts for each top city
for city in top_cities_for_forecast:
    print("\nProcessing city:", city)
    series = city_daily[city_daily['city'] == city].set_index('ds')['y'].sort_index()
    series = series.asfreq('D')  # fill missing days with NaN
    series = series.interpolate(method='time')  # simple interpolation for missing days
    
    if series.dropna().shape[0] < 20:
        print(f"Not enough data for reliable forecasting for {city}. Skipping.")
        continue

    train, test = train_test_split_ts(series, test_days=30)
    # ARIMA forecast
    try:
        arima_forecast, arima_rmse, arima_mae, arima_model = forecast_city_arima(train, test, FORECAST_HORIZON_DAYS)
    except Exception as e:
        print("ARIMA forecast failed for", city, ":", e)
        arima_forecast, arima_rmse, arima_mae, arima_model = None, np.nan, np.nan, None

    # Prophet forecast (if available)
    prophet_forecast, prophet_rmse, prophet_mae = None, np.nan, np.nan
    if PROPHET_AVAILABLE:
        try:
            train_df = train.reset_index().rename(columns={'ds':'ds', 'y':'y'}).set_index('ds')
            test_df = test.reset_index().rename(columns={'ds':'ds', 'y':'y'}).set_index('ds')
            prophet_forecast, prophet_rmse, prophet_mae, prophet_model = forecast_city_prophet(train_df, test_df, FORECAST_HORIZON_DAYS)
        except Exception as e:
            print("Prophet forecast failed for", city, ":", e)
            prophet_forecast, prophet_rmse, prophet_mae = None, np.nan, np.nan

    # Choose best model based on RMSE
    best_model_name = None
    if not np.isnan(arima_rmse) and not np.isnan(prophet_rmse):
        best_model_name = 'ARIMA' if arima_rmse <= prophet_rmse else 'Prophet'
    elif not np.isnan(arima_rmse):
        best_model_name = 'ARIMA'
    elif not np.isnan(prophet_rmse):
        best_model_name = 'Prophet'
    else:
        best_model_name = 'None'

    print(f"City: {city} | ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f} | Prophet RMSE: {prophet_rmse if not np.isnan(prophet_rmse) else 'NA'}")

    # Save forecast CSVs and produce plots comparing history + forecasts
    # Consolidate index ranges: historical entire series + forecast horizon
    horizon_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq='D')

    result_df = pd.DataFrame(index=pd.date_range(start=series.index[0], end=series.index[-1] + pd.Timedelta(days=FORECAST_HORIZON_DAYS), freq='D'))
    result_df['historical'] = series.reindex(result_df.index)

    if arima_forecast is not None:
        # align and put into df
        result_df['arima_forecast'] = arima_forecast.reindex(result_df.index)
    if prophet_forecast is not None:
        result_df['prophet_forecast'] = prophet_forecast.reindex(result_df.index)

    # Save CSV
    csv_out = os.path.join(OUTPUT_DIR, f'{city}_forecast_results.csv')
    result_df.to_csv(csv_out)
    print("Saved forecast CSV to:", csv_out)

    # Plot - history + forecasts + test window
    plt.figure(figsize=(12,6))
    plt.plot(result_df.index, result_df['historical'], label='Historical', linewidth=1)
    if 'arima_forecast' in result_df.columns:
        plt.plot(result_df.index, result_df['arima_forecast'], label='ARIMA Forecast', linestyle='--')
    if 'prophet_forecast' in result_df.columns:
        plt.plot(result_df.index, result_df['prophet_forecast'], label='Prophet Forecast', linestyle=':')
    # mark train/test split
    plt.axvline(x=test.index[0], color='gray', linestyle=':', label='Test start')
    plt.title(f'{city} - Historical and Forecast (next {FORECAST_HORIZON_DAYS} days)')
    plt.xlabel('Date')
    plt.ylabel('Pollutant Average')
    plt.legend()
    plt.tight_layout()
    plot_out = os.path.join(OUTPUT_DIR, f'{city}_forecast_plot.png')
    plt.savefig(plot_out)
    plt.close()
    print("Saved forecast plot to:", plot_out)

    # Summary row
    summary_rows.append({
        'city': city,
        'arima_rmse': float(arima_rmse) if arima_rmse is not None else np.nan,
        'arima_mae': float(arima_mae) if arima_mae is not None else np.nan,
        'prophet_rmse': float(prophet_rmse) if prophet_forecast is not None else np.nan,
        'prophet_mae': float(prophet_mae) if prophet_forecast is not None else np.nan,
        'best_model': best_model_name
    })

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, 'forecast_model_comparison_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print("\nSaved model comparison summary to:", summary_csv)
print(summary_df)

# ---------------------------
# Auto Insights & Recommendations (basic templated logic)
# ---------------------------
insights = []
if not summary_df.empty:
    # Rank cities by best model RMSE (lower better)
    ranked = summary_df.sort_values(by='arima_rmse', na_position='last')  # using ARIMA as baseline
    insights.append("Model performance summary (lower RMSE is better):")
    for _, row in summary_df.iterrows():
        insights.append(f"- {row['city']}: Best model = {row['best_model']}, ARIMA RMSE = {row['arima_rmse']:.2f}, Prophet RMSE = {row['prophet_rmse'] if not np.isnan(row['prophet_rmse']) else 'NA'}")

    # Simple rule: if forecasted mean in horizon > historical mean by threshold => rising trend
    for row in summary_rows:
        city = row['city']
        res_csv = os.path.join(OUTPUT_DIR, f'{city}_forecast_results.csv')
        res = pd.read_csv(res_csv, index_col=0, parse_dates=True)
        hist_mean = res['historical'].loc[:df_ts['ds'].max()].mean()
        # choose forecast series if best model exists
        best = row['best_model']
        if best == 'ARIMA' and 'arima_forecast' in res.columns:
            future_mean = res['arima_forecast'].loc[res.index > df_ts['ds'].max()].mean()
        elif best == 'Prophet' and 'prophet_forecast' in res.columns:
            future_mean = res['prophet_forecast'].loc[res.index > df_ts['ds'].max()].mean()
        else:
            future_mean = np.nan

        if not np.isnan(future_mean):
            pct_change = (future_mean - hist_mean) / (hist_mean + 1e-9) * 100
            if pct_change > 5:
                insights.append(f"Alert: {city} forecast mean is expected to increase by {pct_change:.1f}% over historical mean — consider preemptive measures.")
            elif pct_change < -5:
                insights.append(f"Good sign: {city} forecast mean expected to decrease by {abs(pct_change):.1f}% compared to historical mean.")
            else:
                insights.append(f"{city} expected to remain relatively stable (change {pct_change:.1f}%).")

# Policy recommendations (templated)
insights.append("\nPolicy suggestions based on forecasts and EDA:")
insights.append("- Issue seasonal / festival-based warnings for cities with forecast spikes (e.g., winter months).")
insights.append("- Strengthen vehicular emission controls on days predicted to have poor air quality.")
insights.append("- Identify and monitor high-risk grid cells (use spatial heatmap) and deploy targeted interventions.")
insights.append("- Promote green cover & low-emission zones in consistently high-pollution areas.")
insights.append("- Further improvement: build an ensemble model combining ARIMA & Prophet for robustness.")

# Print and save insights
insights_txt = "\n".join(insights)
print("\n--- Insights & Recommendations ---\n")
print(insights_txt)
with open(os.path.join(OUTPUT_DIR, 'insights_and_recommendations.txt'), 'w') as f:
    f.write(insights_txt)

print("\nAll outputs saved in the folder:", OUTPUT_DIR)
print("If Prophet is not installed, install via: pip install prophet")
print("If pmdarima (auto_arima) is desired, install via: pip install pmdarima")
