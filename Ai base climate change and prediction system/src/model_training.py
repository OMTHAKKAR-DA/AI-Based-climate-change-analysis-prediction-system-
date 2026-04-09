import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from data_preprocessing import load_data, preprocess_for_ml

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculates evaluation metrics (MAE, RMSE).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"--- {model_name} Evaluation ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return mae, rmse

def plot_predictions(dates, y_true, y_pred, title, output_file):
    """
    Plots true vs predicted values.
    """
    plt.figure(figsize=(12, 5))
    # Plotting only the last 365 days to make it readable
    plt.plot(dates[-365:], y_true[-365:].values, label='Actual', color='tab:blue')
    plt.plot(dates[-365:], y_pred[-365:], label='Predicted', color='tab:orange', alpha=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_ml_models(df, output_dir='output'):
    """
    Trains and evaluates ML models for Temperature Prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess
    print("Preprocessing data for Machine Learning...")
    X_train, X_test, y_train, y_test, scaler, ml_data = preprocess_for_ml(df)
    test_dates = y_test.index
    
    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    evaluate_model(y_test, lr_pred, "Linear Regression")
    plot_predictions(test_dates, y_test, lr_pred, 'Linear Regression: Prediction vs Actual (Last 1 Year)', os.path.join(output_dir, 'lr_prediction_vs_actual.png'))
    
    # 2. Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    evaluate_model(y_test, rf_pred, "Random Forest")
    plot_predictions(test_dates, y_test, rf_pred, 'Random Forest: Prediction vs Actual (Last 1 Year)', os.path.join(output_dir, 'rf_prediction_vs_actual.png'))
    
def run_arima(df, output_dir='output'):
    """
    Trains an ARIMA model for Time-Series forecasting on Temperature.
    """
    print("\nTraining ARIMA Model...")
    # Resample to weekly data for ARIMA to speed up computation
    ts_data = df['Temperature_C'].resample('W').mean()
    
    # Train-test split
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]
    
    # Fit ARIMA model (using simplified parameters for weekly data)
    # Note: p,d,q parameters can be optimized using auto_arima, here we use generic
    try:
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Forecast
        predictions = model_fit.forecast(steps=len(test))
        
        # Evaluate
        evaluate_model(test, predictions, "ARIMA")
        
        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(train.index[-52:], train.values[-52:], label='Train (last 1 year)')
        plt.plot(test.index, test.values, label='Actual Test')
        plt.plot(test.index, predictions, label='ARIMA Forecast', color='red')
        plt.title('ARIMA Time-series Forecasting')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'arima_forecast.png'))
        plt.close()
        print("ARIMA completed.")
    except Exception as e:
        print(f"ARIMA modeling failed: {e}")

if __name__ == "__main__":
    df = load_data('data/climate_data.csv')
    run_ml_models(df)
    run_arima(df)
    print("\nAll models trained and outputs generated in 'output/' folder.")
