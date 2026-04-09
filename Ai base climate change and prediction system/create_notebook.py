import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

add_md("# AI-Based Climate Change Analysis & Prediction System\n\nThis notebook demonstrates exploratory data analysis (EDA) and prediction modeling for our synthetic climate dataset.")

add_md("## 1. Import Libraries & Load Data")
add_code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\n\n# Set display options\nsns.set_theme(style=\"whitegrid\")\nplt.rcParams['figure.figsize'] = (12, 6)\n\ndf = pd.read_csv('../data/climate_data.csv', parse_dates=['Date'])\ndf.set_index('Date', inplace=True)\ndf.head()")

add_md("## 2. Exploratory Data Analysis (EDA)")
add_code("df.describe()")
add_code("# Correlation Heatmap\nplt.figure(figsize=(8, 6))\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=\".2f\")\nplt.title('Correlation Heatmap')\nplt.show()")
add_code("# Temperature Trend Over Time\nplt.figure(figsize=(12, 5))\nplt.plot(df.index, df['Temperature_C'], color='tab:red', alpha=0.5)\nplt.plot(df.index, df['Temperature_C'].rolling(window=365).mean(), color='darkred', linewidth=2, label='1-Year Rolling Mean')\nplt.title('Temperature Trend Over Time')\nplt.ylabel('Temperature (°C)')\nplt.legend()\nplt.show()")

add_md("## 3. Data Preprocessing for Machine Learning\nWe create \"lag\" features to predict today's temperature based on the past week.")
add_code("data = df.copy()\nfor lag in range(1, 8):\n    data[f'lag_{lag}'] = data['Temperature_C'].shift(lag)\ndata.dropna(inplace=True)\n\nX = data.drop(columns=['Temperature_C'])\ny = data['Temperature_C']\n\n# Chronological train-test split (80-20)\nsplit_idx = int(len(data) * 0.8)\nX_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\ny_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]")

add_md("## 4. Model Training & Evaluation")
add_code("from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error\n\n# Train Random Forest\nrf = RandomForestRegressor(n_estimators=100, random_state=42)\nrf.fit(X_train, y_train)\n\n# Predict and Evaluate\nrf_pred = rf.predict(X_test)\nmae = mean_absolute_error(y_test, rf_pred)\nrmse = np.sqrt(mean_squared_error(y_test, rf_pred))\n\nprint(f\"Random Forest MAE: {mae:.4f}\")\nprint(f\"Random Forest RMSE: {rmse:.4f}\")")

add_code("# Plot Predictions vs Actual for the last 365 days\nplt.figure(figsize=(12, 5))\nplt.plot(y_test.index[-365:], y_test.values[-365:], label='Actual', color='tab:blue')\nplt.plot(y_test.index[-365:], rf_pred[-365:], label='Predicted', color='tab:orange', alpha=0.8)\nplt.title('Random Forest: Prediction vs Actual')\nplt.ylabel('Temperature (°C)')\nplt.legend()\nplt.show()")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebook/Climate_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)
    
print("Notebook created successfully.")
