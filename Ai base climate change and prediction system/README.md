# AI-Based Climate Change Analysis & Prediction System

## 📌 Project Overview
This project is an AI-driven system designed to analyze and predict climate change patterns. Using historical (synthetic) climate data, it performs Exploratory Data Analysis (EDA) and leverages Machine Learning algorithms (Linear Regression, Random Forest, ARIMA) to forecast future temperature trends.

**Project Capabilities:**
- Preprocessing and feature engineering on time-series climate data.
- Exploratory Data Analysis (EDA) on features like Temperature, Humidity, Rainfall, and AQI.
- Generation of correlation heatmaps and trend analysis charts.
- Machine Learning models to predict temperature based on historical context.
- Evaluation metrics (MAE, RMSE) and "Prediction vs Actual" visualizations.

---

## 📁 Repository Structure

```
├── data/
│   ├── climate_data.csv        # The generated realistic climate dataset
│   └── generate_dataset.py     # Script used to generate the dataset
├── src/
│   ├── data_preprocessing.py   # Data loading and scaling logic
│   ├── eda.py                  # Exploratory Data Analysis & visual plots
│   └── model_training.py       # ML Model training, predicting, & evaluation
├── notebook/
│   └── Climate_Analysis.ipynb  # Interactive Jupyter Notebook version of the project
├── output/                     # Generated charts and predictions
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
```

---

## 📊 Dataset Features
The dataset contains everyday records with the following columns:
- **Date**: The chronological date of the record.
- **Temperature_C**: Average daily temperature in Celsius.
- **Humidity_pct**: Average daily humidity percentage.
- **Rainfall_mm**: Total daily rainfall in millimeters.
- **AQI**: Air Quality Index.

---

## 💻 Technologies Used
- **Language**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Linear Regression, Random Forest), Statsmodels (ARIMA)
- **Data Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

---

## 🚀 How to Run the Project

### 1. Installation
Ensure you have Python installed on your system. Then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional)
The dataset is already provided in the `data/` folder. If you wish to regenerate it (e.g., for more years or different noise thresholds):
```bash
py data/generate_dataset.py
```

### 3. Run EDA & Visualizations
To explore the data and generate output graphs (Saved in the `output/` folder):
```bash
py src/eda.py
```

### 4. Train Models & Predict
To train the Linear Regression, Random Forest, and ARIMA models and evaluate them:
```bash
py src/model_training.py
```

### 5. Jupyter Notebook
If you want an interactive step-by-step walkthough:
```bash
jupyter notebook
```
Navigate to `notebook/Climate_Analysis.ipynb` and run all cells.

---

## 📈 Evaluation Metrics
The project evaluates model accuracy using:
- **MAE** (Mean Absolute Error): Average magnitude of errors in predictions.
- **RMSE** (Root Mean Square Error): Standard deviation of the prediction errors.
