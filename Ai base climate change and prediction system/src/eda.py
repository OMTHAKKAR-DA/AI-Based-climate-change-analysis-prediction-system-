import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='output'):
    """
    Performs EDA on climate data and saves plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary Statistics
    summary = df.describe()
    summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Climate Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Temperature Trend Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Temperature_C'], color='tab:red', alpha=0.6)
    # Add a rolling mean to see trend better
    plt.plot(df.index, df['Temperature_C'].rolling(window=365).mean(), color='darkred', linewidth=2, label='1-Year Rolling Mean')
    plt.title('Temperature Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_trend.png'))
    plt.close()
    
    # 4. Rainfall Pattern (Yearly view)
    # Let's aggregate by month-year
    monthly_rainfall = df['Rainfall_mm'].resample('ME').sum()
    plt.figure(figsize=(12, 5))
    plt.bar(monthly_rainfall.index, monthly_rainfall.values, color='tab:blue', width=20)
    plt.title('Monthly Rainfall Pattern')
    plt.xlabel('Date')
    plt.ylabel('Total Rainfall (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rainfall_patterns.png'))
    plt.close()

if __name__ == "__main__":
    # If run standalone, read the generated dataset and run EDA
    from data_preprocessing import load_data
    df = load_data('data/climate_data.csv')
    perform_eda(df)
    print("EDA completed. Plots saved to 'output/' folder.")
