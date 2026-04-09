import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset and parse dates.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

def preprocess_for_ml(df, target_col='Temperature_C'):
    """
    Creates lag features for time-series forecasting using ML.
    We will try to predict the target using past 7 days of data.
    """
    data = df.copy()
    
    # Create lag features
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data[target_col].shift(lag)
        
    # Extract temporal features
    data['day_of_year'] = data.index.dayofyear
    data['month'] = data.index.month
    
    # Drop NaNs created by lagging
    data.dropna(inplace=True)
    
    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split into train and test sets (chronological split, not random)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data
