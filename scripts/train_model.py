import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from config import CSV_FILE, MODEL_FILE, DATA_DIR, MODEL_DIR

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'aqi'
    
    def load_data(self, filepath=CSV_FILE):
        """Load and preprocess the air quality data"""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except FileNotFoundError:
            print(f"Data file not found: {filepath}")
            print("Please run data_collector.py first to generate the data")
            return None
    
    def engineer_features(self, df):
        """Create additional features from existing data"""
        df = df.copy()
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                            3: 1, 4: 1, 5: 1,    # Spring
                                            6: 2, 7: 2, 8: 2,    # Summer
                                            9: 3, 10: 3, 11: 3}) # Fall
        # Create pollution ratios
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-8)
        # Create composite pollution index
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        available_pollutants = [col for col in pollutants if col in df.columns]
        if available_pollutants:
            df['pollution_index'] = df[available_pollutants].sum(axis=1)
        # Weather-pollution interaction
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        if 'wind_speed' in df.columns and 'pm25' in df.columns:
            df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
        return df

    def prepare_features(self, df):
        """Prepare features for training"""
        # Feature engineering
        df = self.engineer_features(df)
        # Remove timestamp and categorical columns for now
        exclude_columns = ['timestamp', 'aqi_category']
        feature_columns = [col for col in df.columns 
                           if col not in exclude_columns and col != self.target_column]
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        self.feature_columns = feature_columns
        return df[feature_columns], df[self.target_column]
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the machine learning model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        # Evaluate the model
        self.evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        return X_test_scaled, y_test, y_pred_test
    
    def evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test):
        """Evaluate model performance"""
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        print(f"\nModel Performance:")
        print(f"Training MSE: {train_mse:.2f}")
        print(f"Testing MSE: {test_mse:.2f}")
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Testing MAE: {test_mae:.2f}")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Testing R²: {test_r2:.3f}")
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 10 Feature Importances:")
            print(importance_df.head(10))

    def save_model(self, filepath=MODEL_FILE):
        """Save the trained model and scaler"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'trained_at': datetime.now().isoformat()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=MODEL_FILE):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            print(f"Model loaded from {filepath}")
            print(f"Trained at: {model_data.get('trained_at', 'Unknown')}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            return False
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

def main():
    predictor = AirQualityPredictor()
    # Load data
    df = predictor.load_data()
    if df is None:
        return
    # Prepare features
    X, y = predictor.prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    # Train model
    X_test, y_test, y_pred = predictor.train_model(X, y, model_type='random_forest')
    # Save model
    predictor.save_model()
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI')
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
