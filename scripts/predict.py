import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from config import MODEL_FILE

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
    
    def load_model(self, filepath=MODEL_FILE):
        """Load the trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            print(f"Model loaded successfully from {filepath}")
            print(f"Expected features: {self.feature_columns}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            print("Please run train_model.py first to train the model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def engineer_features(self, df):
        """Create additional features from existing data"""
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,
                                            3: 1, 4: 1, 5: 1,
                                            6: 2, 7: 2, 8: 2,
                                            9: 3, 10: 3, 11: 3})
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-8)
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        available_pollutants = [col for col in pollutants if col in df.columns]
        if available_pollutants:
            df['pollution_index'] = df[available_pollutants].sum(axis=1)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        if 'wind_speed' in df.columns and 'pm25' in df.columns:
            df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
        return df
    
    def prepare_input(self, data):
        """Prepare input data for prediction"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        df = self.engineer_features(df)
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features will be set to 0: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        X = df[self.feature_columns]
        X = X.fillna(X.mean())
        return X
    
    def predict(self, data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        X = self.prepare_input(data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_with_confidence(self, data):
        """Make predictions with confidence intervals (for tree-based models)"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        X = self.prepare_input(data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            lower_bound = np.percentile(tree_predictions, 5, axis=0)
            upper_bound = np.percentile(tree_predictions, 95, axis=0)
            return predictions, lower_bound, upper_bound
        return predictions, None, None
    
    def interpret_aqi(self, aqi_value):
        """Interpret AQI value and provide health recommendations"""
        if aqi_value <= 50:
            category = "Good"
            color = "Green"
            message = "Air quality is satisfactory and poses little or no risk."
        elif aqi_value <= 100:
            category = "Moderate"
            color = "Yellow"
            message = "Air quality is acceptable. Sensitive individuals may experience minor issues."
        elif aqi_value <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "Orange"
            message = "Sensitive groups may experience health effects. General public is less likely to be affected."
        elif aqi_value <= 200:
            category = "Unhealthy"
            color = "Red"
            message = "Everyone may experience health effects. Sensitive groups may experience more serious effects."
        elif aqi_value <= 300:
            category = "Very Unhealthy"
            color = "Purple"
            message = "Health alert: everyone may experience more serious health effects."
        else:
            category = "Hazardous"
            color = "Maroon"
            message = "Health warnings of emergency conditions. Everyone is likely to be affected."
        return {
            'category': category,
            'color': color,
            'message': message
        }

def main():
    parser = argparse.ArgumentParser(description='Air Quality Prediction')
    parser.add_argument('--input', type=str, help='Input data (JSON format or CSV file)')
    parser.add_argument('--temperature', type=float, default=25.0, help='Temperature in Celsius')
    parser.add_argument('--humidity', type=float, default=60.0, help='Humidity percentage')
    parser.add_argument('--pressure', type=float, default=1013.0, help='Pressure in hPa')
    parser.add_argument('--wind_speed', type=float, default=3.0, help='Wind speed in m/s')
    parser.add_argument('--pm25', type=float, default=20.0, help='PM2.5 concentration')
    parser.add_argument('--pm10', type=float, default=30.0, help='PM10 concentration')
    parser.add_argument('--no2', type=float, default=15.0, help='NO2 concentration')
    parser.add_argument('--so2', type=float, default=5.0, help='SO2 concentration')
    parser.add_argument('--co', type=float, default=100.0, help='CO concentration')
    parser.add_argument('--o3', type=float, default=80.0, help='O3 concentration')
    
    args = parser.parse_args()
    predictor = AirQualityPredictor()
    
    # Load model
    if not predictor.load_model():
        return
    
    # Prepare input data
    if args.input:
        if args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        else:
            import json
            data = json.loads(args.input)
    else:
        data = {
            'temperature': args.temperature,
            'humidity': args.humidity,
            'pressure': args.pressure,
            'wind_speed': args.wind_speed,
            'pm25': args.pm25,
            'pm10': args.pm10,
            'no2': args.no2,
            'so2': args.so2,
            'co': args.co,
            'o3': args.o3
        }
    
    # Make prediction
    try:
        predictions, lower_bound, upper_bound = predictor.predict_with_confidence(data)
        print(f"\n{'='*50}")
        print(f"AIR QUALITY PREDICTION")
        print(f"{'='*50}")
        
        if isinstance(predictions, np.ndarray):
            for i, pred in enumerate(predictions):
                interpretation = predictor.interpret_aqi(pred)
                print(f"\nPrediction {i+1}:")
                print(f"Predicted AQI: {pred:.1f}")
                print(f"Category: {interpretation['category']} ({interpretation['color']})")
                print(f"Health Message: {interpretation['message']}")
                if lower_bound is not None and upper_bound is not None:
                    print(f"90% Confidence Interval: [{lower_bound[i]:.1f}, {upper_bound[i]:.1f}]")
        else:
            interpretation = predictor.interpret_aqi(predictions)
            print(f"\nPredicted AQI: {predictions:.1f}")
            print(f"Category: {interpretation['category']} ({interpretation['color']})")
            print(f"Health Message: {interpretation['message']}")
            if lower_bound is not None and upper_bound is not None:
                print(f"90% Confidence Interval: [{lower_bound:.1f}, {upper_bound:.1f}]")
        print(f"\n{'='*50}")
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
