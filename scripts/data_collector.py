import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config import CSV_FILE, API_KEY, DATA_DIR



class AirQualityDataCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic air quality data for demonstration"""
        np.random.seed(42)
        
        # Generate realistic air quality data
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='h'),
            'temperature': np.random.normal(20, 10, n_samples),  # Celsius
            'humidity': np.random.uniform(30, 90, n_samples),    # %
            'pressure': np.random.normal(1013, 20, n_samples),  # hPa
            'wind_speed': np.random.exponential(3, n_samples),  # m/s
            'pm25': np.random.lognormal(2.5, 0.8, n_samples),  # μg/m³
            'pm10': np.random.lognormal(3.0, 0.7, n_samples),  # μg/m³
            'no2': np.random.lognormal(2.0, 0.6, n_samples),   # μg/m³
            'so2': np.random.lognormal(1.5, 0.9, n_samples),   # μg/m³
            'co': np.random.lognormal(5.0, 0.5, n_samples),    # μg/m³
            'o3': np.random.lognormal(4.0, 0.6, n_samples),    # μg/m³
        }
        
        df = pd.DataFrame(data)
        
        # Add some correlations to make it more realistic
        # Higher temperature tends to increase ozone
        df['o3'] = df['o3'] * (1 + 0.02 * df['temperature'])
        
        # Higher humidity tends to reduce PM2.5
        df['pm25'] = df['pm25'] * (1 - 0.005 * df['humidity'])
        
        # Create AQI categories based on PM2.5 levels
        df['aqi_category'] = pd.cut(df['pm25'], 
                                   bins=[0, 12, 35, 55, 150, 250, float('inf')],
                                   labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                                           'Unhealthy', 'Very Unhealthy', 'Hazardous'])
        
        # Calculate overall AQI (simplified)
        df['aqi'] = np.maximum(
            df['pm25'] * 4.17,  # PM2.5 to AQI conversion (simplified)
            df['pm10'] * 2.5    # PM10 to AQI conversion (simplified)
        ).round().astype(int)
        
        return df
    
    def collect_real_data(self, lat, lon, days_back=30):
        """Collect real air quality data from OpenWeatherMap API"""
        if not self.api_key:
            raise ValueError("API key required for real data collection")
        
        data_list = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        current_time = start_time
        while current_time <= end_time:
            timestamp = int(current_time.timestamp())
            
            url = f"{self.base_url}/history"
            params = {
                'lat': lat,
                'lon': lon,
                'start': timestamp,
                'end': timestamp + 3600,  # 1 hour window
                'appid': self.api_key
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'list' in data and data['list']:
                    air_data = data['list'][0]
                    record = {
                        'timestamp': current_time,
                        'pm25': air_data['components'].get('pm2_5', 0),
                        'pm10': air_data['components'].get('pm10', 0),
                        'no2': air_data['components'].get('no2', 0),
                        'so2': air_data['components'].get('so2', 0),
                        'co': air_data['components'].get('co', 0),
                        'o3': air_data['components'].get('o3', 0),
                        'aqi': air_data.get('main', {}).get('aqi', 0)
                    }
                    data_list.append(record)
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {current_time}: {e}")
            
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(data_list)
    
    def save_data(self, df, filename=None):
        """Save collected data to CSV file"""
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = filename or CSV_FILE
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath


def main():
    collector = AirQualityDataCollector()
    
    # Generate sample data
    print("Generating sample air quality data...")
    df = collector.generate_sample_data(2000)
    
    # Save data
    filepath = collector.save_data(df)
    
    print(f"Generated {len(df)} records")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Show basic statistics
    print(f"\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
