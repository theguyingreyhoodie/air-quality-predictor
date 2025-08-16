import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config import MODEL_FILE

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def to_scalar(val):
    if isinstance(val, np.ndarray):
        return val.item() if val.size == 1 else float(val[0])
    return float(val)

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
    
    def load_model(self, filepath=MODEL_FILE):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def engineer_features(self, df):
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
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        df = self.engineer_features(df)
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                df[feature] = 0
        X = df[self.feature_columns]
        X = X.fillna(X.mean())
        return X
    
    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        X = self.prepare_input(data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_with_confidence(self, data):
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
        if aqi_value <= 50:
            category = "Good"
            color = "Green"
            message = "Air quality is satisfactory and poses little or no risk."
            css_class = "aqi-good"
        elif aqi_value <= 100:
            category = "Moderate"
            color = "Yellow"
            message = "Air quality is acceptable. Sensitive individuals may experience minor issues."
            css_class = "aqi-moderate"
        elif aqi_value <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "Orange"
            message = "Sensitive groups may experience health effects. General public is less likely to be affected."
            css_class = "aqi-unhealthy-sensitive"
        elif aqi_value <= 200:
            category = "Unhealthy"
            color = "Red"
            message = "Everyone may experience health effects. Sensitive groups may experience more serious effects."
            css_class = "aqi-unhealthy"
        elif aqi_value <= 300:
            category = "Very Unhealthy"
            color = "Purple"
            message = "Health alert: everyone may experience more serious health effects."
            css_class = "aqi-very-unhealthy"
        else:
            category = "Hazardous"
            color = "Maroon"
            message = "Health warnings of emergency conditions. Everyone is likely to be affected."
            css_class = "aqi-hazardous"
        return {
            'category': category,
            'color': color,
            'message': message,
            'css_class': css_class
        }

def main():
    st.markdown('<h1 class="main-header">🌬️ Air Quality Predictor</h1>', unsafe_allow_html=True)
    
    predictor = AirQualityPredictor()
    
    if not predictor.load_model():
        st.error("❌ Model not found! Please train the model first by running `python train_model.py`")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    st.sidebar.header("🔧 Input Parameters")
    
    st.sidebar.subheader("Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", -30.0, 50.0, 25.0, 0.1)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0, 0.1)
    pressure = st.sidebar.slider("Pressure (hPa)", 950.0, 1050.0, 1013.0, 0.1)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 3.0, 0.1)
    
    st.sidebar.subheader("Pollutant Concentrations")
    pm25 = st.sidebar.slider("PM2.5 (μg/m³)", 0.0, 500.0, 20.0, 0.1)
    pm10 = st.sidebar.slider("PM10 (μg/m³)", 0.0, 600.0, 30.0, 0.1)
    no2 = st.sidebar.slider("NO2 (μg/m³)", 0.0, 200.0, 15.0, 0.1)
    so2 = st.sidebar.slider("SO2 (μg/m³)", 0.0, 100.0, 5.0, 0.1)
    co = st.sidebar.slider("CO (μg/m³)", 0.0, 30000.0, 100.0, 1.0)
    o3 = st.sidebar.slider("O3 (μg/m³)", 0.0, 300.0, 80.0, 0.1)
    
    input_data = {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'pm25': pm25,
        'pm10': pm10,
        'no2': no2,
        'so2': so2,
        'co': co,
        'o3': o3
    }
    
    aqi_value = 50
    interpretation = predictor.interpret_aqi(aqi_value)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Input Values")
        
        weather_col, pollutant_col = st.columns(2)
        
        with weather_col:
            st.markdown("**Weather Conditions:**")
            st.markdown(f"- Temperature: {temperature}°C")
            st.markdown(f"- Humidity: {humidity}%")
            st.markdown(f"- Pressure: {pressure} hPa")
            st.markdown(f"- Wind Speed: {wind_speed} m/s")
        
        with pollutant_col:
            st.markdown("**Pollutant Concentrations:**")
            st.markdown(f"- PM2.5: {pm25} μg/m³")
            st.markdown(f"- PM10: {pm10} μg/m³")
            st.markdown(f"- NO2: {no2} μg/m³")
            st.markdown(f"- SO2: {so2} μg/m³")
            st.markdown(f"- CO: {co} μg/m³")
            st.markdown(f"- O3: {o3} μg/m³")
    
    with col2:
        st.subheader("🎯 Prediction")
        
        try:
            prediction, lower_bound, upper_bound = predictor.predict_with_confidence(input_data)
            aqi_value = to_scalar(prediction)
            
            interpretation = predictor.interpret_aqi(aqi_value)
            
            st.markdown(f'<div class="metric-container {interpretation["css_class"]}">', unsafe_allow_html=True)
            st.markdown(f"**AQI Value:** {aqi_value:.1f}")
            st.markdown(f"**Category:** {interpretation['category']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**Health Message:**")
            st.markdown(f"{interpretation['message']}")
            
            if lower_bound is not None and upper_bound is not None:
                lower_val = to_scalar(lower_bound)
                upper_val = to_scalar(upper_bound)
                st.markdown(f"**90% Confidence Interval:** [{lower_val:.1f}, {upper_val:.1f}]")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    st.subheader("📈 Visualizations")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Air Quality Index"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 300]},
            'bar': {'color': interpretation['color'].lower()},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    pollutant_data = pd.DataFrame({
        'Pollutant': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
        'Concentration': [pm25, pm10, no2, so2, co/10, o3],
        'Unit': ['μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³ (×10)', 'μg/m³']
    })
    
    fig_bar = px.bar(pollutant_data, x='Pollutant', y='Concentration', 
                     title='Current Pollutant Concentrations',
                     color='Concentration',
                     color_continuous_scale='Reds')
    
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    if hasattr(predictor.model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': predictor.feature_columns,
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title='Top 10 Feature Importances')
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.subheader("📊 Historical Trend Simulation")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                          end=datetime.now(), freq='D')
    
    np.random.seed(42)
    historical_aqi = np.random.normal(aqi_value, 20, len(dates))
    historical_aqi = np.clip(historical_aqi, 0, 300)
    
    historical_df = pd.DataFrame({
        'Date': dates,
        'AQI': historical_aqi
    })
    
    fig_historical = px.line(historical_df, x='Date', y='AQI', 
                            title='30-Day AQI Trend (Simulated)')
    fig_historical.add_hline(y=50, line_dash="dash", line_color="green", 
                            annotation_text="Good")
    fig_historical.add_hline(y=100, line_dash="dash", line_color="yellow", 
                            annotation_text="Moderate")
    fig_historical.add_hline(y=150, line_dash="dash", line_color="orange", 
                            annotation_text="Unhealthy for Sensitive")
    fig_historical.add_hline(y=200, line_dash="dash", line_color="red", 
                            annotation_text="Unhealthy")
    
    fig_historical.update_layout(height=400)
    st.plotly_chart(fig_historical, use_container_width=True)
    
    st.subheader("ℹ️ About Air Quality Index")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **AQI Categories:**
        - **Good (0-50):** Air quality is satisfactory
        - **Moderate (51-100):** Acceptable for most people
        - **Unhealthy for Sensitive Groups (101-150):** Sensitive individuals may experience health effects
        - **Unhealthy (151-200):** Everyone may experience health effects
        - **Very Unhealthy (201-300):** Health alert level
        - **Hazardous (301+):** Emergency conditions
        """)
    
    with info_col2:
        st.markdown("""
        **Key Pollutants:**
        - **PM2.5:** Fine particulate matter
        - **PM10:** Coarse particulate matter
        - **NO2:** Nitrogen dioxide
        - **SO2:** Sulfur dioxide
        - **CO:** Carbon monoxide
        - **O3:** Ground-level ozone
        """)
    
    st.markdown("---")
    st.markdown("**Note:** This is a predictive model for demonstration purposes. For official air quality information, please consult your local environmental authority.")

if __name__ == "__main__":
    main()
