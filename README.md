# air-quality-predictor

Predicts Air Quality Index (AQI) using weather and pollution data with machine learning. 
Includes synthetic data generation, model training, and a Streamlit app for interactive AQI predictions and visualizations.

## Features

- Generate synthetic or collect real-world data for model training
- Train a Random Forest model to predict AQI from environmental features
- Feature engineering and importance analysis
- Interactive web dashboard (Streamlit) for user inputs and result visualization

<img width="1919" height="868" alt="Screenshot 2025-08-16 185434" src="https://github.com/user-attachments/assets/4073954d-2d47-414a-86ba-db283f495260" />
<img width="704" height="400" alt="newplot" src="https://github.com/user-attachments/assets/2d9ab88a-c000-4db4-8ad6-82904ca368de" />
<img width="1512" height="637" alt="newplot (1)" src="https://github.com/user-attachments/assets/9765f170-ea3b-4900-88b4-f46e6800df5d" />
<img width="704" height="400" alt="newplot (2)" src="https://github.com/user-attachments/assets/0f4210a4-c2f4-4577-8583-1198bc01f24f" />
<img width="704" height="400" alt="newplot (3)" src="https://github.com/user-attachments/assets/186064a5-47f1-4d18-b740-0fedb0c740e0" />
<img width="1919" height="866" alt="Screenshot 2025-08-16 185525" src="https://github.com/user-attachments/assets/f8583196-38b7-4a3b-9050-f3d9fb017f22" />

# Installation

 1. Clone the repository:
git clone https://github.com/theguyingreyhoodie/air-quality-predictor.git
cd air-quality-predictor

 2. (Optional) Create and activate a virtual environment.

 3. Install dependencies:
pip install -r requirements.txt

 4. (Optional, for real data) Set your OpenWeatherMap API key:
 On Windows:
set OWM_API_KEY=your_key_here 
 On Mac/Linux:
export OWM_API_KEY=your_key_here

## Usage

1. Generate data (synthetic or real):
python scripts/data_collector.py

2. Train the model:
python scripts/train_model.py

3. Launch dashboard:
streamlit run app/streamlit_app.py

