# Brain Stroke Prediction System

## Project Overview
The Brain Stroke Prediction System is a machine learning project aimed at predicting the risk of brain strokes in patients based on various health and lifestyle factors. The system utilizes machine learning algorithms to analyze patient data and provide insights that can assist healthcare professionals in making informed decisions.

## What We Are Doing in This Project
In this project, we are developing a predictive model that leverages machine learning techniques to assess the likelihood of a patient experiencing a brain stroke. The key objectives include:

1. **Data Collection and Preprocessing**: 
   - We use a curated dataset containing patient demographics, medical history, and lifestyle factors
   - The data undergoes cleaning and preprocessing to handle missing values and encode categorical variables

2. **Model Development**: 
   - We implement machine learning algorithms to train models on the processed dataset
   - Each model is evaluated based on performance metrics such as accuracy, precision, recall, and F1-score

3. **Dashboard Development**: 
   - We create an interactive dashboard using Streamlit that allows users to:
     - View dataset statistics and visualizations
     - Make stroke predictions using the trained model
     - Explore feature relationships and importance

## Key Components

### 1. Files Structure
```
Brain_Stroke_Prediction/
├── prediction.ipynb          # Model training and evaluation notebook
├── dashboard.py             # Streamlit dashboard
├── requirements.txt         # Project dependencies
├── brain_stroke.csv        # Dataset
└── model_metrics.json      # Model performance metrics
```

### 2. Features Used
- Age
- Gender
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

### 3. System Components
- **Analytics Dashboard** (`dashboard.py`): Interactive data visualization and prediction interface
- **Model Training** (`prediction.ipynb`): Data preprocessing, model training and evaluation

### 4. Installation and Usage

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```     
    or try this
   ```bash
      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      streamlit run dashboard.py
   ```

### 5. Data Processing Pipeline
1. **Data Cleaning**: Handle missing values, encode categorical variables
2. **Feature Engineering**: Process numerical and categorical features
3. **Model Training**: Train and evaluate the model performance

### 6. Best Practices
1. Regular model retraining with new data
2. Data validation before prediction
3. Regular performance monitoring
4. Backup of trained models

### 7. Future Improvements
1. Additional model integration
2. Enhanced visualization features
3. Model performance optimization
4. Extended feature engineering
5. API development

## Dataset Usage

The dataset (`brain_stroke.csv`) contains various patient attributes used for training the model. The data is processed using pandas and scikit-learn libraries for model training and evaluation.

For detailed implementation and model training process, please refer to the `prediction.ipynb` notebook.

## Technical Implementation Details

### Data Processing and Model Training (prediction.ipynb)
The `prediction.ipynb` notebook contains the complete model training pipeline. Here are the key steps:

1. **Data Loading and Initial Exploration**:
```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('brain_stroke.csv')

# Display basic information
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
```

2. **Data Preprocessing**:
```python
# Handle categorical variables
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
```

3. **Model Training and Evaluation**:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the data
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Dashboard Implementation (dashboard.py)
The Streamlit dashboard provides an interactive interface for data visualization and predictions. Here are key components:

1. **Dashboard Setup**:
```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")
st.title("Brain Stroke Prediction System")
```

2. **Data Visualization**:
```python
def plot_feature_distribution(df, feature):
    fig = px.histogram(df, x=feature, color='stroke',
                      title=f'Distribution of {feature} by Stroke Status')
    st.plotly_chart(fig)

# Usage in dashboard
selected_feature = st.selectbox("Select Feature", ['age', 'bmi', 'avg_glucose_level'])
plot_feature_distribution(df, selected_feature)
```

3. **Prediction Interface**:
```python
def make_prediction():
    # Get user inputs
    age = st.number_input("Age", min_value=0, max_value=120)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
    glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0)
    
    # Process inputs and make prediction
    if st.button("Predict"):
        input_data = preprocess_input(age, bmi, glucose)
        prediction = model.predict(input_data)
        st.write("Prediction:", "High Risk" if prediction[0] == 1 else "Low Risk")
```

### Key Features of the Dashboard:
1. **Data Exploration**:
   - Interactive visualizations of feature distributions
   - Correlation analysis between different health factors
   - Summary statistics of the dataset

2. **Prediction System**:
   - User-friendly input form for patient data
   - Real-time predictions using the trained model
   - Confidence scores for predictions

3. **Model Insights**:
   - Feature importance visualization
   - Model performance metrics
   - Distribution of predictions

### Running the Project
To run the complete project:

1. First, train the model using the notebook:
```bash
jupyter notebook prediction.ipynb
```

2. Then launch the dashboard:
```bash
streamlit run dashboard.py
```

The dashboard will be accessible at `http://localhost:8501` by default.

