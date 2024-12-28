from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import numpy as np
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Create required directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)

# Function to load models
def load_models():
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'svm', 'gradient_boosting', 'knn', 'neural_network']
    
    try:
        # Load scaler and feature names first
        with open('models/scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pickle', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load models
        for name in model_names:
            model_path = f'models/{name}.pickle'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {name} not found. Please run train_and_evaluate_models.py first.")
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
        
        return models, scaler, feature_names
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

# Load models, scaler, and feature names
models, scaler, feature_names = load_models()

if models is None or scaler is None or feature_names is None:
    print("Error: Required files not found. Please run train_and_evaluate_models.py first.")
    exit(1)

# Copy visualization files to static directory
def copy_visualizations():
    for filename in os.listdir('visualizations'):
        src = os.path.join('visualizations', filename)
        dst = os.path.join('static/visualizations', filename)
        if os.path.isfile(src):
            with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                f_dst.write(f_src.read())

copy_visualizations()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            # Get selected model
            selected_model = request.form.get('model', 'random_forest')
            model = models[selected_model]
            
            # Create feature dictionary matching the training data columns
            features = {
                'Gender': 1 if request.form.get('gender') == 'Male' else 0,
                'Age': float(request.form['age']),
                'Hypertension': int(request.form['hypertension']),
                'Heart_Disease': int(request.form['disease']),
                'Ever_Married': 1 if request.form.get('married') == 'Yes' else 0,
                'Work_Type': int(request.form.get('work_type', 0)),
                'Residence_Type': 1 if request.form.get('residence') == 'Urban' else 0,
                'Average_Glucose_Level': float(request.form['avg_glucose_level']),
                'BMI': float(request.form['bmi']),
                'Smoking_Status': int(request.form.get('smoking_status', 0))
            }
            
            # Create feature array with correct column names
            input_features = pd.DataFrame([features])
            
            # Ensure columns match training data
            input_features = input_features[feature_names]
            
            # Scale features
            input_scaled = scaler.transform(input_features)
            
            # Generate prediction and report
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Generate detailed report
            report = generate_report(features, prediction_proba[1], selected_model)
            
            # Add visualizations to the report
            report['visualizations'] = {
                'roc_curve': f'visualizations/roc_curve_{selected_model}.png',
                'feature_importance': f'visualizations/feature_importance_{selected_model}.png'
                if os.path.exists(f'static/visualizations/feature_importance_{selected_model}.png') else None
            }
            
            return render_template('result.html',
                                prediction_text=report['summary'],
                                detailed_report=report)
            
        except Exception as e:
            return render_template('index.html',
                                prediction_text=f'Error in prediction: {str(e)}')
    
    return render_template('index.html')

def generate_report(features, risk_percentage, model_name):
    risk_percentage = round(risk_percentage * 100, 2)
    
    report = {
        'summary': f'Stroke Risk Assessment ({model_name})',
        'risk_percentage': risk_percentage,
        'risk_level': 'High' if risk_percentage > 50 else 'Moderate' if risk_percentage > 20 else 'Low',
        'features': features,
        'recommendations': generate_recommendations(risk_percentage, features),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report

def generate_recommendations(risk_percentage, features):
    recommendations = []
    
    if risk_percentage > 50:
        recommendations.append("Immediate medical consultation is strongly advised.")
    
    if features.get('hypertension') == 1:
        recommendations.append("Regular blood pressure monitoring recommended.")
    
    if features.get('bmi') > 25:
        recommendations.append("Consider lifestyle modifications for weight management.")
    
    return recommendations

if __name__ == "__main__":
    app.run(debug=True, port=5000)
