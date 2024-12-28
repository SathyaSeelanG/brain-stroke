import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Brain Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    data = pd.read_csv('stroke_prediction_dataset.csv')
    return data

@st.cache_resource
def load_models():
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'gradient_boosting', 'knn', 'neural_network','svm']
    
    try:
        for name in model_names:
            with open(f'models/{name}.pickle', 'rb') as f:
                models[name] = pickle.load(f)
        
        with open('results/model_evaluation.json', 'r') as f:
            evaluation_results = json.load(f)
        
        # Load feature names
        with open('models/feature_names.pickle', 'rb') as f:
            feature_names = pickle.load(f)
        
        return models, evaluation_results, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Load data and models
data = load_data()
models, evaluation_results, feature_names = load_models()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Patient Records", "Dataset Overview", "Model Performance", "Prediction Interface"])

if page == "Patient Records":
    st.title("Patient Records")
    
    # Add search and filter functionality
    st.sidebar.header("Filters")
    
    # Age filter
    age_range = st.sidebar.slider("Age Range", 
                                int(data['Age'].min()), 
                                int(data['Age'].max()), 
                                (int(data['Age'].min()), int(data['Age'].max())))
    
    # Gender filter
    gender_filter = st.sidebar.multiselect("Gender", 
                                         data['Gender'].unique().tolist(),
                                         default=data['Gender'].unique().tolist())
    
    # Apply filters
    filtered_data = data[
        (data['Age'].between(age_range[0], age_range[1])) &
        (data['Gender'].isin(gender_filter))
    ]
    
    # Search by ID or Name
    search_term = st.text_input("Search by Patient ID or Name")
    if search_term:
        filtered_data = filtered_data[
            (filtered_data['Patient ID'].astype(str).str.contains(search_term, case=False)) |
            (filtered_data['Patient Name'].str.contains(search_term, case=False))
        ]
    
    # Display record count
    st.write(f"Showing {len(filtered_data)} records")
    
    # Display data with pagination
    records_per_page = st.selectbox("Records per page", [10, 20, 50, 100])
    page_number = st.number_input("Page", min_value=1, 
                                max_value=(len(filtered_data) // records_per_page) + 1, 
                                value=1)
    
    start_idx = (page_number - 1) * records_per_page
    end_idx = start_idx + records_per_page
    
    # Display the data
    st.dataframe(filtered_data.iloc[start_idx:end_idx], use_container_width=True)
    
    # Export functionality
    if st.button("Export to CSV"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="patient_records.csv",
            mime="text/csv"
        )

elif page == "Dataset Overview":
    st.title("Brain Stroke Dataset Analysis")
    
    # Dataset info with enhanced metrics
    st.header("Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        stroke_cases = len(data[data['Diagnosis'] == 1])
        st.metric("Stroke Cases", stroke_cases)
    with col3:
        stroke_rate = (stroke_cases / len(data)) * 100
        st.metric("Stroke Rate", f"{stroke_rate:.1f}%")
    with col4:
        st.metric("Features", len(data.columns) - 3)
    
    # Feature Distribution
    st.header("Feature Distributions")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_feature = st.selectbox("Select Feature", numeric_cols)
    
    # Distribution plot with stroke/no-stroke separation
    fig_dist = px.histogram(data, x=selected_feature, color='Diagnosis',
                          title=f"Distribution of {selected_feature}",
                          barmode='overlay')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # 3D scatter plot with enhanced interactivity
    st.header("3D Feature Relationship")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols)
    with col3:
        z_col = st.selectbox("Z-axis", numeric_cols)
    
    fig_3d = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                          color='Diagnosis',
                          title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}")
    fig_3d.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Correlation Matrix with enhanced visualization
    st.header("Feature Correlations")
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr_matrix,
                        title="Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    
    # Model metrics comparison
    metrics_tab, curves_tab = st.tabs(["Performance Metrics", "ROC Curves"])
    
    with metrics_tab:
        # Prepare metrics data
        metrics_data = []
        for model_name, results in evaluation_results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['classification_report']['1']['precision'],
                'Recall': results['classification_report']['1']['recall'],
                'F1-Score': results['classification_report']['1']['f1-score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Plot metrics comparison
        fig_metrics = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig_metrics.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig_metrics.update_layout(
            title="Model Performance Metrics Comparison",
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with curves_tab:
        # Display ROC curves
        for model_name in models.keys():
            img_path = f'visualizations/roc_curve_{model_name}.png'
            if os.path.exists(img_path):
                st.image(img_path, caption=f"ROC Curve - {model_name}")
    
    # Feature importance comparison
    st.header("Feature Importance Analysis")
    feature_importance_data = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Get feature names excluding 'Patient ID', 'Patient Name', and 'Diagnosis'
            feature_cols = [col for col in data.columns if col not in ['Patient ID', 'Patient Name', 'Diagnosis']]
            
            # Create feature importance series with matching lengths
            feature_importance_data[model_name] = pd.Series(
                model.feature_importances_,
                index=feature_cols
            )
    
    if feature_importance_data:
        selected_model = st.selectbox(
            "Select Model for Feature Importance",
            list(feature_importance_data.keys())
        )
        
        fig_importance = px.bar(
            x=feature_importance_data[selected_model].values,
            y=feature_importance_data[selected_model].index,
            orientation='h',
            title=f"Feature Importance - {selected_model}"
        )
        fig_importance.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

else:  # Prediction Interface
    st.title("Stroke Risk Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    
    with col2:
        glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
        work_type = st.selectbox("Work Type", 
                               ["Private", "Self-employed", "Government", "Never worked", "Children"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking = st.selectbox("Smoking Status", 
                             ["Never smoked", "Formerly smoked", "Smokes"])
    
    # Model selection with confidence score display
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Model", list(models.keys()))
    with col2:
        st.info(f"Model Accuracy: {evaluation_results[model_choice]['accuracy']:.2%}")
    
    if st.button("Predict", type="primary"):
        try:
            # Prepare input data matching the feature names used during training
            input_data = {
                'Age': age,
                'Gender': 1 if gender == "Male" else 0,
                'Hypertension': int(hypertension),
                'Heart Disease': int(heart_disease),
                'Marital Status': 1 if ever_married == "Yes" else 0,
                'Work Type': ["Private", "Self-employed", "Government", "Never worked", "Children"].index(work_type),
                'Residence Type': 1 if residence == "Urban" else 0,
                'Average Glucose Level': glucose_level,
                'Body Mass Index (BMI)': bmi,
                'Smoking Status': ["Never smoked", "Formerly smoked", "Smokes"].index(smoking),
                'Alcohol Intake': 0,  # Default values for additional features
                'Physical Activity': 1,
                'Stroke History': 0,
                'Family History of Stroke': 0,
                'Dietary Habits': 1,
                'Stress Levels': 1,
                'Blood Pressure Levels': 120,
                'Cholesterol Levels': 180,
                'Symptoms': 0
            }
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names]  # Ensure correct column order
            
            # Scale features
            with open('models/scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            model = models[model_choice]
            prediction = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.header("Prediction Results")
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stroke Risk", f"{prediction[1]*100:.1f}%")
            with col2:
                confidence = max(prediction[0], prediction[1])
                st.metric("Confidence Score", f"{confidence*100:.1f}%")
            with col3:
                risk_level = "High" if prediction[1] > 0.5 else "Moderate" if prediction[1] > 0.2 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction[1]*100,
                title={'text': "Stroke Risk Percentage"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
            
            # Risk factors analysis
            if prediction[1] > 0.2:
                st.subheader("Key Risk Factors")
                risk_factors = []
                if age > 65:
                    risk_factors.append("Advanced age")
                if hypertension:
                    risk_factors.append("Hypertension")
                if heart_disease:
                    risk_factors.append("Heart disease")
                if glucose_level > 126:
                    risk_factors.append("High glucose level")
                if bmi > 30:
                    risk_factors.append("Obesity")
                if smoking == "Smokes":
                    risk_factors.append("Active smoking")
                
                for factor in risk_factors:
                    st.warning(factor)
                
                st.info("Please consult with a healthcare professional for a thorough evaluation.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure all required features are provided correctly.")

# Footer
st.markdown("---")
st.markdown("Brain Stroke Prediction Dashboard - Created with Streamlit")
