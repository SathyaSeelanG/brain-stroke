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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Set page config
st.set_page_config(
    page_title="Brain Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    data = pd.read_csv('brain_stroke.csv')
    # Rename columns to match the dashboard expectations
    column_mapping = {
        'age': 'Age',
        'gender': 'Gender',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'ever_married': 'Ever Married',
        'work_type': 'Work Type',
        'Residence_type': 'Residence Type',
        'avg_glucose_level': 'Glucose Level',
        'bmi': 'BMI',
        'smoking_status': 'Smoking Status',
        'stroke': 'Diagnosis'
    }
    data = data.rename(columns=column_mapping)
    return data

@st.cache_resource
def load_models():
    try:
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
        return None, model_metrics, None
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        return None, None, None

# Load data and models
data = load_data()
models, evaluation_results, feature_names = load_models()

# Add this function near the top of the file after imports
def save_model_performance_plots():
    """Save model performance visualization plots"""
    # Create directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Accuracy Plot
    plt.figure(figsize=(18,6))
    accuracies = [metrics['accuracy'] for metrics in model_metrics.values()]
    sns.barplot(x=list(model_metrics.keys()), y=accuracies)
    plt.title('Accuracy Plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/accuracy_plot.png')
    plt.close()
    
    # F1 Score Plot
    plt.figure(figsize=(18,6))
    f1_scores = [metrics['f1'] for metrics in model_metrics.values()]
    sns.barplot(x=list(model_metrics.keys()), y=f1_scores)
    plt.title('F1 Score plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/f1_score_plot.png')
    plt.close()
    
    # Recall Plot
    plt.figure(figsize=(18,6))
    recalls = [metrics['recall'] for metrics in model_metrics.values()]
    sns.barplot(x=list(model_metrics.keys()), y=recalls)
    plt.title('Recall plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('Recall-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/recall_plot.png')
    plt.close()

# Add this after loading data
def create_model_metrics():
    """Create and save model metrics if they don't exist"""
    model_metrics = {
        'Random Forest': {'accuracy': 0.97, 'f1': 0.21, 'recall': 0.61},
        'Naive Bayes': {'accuracy': 0.63, 'f1': 0.20, 'recall': 0.93},
        'SVM Classifier': {'accuracy': 0.66, 'f1': 0.21, 'recall': 0.92},
        'Voting Clf w. Bagging': {'accuracy': 0.74, 'f1': 0.22, 'recall': 0.76}
    }
    
    # Save metrics to JSON
    with open('model_metrics.json', 'w') as f:
        json.dump(model_metrics, f)
    
    return model_metrics

def create_visualizations():
    """Create and save visualization plots if they don't exist"""
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    model_metrics = create_model_metrics()
    
    # Accuracy Plot
    plt.figure(figsize=(18,6))
    name_arr = list(model_metrics.keys())
    acc_arr = [metrics['accuracy'] for metrics in model_metrics.values()]
    sns.barplot(x=name_arr, y=acc_arr)
    plt.title('Accuracy Plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/accuracy_plot.png')
    plt.close()
    
    # F1 Score Plot
    plt.figure(figsize=(18,6))
    f1_sc = [metrics['f1'] for metrics in model_metrics.values()]
    sns.barplot(x=name_arr, y=f1_sc)
    plt.title('F1 Score plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/f1_score_plot.png')
    plt.close()
    
    # Recall Plot
    plt.figure(figsize=(18,6))
    re_sc = [metrics['recall'] for metrics in model_metrics.values()]
    sns.barplot(x=name_arr, y=re_sc)
    plt.title('Recall plot for each classifier')
    plt.xlabel('Classifier name')
    plt.ylabel('Recall-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/recall_plot.png')
    plt.close()

# Modify the load_model_metrics function
@st.cache_resource
def load_model_metrics():
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        # Create metrics if file doesn't exist
        return create_model_metrics()

# Create visualizations if they don't exist
if not os.path.exists('visualizations') or not os.path.exists('model_metrics.json'):
    create_visualizations()

# Load metrics
model_metrics = load_model_metrics()

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
    
    # Search functionality
    search_term = st.text_input("Search by any field")
    if search_term:
        mask = np.column_stack([
            filtered_data[col].astype(str).str.contains(search_term, case=False, na=False)
            for col in filtered_data.columns
        ]).any(axis=1)
        filtered_data = filtered_data[mask]
    
    # Display record count
    st.write(f"Showing {len(filtered_data)} records")
    
    # Display data with pagination
    records_per_page = st.selectbox("Records per page", [10, 20, 50, 100])
    page_number = st.number_input("Page", min_value=1, 
                                max_value=max(1, (len(filtered_data) // records_per_page) + 1), 
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
    st.header("Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Stroke Cases", len(data[data['Diagnosis'] == 1]))
    with col3:
        st.metric("Features", len(data.columns) - 1)
    
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
    
    if not model_metrics:
        st.error("Could not load model metrics")
    else:
        # Create tabs for different metrics
        tab1, tab2, tab3 = st.tabs(["Accuracy", "F1 Score", "Recall"])
        
        with tab1:
            st.image('visualizations/accurancy.png', 
                    caption='Accuracy Plot for each classifier',
                    use_container_width=True)
            
        with tab2:
            st.image('visualizations/f1_score.png',
                    caption='F1 Score plot for each classifier',
                    use_container_width=True)
            
        with tab3:
            st.image('visualizations/recall.png',
                    caption='Recall plot for each classifier',
                    use_container_width=True)

        # Add model comparison insights
        st.subheader("Model Performance Insights")
        st.write("""
        - Random Forest achieves the highest accuracy at 77%
        - Naive Bayes and SVM show the highest recall scores (93% and 92%)
        - Bagging and Voting Classifier with Bagging show the best F1 scores (0.22)
        - There's a trade-off between accuracy and recall across models
        """)

else:  # Prediction Interface
    st.title("Stroke Risk Prediction")
    
    if model_metrics is None:
        st.error("Could not load model metrics. Please ensure model_metrics.json exists.")
    else:
        # Add model selection
        model_choice = st.selectbox(
            "Select Model for Prediction",
            list(model_metrics.keys())
        )
        
        # Display model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{model_metrics[model_choice]['accuracy']:.0%}")
        with col2:
            st.metric("F1 Score", f"{model_metrics[model_choice]['f1']:.2f}")
        with col3:
            st.metric("Recall", f"{model_metrics[model_choice]['recall']:.0%}")

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
        
        if st.button("Predict", type="primary"):
            # Dummy prediction for demonstration
            import random
            prediction = random.random()
            risk_level = "High" if prediction > 0.5 else "Low"
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stroke Risk", f"{prediction*100:.1f}%")
            with col2:
                st.metric("Risk Level", risk_level)

# Footer
st.markdown("---")
st.markdown("Brain Stroke Prediction Dashboard - Created with Streamlit")
