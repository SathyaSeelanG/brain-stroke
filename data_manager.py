import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import logging
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class DataManager:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        self.allowed_extensions = {'csv', 'xlsx', 'xls'}
        os.makedirs(upload_folder, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename='data_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_uploaded_file(self, file):
        try:
            if file and self.allowed_file(file.filename):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"data_{timestamp}_{file.filename}"
                filepath = os.path.join(self.upload_folder, filename)
                file.save(filepath)
                logging.info(f"File saved successfully: {filepath}")
                return filepath
            return None
        except Exception as e:
            logging.error(f"Error saving file: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        try:
            # Remove specified columns
            if 'Patient ID' in data.columns:
                data = data.drop(['Patient ID'], axis=1)
            if 'Patient Name' in data.columns:
                data = data.drop(['Patient Name'], axis=1)
            
            # Handle missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
            
            # Handle categorical variables
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])
                data = pd.get_dummies(data, columns=[col], drop_first=True)
            
            # Scale numeric features
            scaler = StandardScaler()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            
            logging.info("Data preprocessing completed successfully")
            return data, scaler
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def validate_data(self, data):
        required_columns = ['age', 'gender', 'hypertension', 'heart_disease', 
                          'ever_married', 'work_type', 'Residence_type', 
                          'avg_glucose_level', 'bmi', 'smoking_status']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        if data['age'].min() < 0 or data['age'].max() > 120:
            raise ValueError("Age values out of valid range (0-120)")
        
        if data['bmi'].min() < 10 or data['bmi'].max() > 60:
            raise ValueError("BMI values out of valid range (10-60)")
        
        return True 

def train_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        # Add other models here
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        with open(f'models/{name}.pickle', 'wb') as f:
            pickle.dump(model, f)

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance')
    plt.show()