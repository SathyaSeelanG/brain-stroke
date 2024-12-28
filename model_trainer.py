import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import json
import os
from datetime import datetime
import logging
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        
        logging.basicConfig(
            filename='model_training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def train_models(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():
            try:
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Calculate metrics
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Save the model
                self.save_model(model, name)
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Save results
        self.save_results(results)
        return results
    
    def save_model(self, model, name):
        os.makedirs('models', exist_ok=True)
        with open(f'models/{name}.pickle', 'wb') as f:
            pickle.dump(model, f)
    
    def save_results(self, results):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'results/training_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4) 

def train_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(max_iter=200),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(probability=True, kernel='linear'),
        'gradient_boosting': GradientBoostingClassifier(),
        'knn': KNeighborsClassifier(),
        # Add Neural Network model if needed
    }
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    total_models = len(models)
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}... ({i + 1}/{total_models})")
        
        # Create a pipeline to scale the data and fit the model
        pipeline = make_pipeline(StandardScaler(), model)
        
        # Start training and measure time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        
        # Save the trained model
        with open(f'models/{name}.pickle', 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Print training completion message
        print(f"{name} trained successfully in {end_time - start_time:.2f} seconds.")   