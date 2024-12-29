import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import pickle
import os
import json
from datetime import datetime
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_and_evaluate():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = pd.read_csv('stroke_prediction_dataset.csv')
    
    # Drop identifier columns
    data = data.drop(['Patient ID', 'Patient Name'], axis=1)
    
    # Handle categorical variables
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    
    # Save label encoders
    with open('models/label_encoders.pickle', 'wb') as f:
        pickle.dump({col: le for col in categorical_columns}, f)
    
    # Split features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Save feature names
    with open('models/feature_names.pickle', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE for balancing the dataset
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Define models with optimized hyperparameters
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=20,
            random_state=42,
            class_weight='balanced'
        ),
        'catboost': CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False,
            class_weights=[1, 2]  # Adjust class weights for imbalanced data
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            activation='relu',
            solver='adam',
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski'
        ),
        'svm': SVC(
            kernel='rbf',
            C=10.0,
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    }
    
    # Train and evaluate models
    results = {}
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform GridSearchCV for hyperparameter tuning
        if name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train_balanced)
            model = grid_search.best_estimator_
        elif name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'learning_rate': [0.01, 0.1]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train_balanced)
            model = grid_search.best_estimator_
        elif name == 'catboost':
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train_balanced)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train_scaled, y_train_balanced)
        
        # Save model
        with open(f'models/{name}.pickle', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=5)
        
        results[name] = {
            'accuracy': model.score(X_test_scaled, y_test),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'visualizations/roc_curve_{name}.png')
        plt.close()
        
        # Feature importance for applicable models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feat_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feat_importance.head(10))
            plt.title(f'Top 10 Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(f'visualizations/feature_importance_{name}.png')
            plt.close()
    
    # Save results
    with open('results/model_evaluation.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison visualizations
    accuracies = {name: results[name]['accuracy'] for name in models.keys()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    
    print("\nTraining complete! Models and results have been saved.")
    return results

if __name__ == "__main__":
    train_and_evaluate()
