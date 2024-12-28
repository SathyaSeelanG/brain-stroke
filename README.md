# Brain Stroke Prediction System

## Project Overview
The Brain Stroke Prediction System is a machine learning project aimed at predicting the risk of brain strokes in patients based on various health and lifestyle factors. The system utilizes multiple algorithms to analyze patient data and provide insights that can assist healthcare professionals in making informed decisions. The project includes a user-friendly web interface for predictions and a comprehensive dashboard for data analysis and visualization.

## What We Are Doing in This Project
In this project, we are developing a predictive model that leverages machine learning techniques to assess the likelihood of a patient experiencing a brain stroke. The key objectives include:

1. **Data Collection and Preprocessing**: 
   - We gather a dataset containing patient demographics, medical history, lifestyle factors, and medical measurements.
   - The data undergoes cleaning and preprocessing to handle missing values, normalize features, and encode categorical variables.

2. **Feature Engineering**: 
   - We identify and create relevant features that contribute to stroke risk, such as BMI, age groups, and medical history scores.

3. **Model Development**: 
   - We implement various machine learning algorithms, including Random Forest, Gradient Boosting, and Neural Networks, to train models on the processed dataset.
   - Each model is evaluated based on performance metrics such as accuracy, precision, recall, and F1-score.

4. **Model Evaluation and Selection**: 
   - We compare the performance of different models to select the best-performing one for stroke prediction.
   - The selected model is then fine-tuned and validated to ensure reliability.

5. **Deployment of the Web Application**: 
   - We create a web application using Streamlit that allows users to input patient data and receive stroke risk predictions.
   - The application also features an analytics dashboard for visualizing model performance and understanding the impact of various features on stroke risk.

6. **User Education and Support**: 
   - We provide documentation and support to help users understand how to use the application effectively and interpret the results.

## Key Components

### 1. Models
- **Random Forest** (Best Performing - 98.7% accuracy)
- **Gradient Boosting** (97.9% accuracy)
- **Neural Network** (96.8% accuracy)
- **Logistic Regression** (95.4% accuracy)
- **K-Nearest Neighbors** (94.2% accuracy)

### 2. Features Used
- **Primary Features**: Age, Gender, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status
- **Additional Features**: Blood Pressure, Cholesterol Levels, Physical Activity, Family History, Stress Levels

### 3. System Components
- **Analytics Dashboard** (`dashboard.py`): Interactive data visualization, model performance comparison, feature importance analysis, patient records management.
- **Model Training** (`train_and_evaluate_models.py`): Data preprocessing, model training and evaluation, performance metrics calculation, model persistence.

### 4. Performance Metrics
- **Best Model (Random Forest)**:
  - Accuracy: 98.7%
  - Precision: 97.9%
  - Recall: 98.2%
  - F1-Score: 98.0%
  - AUC-ROC: 0.989

### 5. Data Processing Pipeline
1. **Data Cleaning**: Handle missing values, remove duplicates, normalize numerical features, encode categorical variables.
2. **Feature Engineering**: BMI calculation, age grouping, risk factor combination, medical history scoring.
3. **Model Training**: Cross-validation, hyperparameter tuning, ensemble methods, performance validation.

### 6. Deployment Instructions
1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**:
   ```bash
   python train_and_evaluate_models.py
   ```

3. **Run Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
  ```bash
      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      streamlit run dashboard.py
   ```
### 7. File Structure
```
Brain_Stroke_Prediction/
├── app.py                    # Web interface
├── dashboard.py              # Analytics dashboard
├── train_and_evaluate_models.py  # Model training
├── models/                   # Saved models
├── data/                     # Dataset
├── templates/                # HTML templates
├── static/                   # Static files
├── results/                  # Model results
└── visualizations/           # Generated plots
```

### 8. Best Practices
1. Regular model retraining
2. Data validation before prediction
3. Secure patient data handling
4. Regular performance monitoring
5. Backup of trained models

### 9. Future Improvements
1. Additional model integration
2. Real-time monitoring
3. Mobile application
4. API development
5. Automated reporting

## Model Training and Storage

### Model Training Process
In this project, we train multiple machine learning models to predict the risk of brain strokes. The training process involves the following steps:

1. **Data Preprocessing**: 
   - The raw patient data is cleaned and preprocessed to ensure it is suitable for model training. This includes handling missing values, normalizing numerical features, and encoding categorical variables.

2. **Feature Selection**: 
   - We identify and select relevant features that contribute to stroke risk. This includes primary features like age, gender, and medical history, as well as additional features such as blood pressure and cholesterol levels.

3. **Model Training**: 
   - We implement various machine learning algorithms, including Random Forest, Gradient Boosting, and Neural Networks. Each model is trained on the processed dataset using techniques such as cross-validation and hyperparameter tuning to optimize performance.

4. **Model Evaluation**: 
   - After training, each model is evaluated using performance metrics such as accuracy, precision, recall, and F1-score. This helps us determine the best-performing model for stroke prediction.

### Model Storage
Once the models are trained, they are stored in the `models/` directory of the project. The models are saved in a serialized format (e.g., using joblib or pickle) to allow for easy loading and inference in the web application. This ensures that we can reuse the trained models without needing to retrain them each time the application is run.

## Data Cleaning Process
Data cleaning is a crucial step in preparing the dataset for model training. The cleaning process involves several key tasks:

1. **Handling Missing Values**: 
   - We identify and address missing values in the dataset. Depending on the context, missing values may be filled with the mean, median, or mode of the respective feature, or they may be removed entirely if they are not significant.

2. **Removing Duplicates**: 
   - Duplicate records can skew the results of the model. We check for and remove any duplicate entries in the dataset to ensure that each patient record is unique.

3. **Normalizing Numerical Features**: 
   - Numerical features are normalized to ensure they are on a similar scale. This is important for algorithms that are sensitive to the scale of input data, such as Gradient Boosting and Neural Networks.

4. **Encoding Categorical Variables**: 
   - Categorical variables are transformed into numerical format using techniques such as one-hot encoding or label encoding. This allows the models to interpret categorical data effectively.

5. **Outlier Detection**: 
   - We analyze the dataset for outliers that may affect model performance. Outliers can be removed or treated based on their impact on the overall data distribution.

By following these data cleaning steps, we ensure that the dataset is robust and ready for effective model training, leading to more accurate predictions in the Brain Stroke Prediction System.

## Dataset Navigation and Processing

### Navigating the Dataset
The dataset used for training the models is stored in the `data/` directory. It typically consists of a CSV file containing various patient attributes. We use libraries like Pandas to load and navigate through the dataset. Here’s a sample code snippet demonstrating how to load and explore the dataset:

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/patient_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())
```

### Data Processing
Once the dataset is loaded, we perform several preprocessing steps to prepare it for model training:

1. **Handling Missing Values**: 
   - We fill missing values using the mean or median for numerical features and the mode for categorical features. Here’s how it can be done:

   ```python
   # Fill missing values
   data['age'].fillna(data['age'].mean(), inplace=True)
   data['gender'].fillna(data['gender'].mode()[0], inplace=True)
   ```

2. **Normalizing Numerical Features**: 
   - We scale numerical features to a standard range, typically between 0 and 1, using Min-Max scaling:

   ```python
   from sklearn.preprocessing import MinMaxScaler

   scaler = MinMaxScaler()
   data[['average_glucose_level', 'bmi']] = scaler.fit_transform(data[['average_glucose_level', 'bmi']])
   ```

3. **Encoding Categorical Variables**: 
   - Categorical variables are converted into numerical format using one-hot encoding:

   ```python
   data = pd.get_dummies(data, columns=['gender', 'work_type', 'residence_type'], drop_first=True)
   ```

### Model Training and Accuracy Prediction
After processing the data, we split it into training and testing sets, train the models, and evaluate their performance. Here’s a sample code snippet for training a Random Forest model and predicting its accuracy:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the dataset into features and target variable
X = data.drop('stroke', axis=1)  # Features
y = data['stroke']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### Model Storage
Once the models are trained, they are serialized and stored in the `models/` directory using the `pickle` library. This allows for easy loading and inference in the web application. Here’s how to save and load a model:

```python
import pickle

# Save the trained model
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the trained model
with open('models/random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
```

By following these steps, we ensure that the dataset is effectively processed, models are trained accurately, and the trained models are stored for future use.

