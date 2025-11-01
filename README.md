Hazardous Asteroid Prediction using Machine Learning
This project aims to classify asteroids as Hazardous or Non-Hazardous using various Machine Learning (ML) models. The system analyzes asteroid data from NASA’s Jet Propulsion Laboratory (JPL) to enhance early detection and improve planetary defense strategies.

Project Overview
Asteroids orbit the Sun, and some come dangerously close to Earth. Early detection of such hazardous asteroids can help prevent potential impacts.
This project uses ML algorithms to automatically identify and predict which asteroids could pose a threat based on parameters like size, speed, and proximity to Earth.

Objectives
Build a machine learning model to classify asteroids as hazardous or non-hazardous.
Preprocess and clean the dataset for better accuracy.
Compare multiple ML algorithms to find the best-performing one.
Handle challenges like imbalanced data and overfitting.

Machine Learning Models Used
Logistic Regression
Random Forest
Naïve Bayes
Decision Tree
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Deep Neural Network (DNN)

Dataset
Source: NASA Jet Propulsion Laboratory (JPL)
Records: 4,688
Features: 40 attributes (size, velocity, proximity to Earth, brightness magnitude, etc.)
Preprocessing:
Handled missing values
Removed redundant columns
Normalized data
Balanced dataset to reduce bias

Implementation
Language: Python
Libraries: scikit-learn, Keras, Pandas, Matplotlib, NumPy
Data Split: 70% training, 30% testing
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Results & Analysis
Top Performing Models:
Random Forest and DNN achieved the best accuracy and precision.

Findings:
ML can significantly improve asteroid detection accuracy.
Overfitting was reduced using cross-validation and hyperparameter tuning.

System Architecture
1. Data Source: NASA JPL dataset
2. Preprocessing: Cleaning, normalization, and balancing
3. Model Training: ML models trained on cleaned data
4. Evaluation: Model performance analyzed
5. Output: Hazardous vs. Non-Hazardous classification

Challenges Faced
Imbalanced dataset
Overfitting
Computational resource limitations
These were mitigated using sampling techniques, regularization, and efficient libraries.

Future Work
Expand dataset with real-time asteroid data
Implement reinforcement learning for adaptive predictions
Collaborate with NASA for real-time monitoring

Authors
Vishanth Reddy Battula, Vignesh Matam, Spandana Pasupulaty
