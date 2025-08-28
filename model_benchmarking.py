# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:52:07 2025

@author: Cosmos
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import csv
import textwrap
from datetime import datetime
import re
from collections import defaultdict
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
#from betareg import Beta
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import f_oneway
from scipy.stats import chi2
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import layers, regularizers, Model, Input
import tensorflow as tf
from attention_layer import Attention
from preprocess_data import preprocess_data
from sklearn.datasets import make_classification


# Specify the file path
file_path = 'enter file path'

# Read the CSV file into a DataFrame
data2 = pd.read_csv(file_path, quotechar='"', delimiter=';', na_values=[
                    "", "\"\""], encoding="UTF-8", engine='python')

data2.rename(columns=lambda x: x.split('.')[0] if '.' in x else x, inplace=True)

# Preprocess
data2 = preprocess_data(data2)

# Display the first few rows of the DataFrame to check
print(data2.head())



############################# columns to include for modeling #####################################

columns_to_include = [

   'Income',
   
   'Commute Time',
   
   #'own cars',

   'Job Category',
    
   'Contract Type',

    'Gender',  # Gender types

    'Age Group',

   'Child Drop Off Frequency',

    
    #'Partner Drop Off',

    'Number of Kid to Drop-off',

    'Practical Arrival Time'   ,   #arrival time types

    'Theoretical Arrival Time Contract',

    'Telework Frequency',

    'Mode of Transport',  # Mode of transport types

    'Shift',


    # 'Before 08:00',
    # '08:00 - 08:29',
    # '08:30 - 08:59',
    # '09:00 - 09:29',
    # 'After 09:30',  # 
    #'Arrival Time',#arrival interval

   #'Rank 1', 
    #'Rank 2',

  #'Traffic Experience',
    
   #'Long Commute Time',
    
    'Congestion Discomfort Rating',
     #'Travel Habits Change',
   #'Traffic Delay Amount',

    # 'Change my mode of transport',  # travel habits changes and why not reasons
    # 'Change my route to avoid traffic jams',
    # 'Leave my home early',
    # 'Leave my home later',
    # 'Cancel my trip (teleworking for example)',
    # 'Take a break in my journey (in a café for example)',
    # 'Other_a',
    # 'I have no other mode of transport available',
    # 'I have no other possible route',
    # 'It is impossible for me to shift my journey in time',
    # 'I do not believe that changing my travel habits would improve my situation',
    # 'I do not want to change my travel habits, despite the traffic jams I experience',
    # 'Due to the time I want to leave work in the evening',
    # 'Other_c',
    
#     'Improve traffic information',
#     'Set up a bonus system',
#     'Limit meetings before 10 a.m.',
# 'Adjust business schedules',
# 'Arrange school/daycare schedules',
# 'Encourage the practice of teleworking',
# 'Authorize the practice of teleworking during peak hours',
#   'Implement a communication campaign',
#   'Encourage carpooling',
#   'Define new rules for access to parking spaces',
#   'Encourage the use of other modes of travel than the personal car',

    'Residence'

]


data_filter = data2.filter(columns_to_include)

# Mapping for the new labels
shift_mapping = {
      'Shift Everyday': 1,
      'Never': 0,
    'Shift but Not Everyday': 1
}

# Apply the mapping to assign numeric codes to 'Arrival Time'
data_filter['Shift'] = data_filter['Shift'].map(shift_mapping)
data_filter.dropna(inplace=True)


# Modeling Starts here

###Apply SMOTE to handle class imbalance (if necessary)
from imblearn.over_sampling import RandomOverSampler



X = data_filter.drop(['Shift'], axis=1)
y = data_filter['Shift']

# Handle class imbalance with RandomOverSampler
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# Prepare categorical data for XGBoost native categorical input
X_cat = X.copy()
for col in X_cat.select_dtypes(include='object').columns:
    X_cat[col] = X_cat[col].astype('category')

# Prepare one-hot encoded data for other models
X_encoded = pd.get_dummies(X, drop_first=False)



def create_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((1, 16))(x)
    x = Attention()(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Prepare cross-validation ===
models = ['RandomForest', 'XGBoost_OHE', 'XGBoost_Cat', 'NeuralNet']
results = {m: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []} for m in models}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X):
    # Split data into train/val for each variant
    X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    X_train_ohe, X_val_ohe = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', max_samples=0.8, random_state=42
    )
    rf.fit(X_train_ohe, y_train)
    y_pred_rf = rf.predict(X_val_ohe)
    results['RandomForest']['accuracy'].append(accuracy_score(y_val, y_pred_rf))
    results['RandomForest']['precision'].append(precision_score(y_val, y_pred_rf, zero_division=0))
    results['RandomForest']['recall'].append(recall_score(y_val, y_pred_rf, zero_division=0))
    results['RandomForest']['f1_score'].append(f1_score(y_val, y_pred_rf, zero_division=0))

    # --- XGBoost with One-Hot Encoding ---
    xgb_ohe = XGBClassifier(
        objective='binary:logistic', max_depth=5, learning_rate=0.1,
        eval_metric='logloss', alpha=0.1, reg_lambda=0.1, random_state=42
    )
    xgb_ohe.fit(X_train_ohe, y_train)
    y_pred_xgb_ohe = xgb_ohe.predict(X_val_ohe)
    results['XGBoost_OHE']['accuracy'].append(accuracy_score(y_val, y_pred_xgb_ohe))
    results['XGBoost_OHE']['precision'].append(precision_score(y_val, y_pred_xgb_ohe, zero_division=0))
    results['XGBoost_OHE']['recall'].append(recall_score(y_val, y_pred_xgb_ohe, zero_division=0))
    results['XGBoost_OHE']['f1_score'].append(f1_score(y_val, y_pred_xgb_ohe, zero_division=0))

    # --- XGBoost with native categorical input ---
    xgb_cat = XGBClassifier(
        objective='binary:logistic', max_depth=5, learning_rate=0.1,
        eval_metric='logloss', alpha=0.1, reg_lambda=0.1, enable_categorical=True, random_state=42
    )
    xgb_cat.fit(X_train_cat, y_train)
    y_pred_xgb_cat = xgb_cat.predict(X_val_cat)
    results['XGBoost_Cat']['accuracy'].append(accuracy_score(y_val, y_pred_xgb_cat))
    results['XGBoost_Cat']['precision'].append(precision_score(y_val, y_pred_xgb_cat, zero_division=0))
    results['XGBoost_Cat']['recall'].append(recall_score(y_val, y_pred_xgb_cat, zero_division=0))
    results['XGBoost_Cat']['f1_score'].append(f1_score(y_val, y_pred_xgb_cat, zero_division=0))

    # --- Neural Network on scaled one-hot encoded data ---
    # No scaling, but ensure numpy float32 for TF
    X_train_nn = X_train_ohe.values.astype('float32')
    X_val_nn = X_val_ohe.values.astype('float32')
    nn = create_attention_model(input_shape=(X_train_nn.shape[1],))
    nn.fit(X_train_nn, y_train, epochs=10, verbose=0)
    y_pred_nn = np.argmax(nn.predict(X_val_nn), axis=1)
    results['NeuralNet']['accuracy'].append(accuracy_score(y_val, y_pred_nn))
    results['NeuralNet']['precision'].append(precision_score(y_val, y_pred_nn, zero_division=0))
    results['NeuralNet']['recall'].append(recall_score(y_val, y_pred_nn, zero_division=0))
    results['NeuralNet']['f1_score'].append(f1_score(y_val, y_pred_nn, zero_division=0))

# === Print average metrics ===
print("\nCross-Validation Results (10-fold):")
for model in models:
    print(f"\n--- {model} ---")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        avg = np.mean(results[model][metric])
        print(f"{metric.capitalize():<10}: {avg:.4f}")
        
        
# === Learning Curve Parameters ===
train_sizes = np.linspace(0.1, 1.0, 10)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = ['RandomForest', 'XGBoost_OHE', 'XGBoost_Cat', 'NeuralNet']
learning_curve_results = {m: {'train': [], 'val': []} for m in models}

for frac in train_sizes:
    fold_results = {m: {'train': [], 'val': []} for m in models}
    for train_idx, val_idx in kf.split(X_encoded):
        # Split folds
        X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_train_ohe, X_val_ohe = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Subsample training set
        sub_size = int(len(X_train_ohe) * frac)
        X_sub_ohe, y_sub = X_train_ohe.iloc[:sub_size], y_train.iloc[:sub_size]
        X_sub_cat = X_train_cat.iloc[:sub_size]

        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    min_samples_split=10, min_samples_leaf=5,
                                    max_features='sqrt', max_samples=0.8, random_state=42)
        rf.fit(X_sub_ohe, y_sub)
        fold_results['RandomForest']['train'].append(accuracy_score(y_sub, rf.predict(X_sub_ohe)))
        fold_results['RandomForest']['val'].append(accuracy_score(y_val, rf.predict(X_val_ohe)))

        # --- XGBoost OHE ---
        xgb_ohe = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1,
                                eval_metric='logloss', alpha=0.1, reg_lambda=0.1, random_state=42)
        xgb_ohe.fit(X_sub_ohe, y_sub)
        fold_results['XGBoost_OHE']['train'].append(accuracy_score(y_sub, xgb_ohe.predict(X_sub_ohe)))
        fold_results['XGBoost_OHE']['val'].append(accuracy_score(y_val, xgb_ohe.predict(X_val_ohe)))

        # --- XGBoost Cat ---
        xgb_cat = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1,
                                eval_metric='logloss', alpha=0.1, reg_lambda=0.1,
                                enable_categorical=True, random_state=42)
        xgb_cat.fit(X_sub_cat, y_sub)
        fold_results['XGBoost_Cat']['train'].append(accuracy_score(y_sub, xgb_cat.predict(X_sub_cat)))
        fold_results['XGBoost_Cat']['val'].append(accuracy_score(y_val, xgb_cat.predict(X_val_cat)))

        # --- Neural Network ---
        X_sub_nn = X_sub_ohe.values.astype('float32')
        X_val_nn = X_val_ohe.values.astype('float32')
        y_sub_nn = y_sub.values.astype('int32')
        y_val_nn = y_val.values.astype('int32')
        nn = create_attention_model((X_sub_nn.shape[1],))
        nn.fit(X_sub_nn, y_sub_nn, epochs=10, batch_size=32, verbose=0)
        train_acc = accuracy_score(y_sub_nn, np.argmax(nn.predict(X_sub_nn, verbose=0), axis=1))
        val_acc = accuracy_score(y_val_nn, np.argmax(nn.predict(X_val_nn, verbose=0), axis=1))
        fold_results['NeuralNet']['train'].append(train_acc)
        fold_results['NeuralNet']['val'].append(val_acc)

    # Average over folds
    for model in models:
        learning_curve_results[model]['train'].append(np.mean(fold_results[model]['train']))
        learning_curve_results[model]['val'].append(np.mean(fold_results[model]['val']))

# === Plot Learning Curves ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()

for i, model in enumerate(models):
    ax = axs[i]
    ax.plot(train_sizes, learning_curve_results[model]['train'], label='Train Accuracy', marker='o')
    ax.plot(train_sizes, learning_curve_results[model]['val'], label='Validation Accuracy', marker='x')
    ax.set_title(f'Learning Curve - {model}')
    ax.set_xlabel('Training Set Size (Fraction)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.6, 1.05)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
