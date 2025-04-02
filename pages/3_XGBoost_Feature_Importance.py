import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import os
import plotly.io as pio

plot_path = "xgb_feature_importance_plot.html"

import plotly.graph_objects as go

st.set_page_config(page_title="YBRFSS Question XGBoost Visualization", layout="wide")

st.title("XGBoost Feature Importance Ranking")

if not os.path.exists(plot_path):

    if 'df_full_whole' not in st.session_state:
        st.warning("Please preprocess the data in the second page first.")
        st.stop()

    data = st.session_state['df_full_whole']

    # Data preprocessing
    # data_cleaned = data.drop(columns=['sitename'])  # Removing non-numerical and irrelevant columns
    data_cleaned = data.copy()
    data_cleaned = data_cleaned.dropna()  # Drop rows with missing values
    mapping = {
        "q25": {1.0: "Yes", 2.0: "No", np.nan: "Null"},
        "q24": {1.0: "Yes", 2.0: "No", np.nan: "Null"},
        "q84":{
        1.0: "Never",
        2.0: "Rarely",
        3.0: "Sometimes",
        4.0: "Most of the time",
        5.0: "Always",
        np.nan: "Null",  # Handle NaN values
        },
        "q49":{
        1.0: "0 times",
        2.0: "1 or 2 times",
        3.0: "3 to 9 times",
        4.0: "10 to 19 times",
        5.0: "20 to 39 times",
        6.0: "40 or more times",
        np.nan: "Null",  # Handle NaN values
        },
        "q52":{
        1.0: "0 times",
        2.0: "1 or 2 times",
        3.0: "3 to 9 times",
        4.0: "10 to 19 times",
        5.0: "20 to 39 times",
        6.0: "40 or more times",
        np.nan: "Null",  # Handle NaN values
        },
        "q42":{
        1.0: "0 days",
        2.0: "1 or 2day",
        3.0: "3 to 5 days",
        4.0: "6 to 9 days",
        5.0: "10 to 19 days",
        6.0: "20 to 29 days",
        7.0: "All 30 days",
        np.nan: "Null",  # Handle NaN values
        },
        "q14":{
        1.0: "0 days",
        2.0: "1 day",
        3.0: "2 or 3 days",
        4.0: "4 or 5 days",
        5.0: "6 or more days",
        np.nan: "Null",
        },
        "grade": {
        1.0: "9th grade",
        2.0: "10th grade",
        3.0: "11th grade",
        4.0: "12th grade",
        5.0: "Ungraded or other grade",
        },
        "sex": {
        1.0: "Female",
        2.0: "Male"},
        "race7": {
        1.0: "American Indian/Alaska Native",
        2.0: "Asian",
        3.0: "Black or African American",
        4.0: "Hispanic/Latino",
        5.0: "Native Hawaiian/Other Pacific Islander",
        6.0: "White",
        7.0: "Multiple Races (Non-Hispanic)"},
        "age":{
            1.0: "12 years old or younger",
            2.0: "13 years old",
            3.0: "14 years old",
            4.0: "15 years old",
            5.0: "16 years old",
            6.0: "17 years old",
            7.0: "18 years old or older"
        } ,
        "qclose2people":{
            1.0: "Strongly agree",
        2.0: "Agree",
        3.0: "Not sure",
        4.0: "Disagree",
        5.0: "Strongly disagree",
        np.nan: "Null",
        }
        }
    data_cleaned = data_cleaned.replace(mapping)
    mapping2 = {
        "q25":"Q25 - Electronically bullied",
        "q24":"Q24 - School bullied",
        "q84":"Q84 - Mental health",
        "q49":"Q49 - Prescription pill",
        "q52":"Q52 - Heroin takein",
        "q42":"Q42 - Alcohol",
        "q14":"Q14 - School unsafe",
        "qclose2people":"Q - Close to people",
        "sex":"Sex",
        "race7":"Race",
        "grade":"Grade",
        "age":"Age",
        "sitename":"Location"
    }
    data_cleaned = data_cleaned.rename(columns=mapping2)
    # data_cleaned.loc[data_cleaned['Label'] >=2, 'Label'] = 2

    # Separate predictors and target
    X = data_cleaned.drop(columns=['Label', 'weight'])
    y = data_cleaned['Label']
    weights = data_cleaned['weight']

    # Encode categorical variables and scale numerical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(
        X_scaled, y_encoded, weights, test_size=0.3, random_state=42
    )

    # 5-Fold Cross-Validation within the Training Set
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_importances = []
    classification_reports = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        train_weights_fold, val_weights_fold = train_weights.iloc[train_index], train_weights.iloc[val_index]

        # Train XGBoost Classifier
        xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
        xgb_model.fit(X_train_fold, y_train_fold, sample_weight=train_weights_fold)

        # Collect feature importances
        feature_importances.append(xgb_model.feature_importances_)

        # Predict and evaluate
        y_pred = xgb_model.predict(X_val_fold)
        report = classification_report(
            y_val_fold, y_pred, target_names=label_encoder.classes_, output_dict=True, sample_weight=val_weights_fold
        )
        classification_reports.append(report)

    feature_names = X_encoded.columns
    # Aggregate feature importances and compute the mean importance across folds
    feature_importances = np.array(feature_importances)
    mean_importances = np.mean(feature_importances, axis=0)
    sorted_indices = np.argsort(mean_importances)
    sorted_features = feature_names[sorted_indices]

    # Prepare data for boxplot
    feature_importances_sorted = pd.DataFrame(feature_importances[:, sorted_indices], columns=sorted_features)

    fig = go.Figure()

    # Add each feature as a horizontal box
    for feature in feature_importances_sorted.columns:
        fig.add_trace(go.Box(
            x=feature_importances_sorted[feature],  # importance scores
            y=[feature] * len(feature_importances_sorted),  # repeat feature name
            name=feature,
            boxpoints='outliers',
            orientation='h',
            marker=dict(color='lightgrey'),
            line=dict(color='black'),
            showlegend=False
        ))

    fig.update_layout(
        title="Feature Importance Across 5 Folds",
        xaxis_title="Importance",
        yaxis_title="Features",
        template='plotly_white',
        height=800,
        margin=dict(l=200)  # give space for long feature names
    )

    # Save to HTML
    pio.write_html(fig, file=plot_path, full_html=True, include_plotlyjs='cdn')

with open(plot_path, 'r', encoding='utf-8') as f:
    st.components.v1.html(f.read(), height=1000, scrolling=True)

# st.plotly_chart(fig, use_container_width=True)