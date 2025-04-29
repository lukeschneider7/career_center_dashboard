import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

def load_or_train_models(df, force_retrain=False):
    """Load models from disk if they exist, otherwise train and save them"""
    models = {}
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Define model paths
    model_paths = {
        'working': 'models/rf_model_working.pkl',
        'education': 'models/rf_model_education.pkl',
        'still_looking': 'models/rf_model_still_looking.pkl'
    }
    
    # Feature encoding paths
    encoding_paths = {
        'working': 'models/X_encoded.pkl',
        'education': 'models/X_encoded2.pkl',
        'still_looking': 'models/X_encoded3.pkl'
    }
    
    # Try to load models if they exist and force_retrain is False
    if not force_retrain:
        try:
            # Load models
            for key, path in model_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        models[key] = pickle.load(f)
            
            # Load encoded features
            if os.path.exists(encoding_paths['working']):
                with open(encoding_paths['working'], 'rb') as f:
                    X_encoded = pickle.load(f)
                    
            if os.path.exists(encoding_paths['education']):
                with open(encoding_paths['education'], 'rb') as f:
                    X_encoded2 = pickle.load(f)
                    
            if os.path.exists(encoding_paths['still_looking']):
                with open(encoding_paths['still_looking'], 'rb') as f:
                    X_encoded3 = pickle.load(f)
                    
            # If all models and encodings loaded successfully
            if len(models) == 3 and 'X_encoded' in locals() and 'X_encoded2' in locals() and 'X_encoded3' in locals():
                return models, X_encoded, X_encoded2, X_encoded3
                
        except Exception as e:
            st.warning(f"Error loading models: {e}. Training new models...")
    
    # If we couldn't load models or force_retrain is True, train new ones
    # Model 1: Working Status
    X = df[['avg_unemployment', 'primary_major', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag', 'total_apps']]
    Y = df['Outcome_binary']
    X_encoded = pd.get_dummies(X, columns=['primary_major'])
    
    # No need to split as we're training on the full dataset for deployment
    rf_model_working = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
    rf_model_working.fit(X_encoded, Y)
    models['working'] = rf_model_working
    
    # Model 2: Continuing Education
    X2 = df[['avg_unemployment', 'primary_major', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag']]
    Y2 = df['binary_cont_education']
    X_encoded2 = pd.get_dummies(X2, columns=['primary_major'])
    
    rf_model_education = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
    rf_model_education.fit(X_encoded2, Y2)
    models['education'] = rf_model_education
    
    # Model 3: Still Looking
    X3 = df[['avg_unemployment', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag', 'primary_major']]
    Y3 = df['binary_still_looking']
    X_encoded3 = pd.get_dummies(X3, columns=['primary_major'])
    
    rf_model_still_looking = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_model_still_looking.fit(X_encoded3, Y3)
    models['still_looking'] = rf_model_still_looking
    
    # Save models and encoded features
    try:
        for key, path in model_paths.items():
            with open(path, 'wb') as f:
                pickle.dump(models[key], f)
        
        with open(encoding_paths['working'], 'wb') as f:
            pickle.dump(X_encoded, f)
            
        with open(encoding_paths['education'], 'wb') as f:
            pickle.dump(X_encoded2, f)
            
        with open(encoding_paths['still_looking'], 'wb') as f:
            pickle.dump(X_encoded3, f)
            
        st.success("Models trained and saved successfully!")
    except Exception as e:
        st.warning(f"Error saving models: {e}. Models will be used but not saved.")
    
    return models, X_encoded, X_encoded2, X_encoded3

def employment_prediction(df, force_retrain=False):
    st.header("Predict Student Employment Status")
    
    models, X_encoded, _, _ = load_or_train_models(df, force_retrain)
    model = models['working']
    
    # Form for user input
    st.write("### Enter Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()))
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()))
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0)
        internship_binary = 1 if internship == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0)
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0)
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0)
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
        
        total_apps = st.number_input("Total Applications Submitted", 
                                   min_value=0, 
                                   max_value=200, 
                                   value=int(df['total_apps'].mean()))
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major],
        'total_apps': [total_apps]
    })
    
    # One-hot encode the features
    feature_cols = X_encoded.columns
    features_encoded = pd.get_dummies(features, columns=['primary_major'])
    
    # Align the columns with the training data
    for col in feature_cols:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    features_aligned = features_encoded[feature_cols]
    
    # Make prediction
    if st.button("Predict Employment Status"):
        prediction = model.predict(features_aligned)[0]
        probability = model.predict_proba(features_aligned)[0][1]
        
        st.write("### Prediction Results")
        if prediction == 1:
            st.success(f"This student is likely to be EMPLOYED after graduation (Probability: {probability:.2f})")
        else:
            st.error(f"This student is likely to be UNEMPLOYED after graduation (Probability: {1-probability:.2f})")
        
        # Feature importance
        plot_feature_importance(model, feature_cols)

def education_prediction(df, force_retrain=False):
    st.header("Predict Continuing Education")
    
    models, _, X_encoded2, _ = load_or_train_models(df, force_retrain)
    model = models['education']
    
    # Form for user input
    st.write("### Enter Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()))
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()))
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0)
        internship_binary = 1 if internship == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0)
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0)
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0)
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major]
    })
    
    # One-hot encode the features
    feature_cols = X_encoded2.columns
    features_encoded = pd.get_dummies(features, columns=['primary_major'])
    
    # Align the columns with the training data
    for col in feature_cols:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    features_aligned = features_encoded[feature_cols]
    
    # Make prediction
    if st.button("Predict Continuing Education"):
        prediction = model.predict(features_aligned)[0]
        probability = model.predict_proba(features_aligned)[0][1]
        
        st.write("### Prediction Results")
        if prediction == 1:
            st.success(f"This student is likely to CONTINUE EDUCATION (Probability: {probability:.2f})")
        else:
            st.error(f"This student is likely to NOT CONTINUE EDUCATION (Probability: {1-probability:.2f})")
        
        # Feature importance
        plot_feature_importance(model, feature_cols)

def still_looking_prediction(df, force_retrain=False):
    st.header("Predict If Student Is Still Looking")
    
    models, _, _, X_encoded3 = load_or_train_models(df, force_retrain)
    model = models['still_looking']
    
    # Form for user input
    st.write("### Enter Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()))
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()))
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0)
        internship_binary = 1 if internship == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0)
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0)
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0)
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major]
    })
    
    # One-hot encode the features
    feature_cols = X_encoded3.columns
    features_encoded = pd.get_dummies(features, columns=['primary_major'])
    
    # Align the columns with the training data
    for col in feature_cols:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    features_aligned = features_encoded[feature_cols]
    
    # Make prediction
    if st.button("Predict Still Looking Status"):
        prediction = model.predict(features_aligned)[0]
        probability = model.predict_proba(features_aligned)[0][1]
        
        st.write("### Prediction Results")
        if prediction == 1:
            st.error(f"This student is likely to be STILL LOOKING for opportunities (Probability: {probability:.2f})")
        else:
            st.success(f"This student is likely to be SETTLED with their current status (Probability: {1-probability:.2f})")
        
        # Feature importance
        plot_feature_importance(model, feature_cols)

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the model"""
    st.write("### Feature Importance")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    # Simplify feature names for better display
    feature_names = [name.replace('primary_major_', '') for name in feature_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Important Features')
    
    st.pyplot(fig)
    
    st.write("""
    ### Interpretation Guide
    - **Higher feature importance** indicates that the feature has more influence on the prediction
    - For binary features like internship completion or program participation, a high importance means these factors significantly affect outcomes
    - For majors, importance indicates how strongly a specific major influences the predicted outcome
    """)

def main():
    st.title("Internship Impact Modeling")
    st.write("""
    ### Predict Student Outcomes Based on Career Center Data
    This tool allows career counselors to predict various student outcomes based on internship 
    and career fair participation, helping to identify which factors most influence student success.
    """)
    
    # Load the dataset
    try:
        df = pd.read_csv('final_modeling_data.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    # Sidebar for prediction type selection and model options
    st.sidebar.header("Prediction Options")
    
    prediction_type = st.sidebar.radio(
        "What would you like to predict?",
        ["Student Employment Status", "Continuing Education", "Still Looking Status"]
    )
    
    # Add retrain option in sidebar
    with st.sidebar.expander("Advanced Options"):
        retrain = st.checkbox("Retrain models with current data", value=False)
        if retrain:
            st.sidebar.warning("Models will be retrained using the current dataset. This may take a moment.")
    
    # Main content based on prediction type
    if prediction_type == "Student Employment Status":
        employment_prediction(df, force_retrain=retrain)
    elif prediction_type == "Continuing Education":
        education_prediction(df, force_retrain=retrain)
    else:
        still_looking_prediction(df, force_retrain=retrain)

if __name__ == "__main__":
    main()

