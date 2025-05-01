import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
#import pickle
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
            st.write("Attempting to load saved models...")
            
            # Load models
            all_models_exist = True
            for key, path in model_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        models[key] = pickle.load(f)
                    st.write(f"✓ Loaded {key} model")
                else:
                    all_models_exist = False
                    st.write(f"✗ {key} model not found")
            
            # Load encoded features
            encodings_exist = True
            if os.path.exists(encoding_paths['working']):
                with open(encoding_paths['working'], 'rb') as f:
                    X_encoded = pickle.load(f)
                    st.write("✓ Loaded working encodings")
            else:
                encodings_exist = False
                st.write("✗ Working encodings not found")
                    
            if os.path.exists(encoding_paths['education']):
                with open(encoding_paths['education'], 'rb') as f:
                    X_encoded2 = pickle.load(f)
                    st.write("✓ Loaded education encodings")
            else:
                encodings_exist = False
                st.write("✗ Education encodings not found")
                    
            if os.path.exists(encoding_paths['still_looking']):
                with open(encoding_paths['still_looking'], 'rb') as f:
                    X_encoded3 = pickle.load(f)
                    st.write("✓ Loaded still_looking encodings")
            else:
                encodings_exist = False
                st.write("✗ Still looking encodings not found")
                    
            # If all models and encodings loaded successfully
            if all_models_exist and encodings_exist:
                st.success("All models loaded successfully!")
                return models, X_encoded, X_encoded2, X_encoded3
            else:
                st.warning("Some models or encodings missing. Training new models...")
                
        except Exception as e:
            st.warning(f"Error loading models: {e}. Training new models...")
    else:
        st.info("Force retrain selected. Training new models...")
    
    # If we couldn't load models or force_retrain is True, train new ones
    with st.spinner('Training models - this may take a few minutes...'):
        # Model 1: Working Status
        X = df[['avg_unemployment', 'primary_major', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag', 'appointment_binary']]
        Y = df['Outcome_binary']
        X_encoded = pd.get_dummies(X, columns=['primary_major'])
        
        st.write("Training employment model...")
        rf_model_working = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
        rf_model_working.fit(X_encoded, Y)
        models['working'] = rf_model_working
        
        # Model 2: Continuing Education
        X2 = df[['avg_unemployment', 'primary_major', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag','appointment_binary']]
        Y2 = df['binary_cont_education']
        X_encoded2 = pd.get_dummies(X2, columns=['primary_major'])
        
        st.write("Training education model...")
        rf_model_education = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
        rf_model_education.fit(X_encoded2, Y2)
        models['education'] = rf_model_education
        
        # Model 3: Still Looking
        X3 = df[['avg_unemployment', 'fairs_above_avg', 'Internship_binary', 'apps_above_avg', 'ipp_flag', 'primary_major','appointment_binary']]
        Y3 = df['binary_still_looking']
        X_encoded3 = pd.get_dummies(X3, columns=['primary_major'])
        
        st.write("Training still looking model...")
        rf_model_still_looking = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf_model_still_looking.fit(X_encoded3, Y3)
        models['still_looking'] = rf_model_still_looking
    
    # Save models and encoded features
    try:
        st.write("Saving models for future use...")
        for key, path in model_paths.items():
            with open(path, 'wb') as f:
                pickle.dump(models[key], f)
            st.write(f"✓ Saved {key} model")
        
        with open(encoding_paths['working'], 'wb') as f:
            pickle.dump(X_encoded, f)
        st.write("✓ Saved working encodings")
            
        with open(encoding_paths['education'], 'wb') as f:
            pickle.dump(X_encoded2, f)
        st.write("✓ Saved education encodings")
            
        with open(encoding_paths['still_looking'], 'wb') as f:
            pickle.dump(X_encoded3, f)
        st.write("✓ Saved still looking encodings")
            
        st.success("Models trained and saved successfully!")
    except Exception as e:
        st.warning(f"Error saving models: {e}. Models will be used but not saved.")
    
    return models, X_encoded, X_encoded2, X_encoded3

def employment_prediction(df, force_retrain=False):
    # Display a loading message while retrieving models
    with st.spinner("Loading models..."):
        models, X_encoded, _, _ = load_or_train_models(df, force_retrain)
        model = models['working']
    
    # Form for user input
    st.markdown('<div class="sub-header">Enter Student Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()),
                                    step=0.1,
                                    key="unemployment_emp")  # Add unique key
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()),
                                    key="major_emp")  # Add unique key
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0,
                            key="internship_emp")  # Add unique key
        internship_binary = 1 if internship == "Yes" else 0
        
        # Add the appointment field
        appointment = st.radio("Attended Career Center Appointment?", 
                             options=["Yes", "No"], 
                             index=0,
                             key="appointment_emp")  # Add unique key
        appointment_binary = 1 if appointment == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0,
                                  key="fairs_emp")  # Add unique key
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0,
                                 key="apps_emp")  # Add unique key
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0,
                           key="ipp_emp")  # Add unique key
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major],
        'appointment_binary': [appointment_binary]  # Add the appointment field
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
    if st.button("Predict Employment Status", key="predict_emp"):
        with st.spinner("Generating prediction..."):
            prediction = model.predict(features_aligned)[0]
            probability = model.predict_proba(features_aligned)[0][1]
            
            st.markdown("### Prediction Results")
            
            if prediction == 1:
                st.success(f"This student is likely to be EMPLOYED after graduation (Probability: {probability:.2f})")
            else:
                st.error(f"This student is likely to be UNEMPLOYED after graduation (Probability: {1-probability:.2f})")
            
            # Feature importance
            plot_feature_importance(model, feature_cols)

def education_prediction(df, force_retrain=False):
    # Display a loading message while retrieving models
    with st.spinner("Loading models..."):
        models, _, X_encoded2, _ = load_or_train_models(df, force_retrain)
        model = models['education']
    
    # Form for user input
    st.markdown('<div class="sub-header">Enter Student Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()),
                                    step=0.1,
                                    key="unemployment_edu")  # Add unique key
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()),
                                    key="major_edu")  # Add unique key
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0,
                            key="internship_edu")  # Add unique key
        internship_binary = 1 if internship == "Yes" else 0
        
        # Add the appointment field
        appointment = st.radio("Attended Career Center Appointment?", 
                             options=["Yes", "No"], 
                             index=0,
                             key="appointment_edu")  # Add unique key
        appointment_binary = 1 if appointment == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0,
                                  key="fairs_edu")  # Add unique key
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0,
                                 key="apps_edu")  # Add unique key
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0,
                           key="ipp_edu")  # Add unique key
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major],
        'appointment_binary': [appointment_binary]  # Add the appointment field
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
    if st.button("Predict Continuing Education", key="predict_edu"):
        with st.spinner("Generating prediction..."):
            prediction = model.predict(features_aligned)[0]
            probability = model.predict_proba(features_aligned)[0][1]
            
            st.markdown("### Prediction Results")
            
            if prediction == 1:
                st.success(f"This student is likely to CONTINUE EDUCATION (Probability: {probability:.2f})")
            else:
                st.error(f"This student is likely to NOT CONTINUE EDUCATION (Probability: {1-probability:.2f})")
            
            # Feature importance
            plot_feature_importance(model, feature_cols)

def still_looking_prediction(df, force_retrain=False):
    # Display a loading message while retrieving models
    with st.spinner("Loading models..."):
        models, _, _, X_encoded3 = load_or_train_models(df, force_retrain)
        model = models['still_looking']
    
    # Form for user input
    st.markdown('<div class="sub-header">Enter Student Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        unemployment_rate = st.slider("Average Unemployment Rate (%)", 
                                    min_value=float(df['avg_unemployment'].min()), 
                                    max_value=float(df['avg_unemployment'].max()), 
                                    value=float(df['avg_unemployment'].mean()),
                                    step=0.1,
                                    key="unemployment_sl")  # Add unique key
        
        primary_major = st.selectbox("Primary Major", 
                                    options=sorted(df['primary_major'].unique()),
                                    key="major_sl")  # Add unique key
        
        internship = st.radio("Completed an Internship?", 
                            options=["Yes", "No"], 
                            index=0,
                            key="internship_sl")  # Add unique key
        internship_binary = 1 if internship == "Yes" else 0
        
        # Add the appointment field
        appointment = st.radio("Attended Career Center Appointment?", 
                             options=["Yes", "No"], 
                             index=0,
                             key="appointment_sl")  # Add unique key
        appointment_binary = 1 if appointment == "Yes" else 0
    
    with col2:
        fairs_above_avg = st.radio("Career Fair Attendance Above Average?", 
                                  options=["Yes", "No"], 
                                  index=0,
                                  key="fairs_sl")  # Add unique key
        fairs_above_avg_binary = 1 if fairs_above_avg == "Yes" else 0
        
        apps_above_avg = st.radio("Applications Submitted Above Average?", 
                                 options=["Yes", "No"], 
                                 index=0,
                                 key="apps_sl")  # Add unique key
        apps_above_avg_binary = 1 if apps_above_avg == "Yes" else 0
        
        ipp_flag = st.radio("Participated in IPP Program?", 
                           options=["Yes", "No"], 
                           index=0,
                           key="ipp_sl")  # Add unique key
        ipp_flag_binary = 1 if ipp_flag == "Yes" else 0
    
    # Create feature array for prediction
    features = pd.DataFrame({
        'avg_unemployment': [unemployment_rate],
        'fairs_above_avg': [fairs_above_avg_binary],
        'Internship_binary': [internship_binary],
        'apps_above_avg': [apps_above_avg_binary],
        'ipp_flag': [ipp_flag_binary],
        'primary_major': [primary_major],
        'appointment_binary': [appointment_binary]  # Add the appointment field
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
    if st.button("Predict Still Looking Status", key="predict_sl"):
        with st.spinner("Generating prediction..."):
            prediction = model.predict(features_aligned)[0]
            probability = model.predict_proba(features_aligned)[0][1]
            
            st.markdown("### Prediction Results")
            
            if prediction == 1:
                st.error(f"This student is likely to be STILL LOOKING for opportunities (Probability: {probability:.2f})")
            else:
                st.success(f"This student is likely to be SETTLED with their current status (Probability: {1-probability:.2f})")
            
            # Feature importance
            plot_feature_importance(model, feature_cols)

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the model"""
    st.markdown("### Feature Importance")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    # Simplify feature names for better display
    simplified_names = [name.replace('primary_major_', '') for name in feature_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align='center', color='#E57200')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([simplified_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Important Features')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    <div class="interpretation">
    <h3>Interpretation Guide</h3>
    <ul>
        <li><strong>Higher feature importance</strong> indicates that the feature has more influence on the prediction</li>
        <li>For binary features like internship completion, appointment attendance, or program participation, a high importance means these factors significantly affect outcomes</li>
        <li>For majors, importance indicates how strongly a specific major influences the predicted outcome</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Add custom styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #232D4B;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #E57200;
        }
        .stButton button {
            background-color: #E57200;
            color: white;
        }
        .interpretation {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Add header
    st.markdown('<div class="main-header">UVA Career Center Model Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predictive Analytics for Student Success</div>', unsafe_allow_html=True)

    # Create a separator
    st.markdown("---")
    
    # Add a spinner while loading data
    with st.spinner("Loading data..."):
        try:
            df = pd.read_csv('final_modeling_data.csv')
            st.success(f"Data loaded successfully! ({len(df)} records)")
            
            # Show a small preview of the data
            if st.checkbox("Preview dataset"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.error("Please make sure 'final_modeling_data.csv' is in the same directory as this script.")
            return
    
    # Create tabs for different prediction types
    tab1, tab2, tab3 = st.tabs([
        "🎓 Student Employment", 
        "📚 Continuing Education", 
        "🔍 Still Looking Status"
    ])
    
    # Add retrain option above tabs
    with st.expander("⚙️ Advanced Options"):
        retrain = st.checkbox("Retrain models with current data", value=False)
        if retrain:
            st.warning("Models will be retrained using the current dataset. This may take a moment.")
    
    # Content for each tab
    with tab1:
        st.header("Predict Student Employment Status")
        employment_prediction(df, force_retrain=retrain)
    
    with tab2:
        st.header("Predict Continuing Education")
        education_prediction(df, force_retrain=retrain)
    
    with tab3:
        st.header("Predict If Student Is Still Looking")
        still_looking_prediction(df, force_retrain=retrain)
    
    # Add footer
    st.markdown("---")
    st.markdown("UVA Career Center Dashboard | Created by Luke Schneider")

if __name__ == "__main__":
    main()

