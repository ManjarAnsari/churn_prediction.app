import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Churn Prediction App", layout="wide")

# -----------------------------------------
# 1. Sidebar Navigation
# -----------------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Action", ["Predict Churn", "Train Model"])

# -----------------------------------------
# Industry & Dynamic Fields
# -----------------------------------------
industries = {
    "Telecom": ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract"],
    "Bank": ["CreditScore", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"],
    "SaaS": ["MonthlySpend", "ActiveUsers", "TenureMonths", "SupportTickets"]
}

industry = st.sidebar.selectbox("Select Industry", list(industries.keys()))

st.title(f"{industry} Churn Prediction")

# -----------------------------------------
# Prediction Mode
# -----------------------------------------
if option == "Predict Churn":
    st.subheader("Enter Customer Details")
    inputs = {}
    for field in industries[industry]:
        if field in ["InternetService", "Contract"]:
            inputs[field] = st.selectbox(f"{field}", ["DSL", "Fiber optic", "No"]) if field == "InternetService" else st.selectbox(f"{field}", ["Month-to-month", "One year", "Two year"])
        else:
            inputs[field] = st.number_input(f"{field}", value=0.0)

    if st.button("Predict"):
        st.write("🔍 Loading model...")
        model_path = f"models/{industry.lower()}_model.pkl"
        try:
            model = joblib.load(model_path)
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            st.success(f"✅ Churn Probability: {prob:.2%}")
            st.write("Prediction:", "Churn" if prediction == 1 else "No Churn")
        except FileNotFoundError:
            st.error("⚠ Model not trained yet. Go to 'Train Model' tab.")

# -----------------------------------------
# Training Mode
# -----------------------------------------
elif option == "Train Model":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data Loaded:", df.shape)
        st.dataframe(df.head())

        target_col = st.selectbox("Select Target Column", df.columns)
        if st.button("Train"):
            X = df.drop(columns=[target_col])
            y = df[target_col]

            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            preprocessor = ColumnTransformer([
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
            ])

            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False))
            ])

            model.fit(X, y)
            joblib.dump(model, f"models/{industry.lower()}_model.pkl")
            st.success(f"✅ Model trained and saved for {industry}")

            # SHAP Explanation
            explainer = shap.Explainer(model.named_steps['classifier'])
            shap_values = explainer(model.named_steps['preprocessor'].transform(X))
            st.subheader("Feature Importance (SHAP)")
            st.pyplot(shap.plots.bar(shap_values, show=False))

    st.download_button("Download Trained Model", data=open(f"models/{industry.lower()}_model.pkl", "rb"), file_name=f"{industry}_model.pkl")
