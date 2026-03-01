import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Choose Prediction Mode", 
                        ("Single Employee Prediction", "Bulk Prediction (CSV Upload)"))

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    'Choose a Regression Model',
    ('Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Elastic Net')
)

# --- Load Model & Artifacts ---
@st.cache_resource
def load_model(model_name):
    model_filename_map = {
        'Linear Regression': 'linear_model.joblib',
        'Lasso Regression': 'lasso_model.joblib',
        'Ridge Regression': 'ridge_model.joblib',
        'Elastic Net': 'elastic_model.joblib'
    }
    model = joblib.load(model_filename_map[model_name])
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return model, scaler, model_columns

model, scaler, model_columns = load_model(model_choice)

# --- App Title ---
st.title("Employee Monthly Income Prediction App 💰")
st.markdown(f"**Model in use:** {model_choice}")

# --- Common Prediction Function ---
def predict_salary(input_df):
    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Scale
    scaled_input = scaler.transform(input_df)
    # Predict
    predictions = model.predict(scaled_input)
    return predictions

# ==================== SINGLE EMPLOYEE PREDICTION ====================
if mode == "Single Employee Prediction":
    st.subheader("Enter Employee Details")

    with st.sidebar:
        st.header("Employee Details")

        def user_input_features():
            department = st.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
            education_field = st.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))
            job_role = st.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
            gender = st.radio('Gender', ('Male', 'Female'))
            marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
            overtime = st.radio('OverTime', ('No', 'Yes'))
            business_travel = st.selectbox('Business Travel', ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))

            age = st.slider('Age', 18, 60, 30)
            distance_from_home = st.slider('Distance From Home (km)', 1, 29, 10)
            education = st.selectbox('Education Level (1-5)', (1, 2, 3, 4, 5), index=2)
            environment_satisfaction = st.selectbox('Environment Satisfaction (1-4)', (1, 2, 3, 4), index=2)
            job_involvement = st.selectbox('Job Involvement (1-4)', (1, 2, 3, 4), index=2)
            job_level = st.selectbox('Job Level (1-5)', (1, 2, 3, 4, 5), index=1)
            num_companies_worked = st.slider('Number of Companies Worked For', 0, 9, 2)
            percent_salary_hike = st.slider('Percent Salary Hike (%)', 11, 25, 15)
            performance_rating = st.selectbox('Performance Rating (1-4)', (1, 2, 3, 4), index=2)
            relationship_satisfaction = st.selectbox('Relationship Satisfaction (1-4)', (1, 2, 3, 4), index=2)
            total_working_years = st.slider('Total Working Years', 0, 40, 10)
            work_life_balance = st.selectbox('Work Life Balance (1-4)', (1, 2, 3, 4), index=2)
            years_at_company = st.slider('Years at Company', 0, 40, 5)

            data = {
                'Age': age, 'DistanceFromHome': distance_from_home, 'Education': education,
                'EnvironmentSatisfaction': environment_satisfaction, 'JobInvolvement': job_involvement,
                'JobLevel': job_level, 'NumCompaniesWorked': num_companies_worked,
                'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
                'RelationshipSatisfaction': relationship_satisfaction, 'TotalWorkingYears': total_working_years,
                'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
                'BusinessTravel_Travel_Frequently': 1 if business_travel == 'Travel_Frequently' else 0,
                'BusinessTravel_Travel_Rarely': 1 if business_travel == 'Travel_Rarely' else 0,
                'Department_Research & Development': 1 if department == 'Research & Development' else 0,
                'Department_Sales': 1 if department == 'Sales' else 0,
                'EducationField_Human Resources': 1 if education_field == 'Human Resources' else 0,
                'EducationField_Life Sciences': 1 if education_field == 'Life Sciences' else 0,
                'EducationField_Marketing': 1 if education_field == 'Marketing' else 0,
                'EducationField_Medical': 1 if education_field == 'Medical' else 0,
                'EducationField_Other': 1 if education_field == 'Other' else 0,
                'EducationField_Technical Degree': 1 if education_field == 'Technical Degree' else 0,
                'Gender_Male': 1 if gender == 'Male' else 0,
                'JobRole_Human Resources': 1 if job_role == 'Human Resources' else 0,
                'JobRole_Laboratory Technician': 1 if job_role == 'Laboratory Technician' else 0,
                'JobRole_Manager': 1 if job_role == 'Manager' else 0,
                'JobRole_Manufacturing Director': 1 if job_role == 'Manufacturing Director' else 0,
                'JobRole_Research Director': 1 if job_role == 'Research Director' else 0,
                'JobRole_Research Scientist': 1 if job_role == 'Research Scientist' else 0,
                'JobRole_Sales Executive': 1 if job_role == 'Sales Executive' else 0,
                'JobRole_Sales Representative': 1 if job_role == 'Sales Representative' else 0,
                'MaritalStatus_Married': 1 if marital_status == 'Married' else 0,
                'MaritalStatus_Single': 1 if marital_status == 'Single' else 0,
                'OverTime_Yes': 1 if overtime == 'Yes' else 0,
            }
            return pd.DataFrame(data, index=[0])

        input_df = user_input_features()

    # Display input
    st.subheader("Employee Configuration")
    st.write(input_df)

    if st.button("Predict Salary"):
        prediction = predict_salary(input_df)
        st.subheader("Prediction Result")
        st.success(f"Predicted Monthly Income: **${prediction[0]:,.2f}**")
        st.info(f"Model used: {model_choice}")

# ==================== BULK PREDICTION (CSV UPLOAD) ====================
elif mode == "Bulk Prediction (CSV Upload)":
    st.subheader("Upload CSV File (same format as training data)")

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        df_original = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df_original.head())

        # --- Preprocessing & Visualization ---
        st.subheader("Data Preprocessing & Visualization (Same as Training)")

        # 1. Distribution of Target (if present)
        if 'MonthlyIncome' in df_original.columns:
            st.markdown("### Monthly Income Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_original['MonthlyIncome'], kde=True, ax=ax)
            ax.set_title("Distribution of Monthly Income (Original)")
            st.pyplot(fig)

        # 2. Boxplot of Monthly Income by Job Level
        if 'JobLevel' in df_original.columns and 'MonthlyIncome' in df_original.columns:
            st.markdown("### Monthly Income by Job Level")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='JobLevel', y='MonthlyIncome', data=df_original, ax=ax)
            st.pyplot(fig)

        # --- Make Predictions ---
        st.subheader("Making Predictions...")
        predictions = predict_salary(df_original)

        # Add prediction column
        df_original['Predicted_Monthly_Income'] = predictions

        st.write("Predicted Data Preview:")
        st.dataframe(df_original.head())

        # --- Download ---
        st.subheader("Download Results")
        csv = df_original.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv,
            file_name="predicted_salaries.csv",
            mime="text/csv"
        )

        st.success("Bulk prediction completed! You can now download the CSV with predicted salaries.")

st.sidebar.markdown("---")
st.sidebar.info("Upload CSV or enter details to predict salaries using the selected model.")