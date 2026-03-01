import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Model Selection in Sidebar ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    'Choose a Regression Model',
    ('Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Elastic Net')
)

# --- Load Artifacts with Caching based on model choice ---
@st.cache_resource
def load_model(model_name):
    """Loads the specified model, scaler, and columns."""
    # Map the choice to the correct filename
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

# Load the selected model
model, scaler, model_columns = load_model(model_choice)

# --- App Title and Description ---
st.title("Employee Monthly Income Prediction 💰")
st.markdown(f"""
This app predicts an employee's monthly income using a **{model_choice}** model.
Fill in the employee's information on the left and click **Predict Salary**!
""")

# --- Sidebar for User Input ---
st.sidebar.header("Employee Details")

def user_input_features():
    # Categorical Inputs
    department = st.sidebar.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
    education_field = st.sidebar.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))
    job_role = st.sidebar.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    marital_status = st.sidebar.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    overtime = st.sidebar.radio('OverTime', ('No', 'Yes'))
    business_travel = st.sidebar.selectbox('Business Travel', ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))

    # Numerical Inputs
    age = st.sidebar.slider('Age', 18, 60, 30)
    distance_from_home = st.sidebar.slider('Distance From Home (km)', 1, 29, 10)
    education = st.sidebar.selectbox('Education Level (1-5)', (1, 2, 3, 4, 5), index=2)
    environment_satisfaction = st.sidebar.selectbox('Environment Satisfaction (1-4)', (1, 2, 3, 4), index=2)
    job_involvement = st.sidebar.selectbox('Job Involvement (1-4)', (1, 2, 3, 4), index=2)
    job_level = st.sidebar.selectbox('Job Level (1-5)', (1, 2, 3, 4, 5), index=1)
    num_companies_worked = st.sidebar.slider('Number of Companies Worked For', 0, 9, 2)
    percent_salary_hike = st.sidebar.slider('Percent Salary Hike (%)', 11, 25, 15)
    performance_rating = st.sidebar.selectbox('Performance Rating (1-4)', (1, 2, 3, 4), index=2)
    relationship_satisfaction = st.sidebar.selectbox('Relationship Satisfaction (1-4)', (1, 2, 3, 4), index=2)
    total_working_years = st.sidebar.slider('Total Working Years', 0, 40, 10)
    work_life_balance = st.sidebar.selectbox('Work Life Balance (1-4)', (1, 2, 3, 4), index=2)
    years_at_company = st.sidebar.slider('Years at Company', 0, 40, 5)

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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Panel for Prediction ---
st.subheader('Employee Configuration')
st.write(input_df)

if st.button("Predict Salary"):
    # Align input DataFrame columns with model columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)
    predicted_salary = prediction[0]

    # Display the result
    st.subheader('Prediction Result')
    st.success(f"The predicted monthly income is: **${predicted_salary:,.2f}**")
    st.info(f"Prediction made using: {model_choice}")
