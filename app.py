#app.py

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate

st.title("Diabetes Prediction App")

#Load model and data
model = joblib.load("diabetes_catboost_model.pkl")
df = pd.read_csv("diabetes_prediction_dataset.csv")

categorical = ["gender", "smoking_history"]
numerical = ["age","bmi","HbA1c_level", "blood_glucose_level"]
binary = ["hypertension", "heart_disease"]

#Sidebar imputs
st.sidebar.header("Input Patient Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", df.gender.unique())
    age = st.sidebar.slider("Age", 0, 100, 30)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    smoking_history = st.sidebar.selectbox("Smoking History", df.smoking_history.unique())
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 24.0)
    hba1c = st.sidebar.slider("HbA1c Level", 3.0, 10.0, 5.5)
    glucose = st.sidebar.slider("Blood Glucose Level", 50, 300, 120)

    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

#Fairness Dashboard
st.header("Fairness Dashboard")
with st.expander("Check group-wise fairness metrics"):
    sensitive_feature = st.selectbox("Select sensitive feature:", categorical + binary)
    X_eval = df.drop("diabetes", axis=1)
    y_true = df["diabetes"]
    y_pred = model.predict(X_eval)

    fairness_metrics = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=X_eval[sensitive_feature]
    )

    st.subheader(f"Fairness Metrics by {sensitive_feature.title()}")
    st.dataframe(fairness_metrics.by_group)



#Prediction Section
if st.button("Predict"):
    st.header("Prediction Result")
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown(f"**Diabtes Prediction:** {'yes' if prediction else 'No' }")
    st.markdown(f"**Probability of Diabetes:** {prob:.2f}")

    # SHAP explanation
    st.header("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    X_data = df.drop("diabetes", axis=1)
    shap_values_all = explainer.shap_values(X_data)

    #Waterfall Plot
    st.subheader("Waterfall Plot (Your Prediction)")
    st.caption("This plot explains the current prediction by showing how each feature pushed the prediction higher or lower starting from the model’s average.")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    ), max_display=10, show=False)
    st.pyplot(fig)

    #Shap Summary plot
    st.subheader("Summary Plot")
    st.caption("Each dot represents a person in the dataset. Features are sorted by importance, and colors show whether high or low feature values increase or decrease the prediction.")
    fig, ax= plt.subplots()
    shap.summary_plot(shap_values_all, X_data, show=False)
    st.pyplot(fig)

    #Dependence Plot
    st.subheader("Dependence Plot (Age)")
    st.caption("This plot shows how the SHAP value for each individual prediction changes as age changes. It helps you understand how age affects the model’s output.")
    fig, ax = plt.subplots()
    shap.dependence_plot("age", shap_values_all, X_data, ax=ax, show=False)
    st.pyplot(fig)

    #Feature Imporatance Bar
    st.subheader("Feature Importance")
    st.caption("This bar chart shows which features have the biggest overall impact on the model’s predictions, ranked by average SHAP value.")
    shap_expl = shap.Explanation(values=shap_values_all,
                                 base_values=explainer.expected_value,
                                 data=X_data.values,
                                 feature_names=X_data.columns)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_expl, show=False)
    st.pyplot(fig)







