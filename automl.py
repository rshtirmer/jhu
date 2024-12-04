import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "results" not in st.session_state:
    st.session_state.results = None


def landing_page():
    st.title("Welcome to AutoML!")
    st.write("""
        This tool allows you to upload a dataset, automatically run multiple machine learning models, 
        and view concise, clear results to guide your decisions.
    """)
    st.write("### Steps to Get Started:")
    st.write("1. Upload your dataset.")
    st.write("2. Run models automatically.")
    st.write("3. View and download results.")
    if st.button("Start Now"):
        st.session_state.page = "upload"


def dataset_upload():
    st.title("Upload Your Dataset")
    st.write("""
        Please upload a CSV file containing tabular data. Ensure the dataset has a target column for classification.
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        dataset = pd.read_csv(uploaded_file)
        st.session_state.dataset = dataset
        st.write("### Dataset Preview:")
        st.dataframe(dataset.head())
        target_column = st.text_input("Enter the target column name:")
        if st.button("Run Models"):
            if target_column in dataset.columns:
                st.session_state.target_column = target_column
                st.session_state.page = "models"
            else:
                st.error("Target column not found in the dataset.")


def model_execution():
    st.title("Running Models")
    st.write("We are training three machine learning models on your dataset:")
    st.write("1. Logistic Regression")
    st.write("2. Random Forest")
    st.write("3. Gradient Boosting")

    dataset = st.session_state.dataset
    target_column = st.session_state.target_column

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for model_name, model in models.items():
        with st.spinner(f"Training {model_name}..."):
            time.sleep(1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

    st.session_state.results = pd.DataFrame(results)
    st.session_state.page = "results"
    model_results()


def model_results():
    st.title("Model Results")
    st.write("""
        Below are the evaluation metrics for each model. Use this information to select the best 
        performing model for your dataset.
    """)

    results = st.session_state.results
    st.write("### Performance Metrics:")
    st.dataframe(results)

    st.write("### Visual Comparison:")
    st.bar_chart(results.set_index("Model")[["Accuracy", "F1-Score"]])

    st.write("### Next Steps:")
    if st.download_button("Download Results as CSV", results.to_csv(index=False).encode('utf-8'), "model_results.csv", "text/csv"):
        st.success("Results downloaded!")
    if st.button("Upload Another Dataset"):
        st.session_state.page = "upload"


if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "upload":
    dataset_upload()
elif st.session_state.page == "models":
    model_execution()
elif st.session_state.page == "results":
    model_results()
