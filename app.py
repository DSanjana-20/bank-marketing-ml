import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("Bank Marketing Classification")
st.subheader("Machine Learning Model Comparison")

st.write(
    "Upload the Bank Marketing test dataset and select a "
    "machine learning model to obtain predictions and evaluation metrics."
)


# ============================================================
# LOAD MODELS
# ============================================================


@st.cache_resource
def load_models():
    models = {}

    models["Logistic Regression"] = joblib.load("logistic_model.pkl")
    models["Decision Tree"] = joblib.load("decision_tree_model.pkl")
    models["kNN"] = joblib.load("knn_model.pkl")
    models["Naive Bayes"] = joblib.load("naive_bayes_model.pkl")
    models["Random Forest"] = joblib.load("random_forest_model.pkl")

    return models


# Load all models
try:
    models = load_models()
    st.success("All five ML models loaded successfully.")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
    ],
)

selected_model = models[selected_model_name]


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
)


# ============================================================
# PROCESS UPLOADED DATA
# ============================================================

if uploaded_file is not None:
    try:
        # Read CSV
        test_data = pd.read_csv(uploaded_file)

        st.success("Test dataset uploaded successfully.")

        # ====================================================
        # DISPLAY DATASET INFORMATION
        # ====================================================

        st.header("Uploaded Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Number of Rows", test_data.shape[0])

        with col2:
            st.metric("Number of Columns", test_data.shape[1])

        st.dataframe(test_data.head(10), use_container_width=True)

        # ====================================================
        # CHECK TARGET COLUMN
        # ====================================================

        if "y" not in test_data.columns:
            st.error("The uploaded CSV must contain the target column 'y'.")
            st.stop()

        # ====================================================
        # SEPARATE FEATURES AND TARGET
        # ====================================================

        X_test = test_data.drop("y", axis=1)
        y_test = test_data["y"]

        # Handle string targets
        if y_test.dtype == "object":
            y_test = (
                y_test.astype(str)
                .str.strip()
                .str.lower()
                .str.rstrip(".")
                .map({"no": 0, "yes": 1})
            )
        else:
            # Handle numeric targets
            y_test = pd.to_numeric(y_test, errors="coerce")
            y_test = y_test.map({0: 0, 1: 1})

        # Check for invalid values
        if y_test.isnull().any():
            st.error("Target column 'y' contains values other than yes/no or 0/1.")
            st.write("Unique target values found:", test_data["y"].unique())
            st.stop()

        # ====================================================
        # MAKE PREDICTIONS
        # ====================================================

        predictions = selected_model.predict(X_test)

        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(X_test)[:, 1]
        else:
            st.error(
                f"{selected_model_name} does not support probability predictions."
            )
            st.stop()

        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_test, predictions)

        # ====================================================
        # DISPLAY SELECTED MODEL
        # ====================================================

        st.header(selected_model_name)

        # ====================================================
        # DISPLAY SIX METRICS
        # ====================================================

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")

        with col2:
            st.metric("AUC", f"{auc:.4f}")

        with col3:
            st.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Recall", f"{recall:.4f}")

        with col5:
            st.metric("F1 Score", f"{f1:.4f}")

        with col6:
            st.metric("MCC", f"{mcc:.4f}")

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, predictions)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax,
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"{selected_model_name} - Confusion Matrix")

        st.pyplot(fig)

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4), use_container_width=True)

        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.subheader("Prediction Results")

        prediction_output = X_test.copy()
        prediction_output["Actual"] = y_test.map({0: "no", 1: "yes"})
        prediction_output["Predicted"] = pd.Series(predictions).map({0: "no", 1: "yes"})
        prediction_output["Probability"] = probabilities

        st.dataframe(prediction_output.head(20), use_container_width=True)

        # ====================================================
        # DOWNLOAD PREDICTIONS
        # ====================================================

        csv_output = prediction_output.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")
else:
    st.info("Please upload your test_data.csv file to begin.")
