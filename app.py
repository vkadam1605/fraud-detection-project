import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# The filename below matches the filename used in the notebook's
# final joblib.dump() cell.
MODEL_PATH = Path(__file__).resolve().parent / "Fraud_Dtection_Model.joblib"

# These are the exact features used to train best_rf in the notebook.
FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# App interface
# --------------------------------------------------
st.title("🛡️ Fraud Detection System")
st.write(
    "Enter the transaction details below to check whether the "
    "transaction is predicted to be genuine or potentially fraudulent."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}\n\n"
        "First run the final model-saving cell in your Jupyter Notebook. "
        "Then place the generated .joblib file in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# Transaction inputs
# --------------------------------------------------
st.subheader("Transaction Details")

with st.form("transaction_form"):
    step = st.number_input(
        "Step",
        min_value=1,
        value=1,
        step=1,
        help="Time step of the transaction, as represented in the dataset."
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        options=TRANSACTION_TYPES
    )

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f"
    )

    oldbalance_org = st.number_input(
        "Sender's Old Balance (₹)",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        format="%.2f"
    )

    oldbalance_dest = st.number_input(
        "Receiver's Old Balance (₹)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.2f"
    )

    submitted = st.form_submit_button(
        "Check Transaction",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    transaction = pd.DataFrame(
        [[
            step,
            transaction_type,
            amount,
            oldbalance_org,
            oldbalance_dest
        ]],
        columns=FEATURES
    )

    try:
        prediction = model.predict(transaction)[0]

        # Get the probability corresponding to the fraud class (1).
        fraud_class_index = list(model.classes_).index(1)
        fraud_probability = model.predict_proba(transaction)[0][fraud_class_index]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Potentially Fraudulent Transaction")
            st.write(
                "The model has classified this transaction as potentially fraudulent."
            )
        else:
            st.success("✅ Genuine Transaction")
            st.write(
                "The model has classified this transaction as genuine."
            )

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

        st.progress(float(fraud_probability))

        st.caption(
            "Note: This is a machine learning prediction, not a confirmation "
            "of fraud. Transactions should be reviewed according to the "
            "organization's verification process."
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Fraud Detection | Machine Learning Capstone Project")
