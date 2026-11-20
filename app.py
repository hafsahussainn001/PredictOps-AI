from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "predictops_pipeline.pkl"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

st.set_page_config(page_title="PredictOps AI", page_icon="⚙️", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train.py` before starting the app."
        )
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)
    return model, metadata

st.title("⚙️ PredictOps AI")
st.subheader("Predictive Maintenance & Explainable ML")
st.write(
    "Enter machine sensor readings to estimate the model's machine-failure risk. "
    "This is a decision-support estimate, not a guaranteed diagnosis."
)

try:
    model, metadata = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.caption(f"Model: {metadata['model_name']} | Decision threshold: {metadata['threshold']:.3f}")

with st.form("machine_input_form"):
    st.markdown("### Machine information")
    product_type = st.selectbox("Product type", ["L", "M", "H"])

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input(
            "Air temperature (K)", min_value=250.0, max_value=350.0,
            value=300.0, step=0.1
        )
        rotational_speed = st.number_input(
            "Rotational speed (rpm)", min_value=500, max_value=5000,
            value=1500, step=10
        )
        tool_wear = st.number_input(
            "Tool wear (minutes)", min_value=0, max_value=300,
            value=100, step=1
        )
    with col2:
        process_temp = st.number_input(
            "Process temperature (K)", min_value=250.0, max_value=400.0,
            value=310.0, step=0.1
        )
        torque = st.number_input(
            "Torque (Nm)", min_value=0.0, max_value=100.0,
            value=40.0, step=0.1
        )

    submitted = st.form_submit_button("Predict machine failure", type="primary")

if submitted:
    if process_temp < air_temp:
        st.error("Process temperature should not be lower than air temperature for this dataset.")
    else:
        input_data = pd.DataFrame([{
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])

        try:
            probability = float(model.predict_proba(input_data)[0, 1])
            threshold = float(metadata["threshold"])
            prediction = int(probability >= threshold)

            st.markdown("---")
            st.markdown("### Prediction result")
            st.metric("Estimated failure probability", f"{probability:.1%}")

            if prediction == 1:
                st.error("Higher predicted failure risk — inspection is recommended.")
            else:
                st.success("Lower predicted failure risk under the selected threshold.")

            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                f"Decision rule: failure is flagged when estimated probability "
                f"is at least {threshold:.3f}."
            )
            st.warning(
                "This model is trained on a synthetic dataset. Its output is not "
                "a safety guarantee or a substitute for qualified maintenance inspection."
            )
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")

st.markdown("---")
st.caption("PredictOps AI | Week 04 AI Engineering Capstone")
