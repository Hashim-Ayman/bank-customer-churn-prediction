"""
Streamlit version of the FastAPI churn-prediction app.

Instead of exposing HTTP endpoints (/predict/forest, /predict/xgboost),
this calls `predict_new` directly in-process, reusing the same
preprocessor/model objects from utils.config.

Run with:
    streamlit run streamlit_app.py
(from the project root, same place you'd run `python -m utils.inference`)
"""

import streamlit as st
import pandas as pd
from pydantic_core import PydanticUndefined

from utils.inference import predict_new
from utils.config import APP_NAME, VERSION, preprocessor, forest_model, xgboost_model
from utils.CustomerData import CustomerData

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="centered")

st.title(f"{APP_NAME}")
st.caption(f"v{VERSION}")

MODELS = {
    "Random Forest": forest_model,
    "XGBoost": xgboost_model,
}


def build_input_form(model_cls) -> dict:
    """
    Dynamically render one input widget per field on the CustomerData
    pydantic model, so the form doesn't need to be manually kept in
    sync with the schema.
    """
    values = {}
    for name, field in model_cls.model_fields.items():
        label = name.replace("_", " ").title()
        default = field.default if field.default is not PydanticUndefined else None
        annotation = field.annotation

        # bool fields -> checkbox
        if annotation is bool:
            values[name] = st.checkbox(
                label, value=bool(default) if default is not None else False
            )

        # int fields -> number_input (integer step)
        elif annotation is int:
            values[name] = st.number_input(
                label,
                value=int(default) if default is not None else 0,
                step=1,
                format="%d",
            )

        # float fields -> number_input (float step)
        elif annotation is float:
            values[name] = st.number_input(
                label, value=float(default) if default is not None else 0.0
            )

        # Literal / enum-like fields -> selectbox, if metadata exposes choices
        elif hasattr(annotation, "__args__") and annotation.__args__:
            options = list(annotation.__args__)
            idx = options.index(default) if default in options else 0
            values[name] = st.selectbox(label, options, index=idx)

        # fallback -> text input
        else:
            values[name] = st.text_input(
                label, value=str(default) if default is not None else ""
            )

    return values


with st.form("prediction_form"):
    model_choice = st.selectbox("Model", list(MODELS.keys()))
    st.divider()
    st.subheader("Customer details")
    input_values = build_input_form(CustomerData)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        customer = CustomerData(**input_values)
        selected_model = MODELS[model_choice]

        result = predict_new(
            data=customer, preprocessor=preprocessor, model=selected_model
        )

        st.divider()
        churn = result.get("churn_prediction")
        prob = result.get("churn_probability")

        if churn:
            st.error(f"⚠️ Predicted: **Churn** (probability: {prob:.2%})")
        else:
            st.success(f"✅ Predicted: **No churn** (probability of churn: {prob:.2%})")

        with st.expander("Raw result"):
            st.json(result)

    except Exception as e:
        st.exception(e)
