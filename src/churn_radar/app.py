"""Gradio demo UI: score one customer and explain the risk in plain language.

python -m churn_radar.app    # opens at http://127.0.0.1:7860
"""

from __future__ import annotations

import gradio as gr
import joblib

from churn_radar import config
from churn_radar.explain import recommended_action, risk_drivers, risk_tier
from churn_radar.features import customers_to_frame

YN = ["Yes", "No"]
ADD_ON = ["Yes", "No", "No internet service"]
TIER_LABEL = {"high": "HIGH RISK", "medium": "MEDIUM RISK", "low": "LOW RISK"}

FIELDS = [
    ("gender", gr.Radio, {"choices": ["Female", "Male"], "value": "Female", "label": "Gender"}),
    ("SeniorCitizen", gr.Radio, {"choices": YN, "value": "No", "label": "Senior citizen"}),
    ("Partner", gr.Radio, {"choices": YN, "value": "No", "label": "Has partner"}),
    ("Dependents", gr.Radio, {"choices": YN, "value": "No", "label": "Has dependents"}),
    ("tenure", gr.Slider, {"minimum": 0, "maximum": 72, "value": 3, "step": 1,
                           "label": "Tenure (months)"}),
    ("PhoneService", gr.Radio, {"choices": YN, "value": "Yes", "label": "Phone service"}),
    ("MultipleLines", gr.Radio, {"choices": ["Yes", "No", "No phone service"], "value": "No",
                                 "label": "Multiple lines"}),
    ("InternetService", gr.Radio, {"choices": ["DSL", "Fiber optic", "No"],
                                   "value": "Fiber optic", "label": "Internet service"}),
    ("OnlineSecurity", gr.Radio, {"choices": ADD_ON, "value": "No", "label": "Online security"}),
    ("OnlineBackup", gr.Radio, {"choices": ADD_ON, "value": "No", "label": "Online backup"}),
    ("DeviceProtection", gr.Radio, {"choices": ADD_ON, "value": "No",
                                    "label": "Device protection"}),
    ("TechSupport", gr.Radio, {"choices": ADD_ON, "value": "No", "label": "Tech support"}),
    ("StreamingTV", gr.Radio, {"choices": ADD_ON, "value": "No", "label": "Streaming TV"}),
    ("StreamingMovies", gr.Radio, {"choices": ADD_ON, "value": "No",
                                   "label": "Streaming movies"}),
    ("Contract", gr.Radio, {"choices": ["Month-to-month", "One year", "Two year"],
                            "value": "Month-to-month", "label": "Contract"}),
    ("PaperlessBilling", gr.Radio, {"choices": YN, "value": "Yes", "label": "Paperless billing"}),
    ("PaymentMethod", gr.Radio, {"choices": ["Electronic check", "Mailed check",
                                             "Bank transfer (automatic)",
                                             "Credit card (automatic)"],
                                 "value": "Electronic check", "label": "Payment method"}),
    ("MonthlyCharges", gr.Slider, {"minimum": 18, "maximum": 120, "value": 80, "step": 0.5,
                                   "label": "Monthly charges ($)"}),
]  # fmt: skip

MODEL = joblib.load(config.MODEL_PATH)


def score(*values):
    customer = dict(zip([name for name, _, _ in FIELDS], values, strict=True))
    customer["SeniorCitizen"] = int(customer["SeniorCitizen"] == "Yes")
    prob = float(MODEL.predict_proba(customers_to_frame([customer]))[0, 1])
    drivers = risk_drivers(customer)
    return (
        f"{TIER_LABEL[risk_tier(prob)]}: churn probability {prob:.0%}",
        "\n".join(f"- {d}" for d in drivers) if drivers else "No major risk flags",
        recommended_action(prob),
    )


demo = gr.Interface(
    fn=score,
    inputs=[component(**kwargs) for _, component, kwargs in FIELDS],
    outputs=[
        gr.Textbox(label="Risk verdict"),
        gr.Textbox(label="Why (risk drivers)"),
        gr.Textbox(label="Recommended retention action"),
    ],
    title="ChurnRadar: Customer Churn Early-Warning System",
    description="Score any customer profile in real time and get a plain-language "
    "explanation plus a recommended retention action.",
)

if __name__ == "__main__":
    demo.launch()
