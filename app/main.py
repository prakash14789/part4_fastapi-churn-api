from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

THRESHOLD = 0.40

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)
@app.get("/")
def root():
    return {
  "message": "Customer Churn Prediction API",
  "docs": "/docs"
}
# -------------------------
# Input Schema
# -------------------------

class CustomerFeatures(BaseModel):

    city_tier: str
    age_group: str
    acquisition_channel: str
    loyalty_tier: str | None = None
    preferred_category: str
    marketing_consent: str

    recency_days: int = Field(ge=0)
    frequency_180d: int = Field(ge=0)
    monetary_180d: float = Field(ge=0)

    return_rate_180d: float = Field(ge=0)
    avg_discount_pct_180d: float = Field(ge=0)

    avg_rating_180d: float = Field(ge=0)
    category_diversity_180d: int = Field(ge=0)

    ticket_count_90d: int = Field(ge=0)

    negative_ticket_rate_90d: float = Field(ge=0)

    avg_resolution_hours_90d: float = Field(ge=0)

    days_since_signup: int = Field(ge=0)

    sessions_30d: int = Field(ge=0)
    product_views_30d: int = Field(ge=0)
    cart_adds_30d: int = Field(ge=0)
    wishlist_adds_30d: int = Field(ge=0)
    abandoned_carts_30d: int = Field(ge=0)

    email_opens_30d: int = Field(ge=0)
    campaign_clicks_30d: int = Field(ge=0)

    last_visit_days_ago: int = Field(ge=0)

# -------------------------
# Risk Explanation
# -------------------------

def generate_explanation(customer):

    reasons = []

    if customer.recency_days > 90:
        reasons.append("customer has not purchased recently")

    if customer.sessions_30d < 5:
        reasons.append("low recent engagement")

    if customer.ticket_count_90d > 2:
        reasons.append("multiple support tickets")

    if customer.campaign_clicks_30d == 0:
        reasons.append("no campaign engagement")

    if len(reasons) == 0:
        reasons.append("healthy customer activity observed")

    return ", ".join(reasons)

# -------------------------
# Health Endpoint
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Predict Endpoint
# -------------------------

@app.post("/predict")
def predict(customer: CustomerFeatures):

    df = pd.DataFrame([customer.model_dump()])

    probability = model.predict_proba(df)[0][1]

    predicted_class = int(probability >= THRESHOLD)

    risk_level = (
        "high"
        if probability >= 0.70
        else "medium"
        if probability >= 0.40
        else "low"
    )

    return {
        "churn_probability": round(float(probability), 4),
        "predicted_class": predicted_class,
        "risk_level": risk_level,
        "risk_explanation": generate_explanation(customer)
    }

# -------------------------
# Batch Predict
# -------------------------

@app.post("/batch_predict")
def batch_predict(customers: List[CustomerFeatures]):

    results = []

    for customer in customers:

        df = pd.DataFrame([customer.model_dump()])

        probability = model.predict_proba(df)[0][1]

        predicted_class = int(probability >= THRESHOLD)

        risk_level = (
            "high"
            if probability >= 0.70
            else "medium"
            if probability >= 0.40
            else "low"
        )

        results.append({
            "churn_probability": round(float(probability), 4),
            "predicted_class": predicted_class,
            "risk_level": risk_level,
            "risk_explanation": generate_explanation(customer)
        })

    return results