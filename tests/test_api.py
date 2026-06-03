import sys
import os

sys.path.append(os.path.abspath("."))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():

    payload = {
        "city_tier": "Tier 1",
        "age_group": "25-34",
        "acquisition_channel": "Instagram",
        "loyalty_tier": None,
        "preferred_category": "Hair Care",
        "marketing_consent": "Yes",
        "recency_days": 40,
        "frequency_180d": 3,
        "monetary_180d": 1500,
        "return_rate_180d": 0.1,
        "avg_discount_pct_180d": 0.2,
        "avg_rating_180d": 4.2,
        "category_diversity_180d": 3,
        "ticket_count_90d": 1,
        "negative_ticket_rate_90d": 0,
        "avg_resolution_hours_90d": 4,
        "days_since_signup": 300,
        "sessions_30d": 15,
        "product_views_30d": 40,
        "cart_adds_30d": 5,
        "wishlist_adds_30d": 2,
        "abandoned_carts_30d": 1,
        "email_opens_30d": 5,
        "campaign_clicks_30d": 2,
        "last_visit_days_ago": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "predicted_class" in result
    assert "risk_level" in result


def test_batch_predict():

    payload = [{
        "city_tier": "Tier 1",
        "age_group": "25-34",
        "acquisition_channel": "Instagram",
        "loyalty_tier": None,
        "preferred_category": "Hair Care",
        "marketing_consent": "Yes",
        "recency_days": 40,
        "frequency_180d": 3,
        "monetary_180d": 1500,
        "return_rate_180d": 0.1,
        "avg_discount_pct_180d": 0.2,
        "avg_rating_180d": 4.2,
        "category_diversity_180d": 3,
        "ticket_count_90d": 1,
        "negative_ticket_rate_90d": 0,
        "avg_resolution_hours_90d": 4,
        "days_since_signup": 300,
        "sessions_30d": 15,
        "product_views_30d": 40,
        "cart_adds_30d": 5,
        "wishlist_adds_30d": 2,
        "abandoned_carts_30d": 1,
        "email_opens_30d": 5,
        "campaign_clicks_30d": 2,
        "last_visit_days_ago": 3
    }]

    response = client.post("/batch_predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result, list)
    assert len(result) > 0

def test_invalid_predict():

    response = client.post(
        "/predict",
        json={
            "city_tier": "Tier 1"
        }
    )

    assert response.status_code == 422



