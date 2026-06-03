# Customer Churn Prediction API

## Project Overview

This project provides a FastAPI-based machine learning service for predicting customer churn risk.

The application loads a trained Random Forest model and exposes REST API endpoints that allow CRM or retention systems to:

* Check API health
* Predict churn risk for a single customer
* Predict churn risk for multiple customers
* Return churn probability
* Return churn classification
* Return risk level and explanation

---

## Project Structure

```text
part4_fastapi_churn_api
│
├── app
│   ├── __init__.py
│   └── main.py
│
├── data
│   └── rfm_modeling_snapshot.csv
│
├── tests
│   └── test_api.py
│
├── train_model.py
├── model.pkl
├── metrics.json
├── requirements.txt
├── monitoring_plan.md
├── README.md
├── Dockerfile
├── .dockerignore
├── .gitignore
└── venv
```

---

## Dataset

The model was trained using the provided `rfm_modeling_snapshot.csv` dataset.

### Target Variable

```text
churn_next_60d
```

The dataset contains customer transaction, engagement, support, and behavioral features calculated before the customer snapshot date.

---

## Model Information

### Model Used

```text
Random Forest Classifier
```

### Evaluation Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.7946 |
| Precision | 0.8235 |
| Recall    | 0.7500 |
| F1 Score  | 0.7850 |
| ROC-AUC   | 0.8807 |



### Threshold Selection

The model was evaluated across multiple classification thresholds.

A threshold of:

0.40

was selected for deployment because it improved churn detection performance.

| Threshold | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| 0.40      | 0.799     | 0.875  | 0.835    |

Business Rationale:

For churn prediction, missing customers who are likely to churn can be more costly than contacting a few additional customers. A threshold of 0.40 increased recall from 75.0% to 87.5% while maintaining strong precision, making it more suitable for retention campaigns.


## Installation

### Clone Repository

```bash
git clone <repository-url>
cd part4_fastapi_churn_api
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

To retrain the model and generate a new model artifact:

```bash
python train_model.py
```

Generated files:

```text
model.pkl
metrics.json
```

---

## Run API

Start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Docker Setup

### Build Docker Image

```bash
docker build -t churn-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 churn-api
```

### Access API Documentation

```text
http://localhost:8000/docs
```

This Docker setup provides an additional reproducible deployment option and allows the application to run consistently across different environments.

---

# API Endpoints

## GET /

Returns API status information.


### Example Response

```json
{
  "message": "Customer Churn Prediction API",
  "docs": "/docs"
}
```


---
## GET /health

Health check endpoint.

### Example Response

```json
{
  "status": "ok"
}
```

---

## POST /predict

Predict churn risk for a single customer.

### Sample Request

```json
{
  "city_tier": "Tier 1",
  "age_group": "25-34",
  "acquisition_channel": "Instagram",
  "loyalty_tier": null,
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
```

### Sample Response

```json
{
  "churn_probability": 0.1263,
  "predicted_class": 0,
  "risk_level": "low",
  "risk_explanation": "healthy customer activity observed"
}
```

---

## POST /batch_predict

Predict churn risk for multiple customers.

### Sample Request

```json
[
  {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Instagram",
    "loyalty_tier": null,
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
  },
  {
    "city_tier": "Tier 2",
    "age_group": "35-44",
    "acquisition_channel": "Facebook",
    "loyalty_tier": null,
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 120,
    "frequency_180d": 1,
    "monetary_180d": 500,
    "return_rate_180d": 0.3,
    "avg_discount_pct_180d": 0.5,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 1,
    "ticket_count_90d": 4,
    "negative_ticket_rate_90d": 0.5,
    "avg_resolution_hours_90d": 48,
    "days_since_signup": 100,
    "sessions_30d": 2,
    "product_views_30d": 5,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 2,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 60
  }
]
```

### Sample Response

```json
[
  {
    "churn_probability": 0.1263,
    "predicted_class": 0,
    "risk_level": "low",
    "risk_explanation": "healthy customer activity observed"
  },
  {
    "churn_probability": 0.6944,
    "predicted_class": 1,
    "risk_level": "medium",
    "risk_explanation": "customer has not purchased recently, low recent engagement, multiple support tickets, no campaign engagement"
  }
]
```

---

## Running Tests

Execute all API tests:

```bash
pytest
```

Expected Result:

```text
3 passed
```

---

## Monitoring and Responsible Use

See:

```text
monitoring_plan.md
```

for:

* Data drift monitoring
* Prediction distribution monitoring
* Business outcome monitoring
* API monitoring
* Retraining triggers
* Responsible use guidelines

---

## Reproducibility

This project supports reproducibility through:

* requirements.txt for dependency management
* train_model.py for model retraining
* model.pkl for deployment-ready inference
* Dockerfile for containerized deployment
* README instructions for setup and execution

A new user can reproduce the complete workflow by:

1. Installing dependencies
2. Training the model
3. Running the FastAPI service
4. Executing API tests

---

## Notes

* The API uses a trained Random Forest model.
* Pydantic validation is used to validate request payloads.
* The repository is fully reproducible using requirements.txt.
* Docker support is included through the provided Dockerfile.
* The project can be reproduced either through requirements.txt or Docker.
* The model can be retrained at any time using train_model.py.
* The API returns probability-based churn predictions suitable for CRM and retention workflows.
