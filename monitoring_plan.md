# Monitoring Plan

## Data Drift Monitoring

The following features should be monitored regularly:

* recency_days
* frequency_180d
* monetary_180d
* sessions_30d
* campaign_clicks_30d

Significant changes in feature distributions may indicate data drift and require investigation.

## Prediction Distribution Monitoring

Track:

* Average churn probability
* Percentage of customers classified as high risk
* Weekly and monthly prediction trends

Unexpected changes may indicate model degradation.

## Business Outcome Monitoring

Track:

* Actual customer churn rate
* Retention campaign success rate
* Revenue retained through interventions

These metrics help evaluate business impact.

## API Monitoring

Track:

* API uptime
* Response latency
* Failed requests
* Error rates

Alerts should be configured for abnormal behavior.

## Retraining Triggers

Retrain the model when:

* ROC-AUC decreases by more than 10%
* Significant data drift is detected
* Every 90 days as part of scheduled maintenance

## Responsible Use

The model should be used to prioritize retention efforts and support business decision-making.

The model should not be used as the sole basis for customer treatment decisions. Human review and business context should always be considered.

Predictions should not be used to discriminate against customers based on demographic characteristics such as age, location, or other personal attributes.

The model may reflect historical patterns and biases present in the training data. Predictions should therefore be monitored regularly and evaluated for fairness.

Model outputs are probabilistic estimates, not guaranteed outcomes. Predictions should be reviewed alongside human judgment before retention actions are taken.

