# AI-Powered Credit Risk Assessment & Automation

An end-to-end machine learning and workflow automation project that assesses applicant credit risk, generates interpretable predictions, and automates the processing and storage of assessment results using Google Sheets, Make, and a deployed Flask API.

## Overview

This project demonstrates how machine learning and business process automation can be combined to streamline credit risk assessment. An XGBoost classification model evaluates applicant information and produces risk scores, risk classifications, decision recommendations, and predicted probabilities.

A Flask REST API deployed on Render serves the model, while Make orchestrates the workflow between Google Sheets and the API. SHAP-based explanations provide additional insight into the factors influencing individual predictions.

The workflow has successfully processed **500 applicant records** in the current demonstration, with assessment results automatically written back to Google Sheets.

## Key Features

* **Machine Learning:** XGBoost classification for predicting credit risk.
* **Automated Risk Scoring:** Generates risk scores, risk levels, and decision recommendations.
* **Explainable AI:** Uses SHAP-based feature contributions to help interpret model predictions.
* **REST API:** Serves predictions through a Flask API deployed on Render.
* **Workflow Automation:** Uses Make to connect Google Sheets with the assessment API.
* **Automated Results Management:** Writes assessment outputs back to Google Sheets.
* **Analytics-Ready Data:** Structures results for further analysis and dashboard visualization.

## System Architecture

```text
Applicant Data
  (Google Sheets)
        |
        v
Make Automation Workflow
        |
        v
Flask REST API
  (Deployed on Render)
        |
        v
XGBoost Classification Model
        |
        v
Risk Scoring + SHAP Explanations
        |
        v
Structured Assessment Results
        |
        v
Google Sheets
        |
        v
Dashboard / Analytics
```

## Technology Stack

| Category               | Technologies                    |
| ---------------------- | ------------------------------- |
| Programming Language   | Python                          |
| Machine Learning       | XGBoost, scikit-learn           |
| Explainable AI         | SHAP                            |
| API Development        | Flask, REST API                 |
| Data Processing        | Pandas, NumPy                   |
| Deployment             | Render                          |
| Workflow Automation    | Make                            |
| Data Input and Storage | Google Sheets                   |
| Analytics              | Dashboard-ready structured data |

## Assessment Outputs

For each processed applicant, the system can generate:

* **Risk Score:** A score derived from the model's estimated probability of bad credit.
* **Risk Level:** A low, medium, or high risk classification based on configured thresholds.
* **Decision Recommendation:** An illustrative recommendation such as Approved, Manual Review, or Rejected.
* **Predicted Probabilities:** Estimated probabilities for the good-credit and bad-credit classes.
* **Risk Factors:** Feature contributions that help explain the model's prediction.
* **Assessment Summary:** A human-readable overview of the assessment and its main contributing factors.

## Workflow

1. Applicant records are stored in Google Sheets.
2. Make retrieves applicant data and sends each record to the deployed Flask API.
3. The API validates the input and prepares the features required by the model.
4. The XGBoost pipeline generates a prediction and class probabilities.
5. SHAP-based analysis identifies features contributing to the prediction.
6. The API returns structured assessment results.
7. Make writes the results back to Google Sheets for review and further analysis.

## Results

The project successfully processed **500 applicant records** through the automated credit risk assessment workflow.

Key achievements:

* **500 records processed:** Applied the automated assessment workflow to 500 applicant records of which **348 (69.6%) were approved**, **104 (20.8%) rejected** and **48 (9.6%) sent to manual review**.
* **End-to-end integration:** Connected Google Sheets, Make, a Flask REST API, and an XGBoost classification model.
* **Automated risk assessment:** Generated risk scores, risk classifications, decision recommendations, and predicted probabilities.
* **Explainable AI:** Integrated SHAP-based explanations to identify factors contributing to individual model predictions.
* **Automated results management:** Wrote assessment outputs back to Google Sheets for review and analysis.
* **Analytics readiness:** Organized assessment results for downstream reporting and dashboard visualization.

These results demonstrate the practical integration of machine learning and workflow automation to process applicant data at scale within a portfolio demonstration.


## Potential Applications

This project demonstrates techniques relevant to:

* Credit risk analytics and decision-support prototypes.
* AI-powered business process automation.
* Automated data processing and reporting.
* Explainable machine learning applications.
* API-based integration of predictive models with business workflows.

## Limitations and Responsible Use

This project is an educational and portfolio demonstration using a credit-risk dataset. The decision thresholds are configurable demonstration rules and should not be interpreted as validated financial underwriting criteria.

Model predictions may contain bias or errors and should not be used as the sole basis for real lending decisions. Production deployment would require appropriate validation, fairness assessment, security controls, regulatory review, and human oversight.

## Future Improvements

* Build an interactive dashboard for risk distribution, applicant-level results, and contributing factors.
* Evaluate model performance using appropriate classification metrics and validation methods.
* Add API authentication, input safeguards, logging, and monitoring.
* Expand batch-processing capabilities and workflow error handling.
* Evaluate fairness, calibration, and model stability before any real-world use.

## Project Summary

This project showcases the integration of **machine learning, explainable AI, API deployment, and no-code workflow automation** into a single working application. It demonstrates how predictive models can be connected to existing business tools to automate data processing and make model outputs more accessible for analysis.

**Disclaimer:** This project is a portfolio prototype and does not constitute financial advice or a production-ready credit decision system.
