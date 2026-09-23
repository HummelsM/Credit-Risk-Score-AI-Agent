from flask import Flask, request, jsonify
import pandas as pd
import joblib
import shap


# ============================================================
# Load trained pipeline
# ============================================================

pipeline = joblib.load("credit_risk_pipeline.pkl")

# Extract XGBoost model and preprocessing component
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

# SHAP explainer for the XGBoost model
explainer = shap.TreeExplainer(model)


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Required input fields
# ============================================================

REQUIRED_FIELDS = [
    "Checking_Status",
    "Duration",
    "Credit_History",
    "Purpose",
    "Credit_Amount",
    "Savings_Status",
    "Employment",
    "Installment_Commitment",
    "Relationship_Status",
    "Other_Parties",
    "Residence_Since",
    "Property_Type",
    "Age",
    "Other_Payment_Plans",
    "Living_Status",
    "Existing_Credits",
    "Job",
    "Dependents",
    "Telephone",
    "Foreign_Worker"
]


# ============================================================
# Feature engineering
# ============================================================

def engineer_features(applicant):

    applicant = applicant.copy()

    # Monthly credit burden
    applicant["Monthly_Burden"] = (
        applicant["Credit_Amount"] /
        applicant["Duration"]
    )

    # Whether applicant has multiple existing credits
    applicant["Multiple_Credits"] = (
        applicant["Existing_Credits"] > 1
    ).astype(int)

    # Age group
    applicant["Age_Group"] = pd.cut(
        applicant["Age"],
        bins=[18, 30, 45, 60, 100],
        labels=[
            "Young",
            "Middle",
            "Senior",
            "Retired"
        ],
        include_lowest=True
    )

    return applicant


# ============================================================
# SHAP explanation
# ============================================================

def get_shap_explanation(applicant):

    # Apply the same preprocessing used during training
    X_processed = preprocessor.transform(applicant)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)

    # XGBoost binary classification normally gives
    # one SHAP value per transformed feature.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    # Get names after OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Create a dataframe containing feature names and SHAP values
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    # Sort by absolute contribution
    shap_df["absolute_shap"] = (
        shap_df["shap_value"].abs()
    )

    shap_df = shap_df.sort_values(
        "absolute_shap",
        ascending=False
    )

    # Return top five factors
    top_features = shap_df.head(5)

    explanations = []

    for _, row in top_features.iterrows():

        feature = row["feature"]
        value = float(row["shap_value"])

        if value > 0:
            direction = "increased"
        else:
            direction = "decreased"

        explanations.append({
            "feature": feature,
            "impact": round(abs(value), 4),
            "direction": direction
        })

    return explanations


# ============================================================
# Credit assessment endpoint
# ============================================================

@app.route("/api/assess-credit", methods=["POST"])
def assess_credit():

    try:

        # ----------------------------------------------------
        # Get JSON request
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400


        # ----------------------------------------------------
        # Check required fields
        # ----------------------------------------------------

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:

            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400


        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        applicant = pd.DataFrame([data])


        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        applicant = engineer_features(applicant)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = pipeline.predict(applicant)[0]

        probabilities = pipeline.predict_proba(applicant)[0]


        # Your target encoding was:
        #
        # good = 1
        # bad  = 0
        #
        # Therefore probabilities[0] = bad
        # and probabilities[1] = good.

        probability_bad = float(probabilities[0])
        probability_good = float(probabilities[1])


        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = round(
            probability_bad * 100,
            2
        )


        # ----------------------------------------------------
        # Business decision
        # ----------------------------------------------------

        if risk_score < 30:

            decision = "Approved"

        elif risk_score < 70:

            decision = "Manual Review"

        else:

            decision = "Rejected"


        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if risk_score < 30:

            risk_level = "Low"

        elif risk_score < 70:

            risk_level = "Medium"

        else:

            risk_level = "High"


        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        shap_explanation = get_shap_explanation(
            applicant
        )


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "prediction": int(prediction),

            "probability_good": round(
                probability_good,
                4
            ),

            "probability_bad": round(
                probability_bad,
                4
            ),

            "risk_score": risk_score,

            "risk_level": risk_level,

            "decision": decision,

            "top_risk_factors": shap_explanation

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Health check
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "Credit Risk Assessment API"
    })


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
