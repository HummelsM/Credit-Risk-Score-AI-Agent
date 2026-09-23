from flask import Flask, request, jsonify
import pandas as pd
import joblib
import shap


# ============================================================
# Load trained pipeline
# ============================================================

pipeline = joblib.load("credit_risk_pipeline.pkl")

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

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
# Root endpoint
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "Credit Risk Assessment API",
        "status": "running",
        "available_endpoints": [
            "/",
            "/health",
            "/debug",
            "/api/assess-credit"
        ]
    })


# ============================================================
# Feature engineering
# ============================================================

def engineer_features(applicant):

    applicant = applicant.copy()

    applicant["Monthly_Burden"] = (
        applicant["Credit_Amount"]
        / applicant["Duration"]
    )

    applicant["Multiple_Credits"] = (
        applicant["Existing_Credits"] > 1
    ).astype(int)

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

    X_processed = preprocessor.transform(applicant)

    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    shap_df["absolute_shap"] = (
        shap_df["shap_value"].abs()
    )

    shap_df = shap_df.sort_values(
        "absolute_shap",
        ascending=False
    )

    top_features = shap_df.head(5)

    explanations = []

    for _, row in top_features.iterrows():

        direction = (
            "increased"
            if row["shap_value"] > 0
            else "decreased"
        )

        explanations.append({
            "feature": row["feature"],
            "impact": round(
                abs(float(row["shap_value"])),
                4
            ),
            "direction": direction
        })

    return explanations


# ============================================================
# Debug endpoint
# ============================================================

@app.route("/debug", methods=["POST"])
def debug():

    data = request.get_json(silent=True)

    return jsonify({
        "success": True,
        "received_data": data,
        "received_fields": (
            list(data.keys())
            if isinstance(data, dict)
            else []
        ),
        "content_type": request.content_type
    })


# ============================================================
# Credit assessment endpoint
# ============================================================

@app.route("/api/assess-credit", methods=["POST"])
def assess_credit():

    try:

        # ----------------------------------------------------
        # Receive JSON
        # ----------------------------------------------------

        data = request.get_json(silent=True)

        print("=" * 60)
        print("REQUEST RECEIVED")
        print("=" * 60)
        print("Headers:")
        print(dict(request.headers))
        print("-" * 60)
        print("JSON Data:")
        print(data)
        print("=" * 60)

        if not data:

            return jsonify({
                "error": "No JSON data received.",
                "content_type": request.content_type
            }), 400


        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:

            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields,
                "received_fields": list(data.keys()),
                "received_data": data
            }), 400


        # ----------------------------------------------------
        # Convert to DataFrame
        # ----------------------------------------------------

        applicant = pd.DataFrame([data])


        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        applicant = engineer_features(
            applicant
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = pipeline.predict(
            applicant
        )[0]

        probabilities = pipeline.predict_proba(
            applicant
        )[0]

        probability_bad = float(
            probabilities[0]
        )

        probability_good = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = round(
            probability_bad * 100,
            2
        )


        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if risk_score < 30:

            decision = "Approved"
            risk_level = "Low"

        elif risk_score < 70:

            decision = "Manual Review"
            risk_level = "Medium"

        else:

            decision = "Rejected"
            risk_level = "High"


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_explanation = (
            get_shap_explanation(
                applicant
            )
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "prediction": int(
                prediction
            ),

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

            "top_risk_factors":
                shap_explanation

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
