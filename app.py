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
# Numeric fields
# ============================================================

NUMERIC_FIELDS = [
    "Duration",
    "Credit_Amount",
    "Installment_Commitment",
    "Residence_Since",
    "Age",
    "Existing_Credits",
    "Dependents"
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

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    for field in NUMERIC_FIELDS:

        applicant[field] = pd.to_numeric(
            applicant[field],
            errors="raise"
        )


    # --------------------------------------------------------
    # Monthly credit burden
    # --------------------------------------------------------

    applicant["Monthly_Burden"] = (
        applicant["Credit_Amount"]
        / applicant["Duration"]
    )


    # --------------------------------------------------------
    # Multiple existing credits
    # --------------------------------------------------------

    applicant["Multiple_Credits"] = (
        applicant["Existing_Credits"] > 1
    ).astype(int)


    # --------------------------------------------------------
    # Age group
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Apply trained preprocessing
    # --------------------------------------------------------

    X_processed = preprocessor.transform(
        applicant
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X_processed
    )


    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        shap_values = shap_values[0]


    shap_values = shap_values[0]


    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )


    # --------------------------------------------------------
    # Create SHAP dataframe
    # --------------------------------------------------------

    shap_df = pd.DataFrame({

        "feature": feature_names,

        "shap_value": shap_values

    })


    # --------------------------------------------------------
    # Calculate absolute contribution
    # --------------------------------------------------------

    shap_df["absolute_shap"] = (
        shap_df["shap_value"].abs()
    )


    # --------------------------------------------------------
    # Sort by importance
    # --------------------------------------------------------

    shap_df = shap_df.sort_values(
        "absolute_shap",
        ascending=False
    )


    # --------------------------------------------------------
    # Top five factors
    # --------------------------------------------------------

    top_features = shap_df.head(5)


    explanations = []


    for _, row in top_features.iterrows():

        feature = row["feature"]

        value = float(
            row["shap_value"]
        )


        if value > 0:

            direction = "increased"

        else:

            direction = "decreased"


        explanations.append({

            "feature": feature,

            "impact": round(
                abs(value),
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

    data = request.get_json(
        silent=True
    )


    return jsonify({

        "success": True,

        "received_data": data,

        "received_fields": (

            list(data.keys())

            if isinstance(data, dict)

            else []

        ),

        "content_type":
            request.content_type

    })


# ============================================================
# Credit assessment endpoint
# ============================================================

@app.route(
    "/api/assess-credit",
    methods=["POST"]
)
def assess_credit():

    try:

        # ----------------------------------------------------
        # Receive JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        # ----------------------------------------------------
        # Log incoming request
        # ----------------------------------------------------

        print("=" * 60)
        print("CREDIT ASSESSMENT REQUEST")
        print("=" * 60)

        print("Content-Type:")
        print(request.content_type)

        print("-" * 60)

        print("Received data:")
        print(data)

        print("=" * 60)


        # ----------------------------------------------------
        # Check JSON
        # ----------------------------------------------------

        if not data:

            return jsonify({

                "error":
                    "No JSON data received.",

                "content_type":
                    request.content_type

            }), 400


        # ----------------------------------------------------
        # Check that request is an object
        # ----------------------------------------------------

        if not isinstance(
            data,
            dict
        ):

            return jsonify({

                "error":
                    "JSON payload must be an object.",

                "received_type":
                    type(data).__name__

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

                "error":
                    "Missing required fields.",

                "missing_fields":
                    missing_fields,

                "received_fields":
                    list(data.keys()),

                "received_data":
                    data

            }), 400


        # ----------------------------------------------------
        # Check numeric fields
        # ----------------------------------------------------

        invalid_numeric_fields = []


        for field in NUMERIC_FIELDS:

            try:

                float(data[field])

            except (
                ValueError,
                TypeError
            ):

                invalid_numeric_fields.append(
                    field
                )


        if invalid_numeric_fields:

            return jsonify({

                "error":
                    "Invalid numeric values.",

                "invalid_numeric_fields":
                    invalid_numeric_fields,

                "received_data":
                    {
                        field: data.get(field)
                        for field
                        in invalid_numeric_fields
                    }

            }), 400


        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        applicant = pd.DataFrame(
            [data]
        )


        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        applicant = engineer_features(
            applicant
        )


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = pipeline.predict(
            applicant
        )[0]


        probabilities = pipeline.predict_proba(
            applicant
        )[0]


        # ----------------------------------------------------
        # Probability values
        # ----------------------------------------------------

        # good = 1
        # bad  = 0

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
        # Business decision
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
        # SHAP explanation
        # ----------------------------------------------------

        shap_explanation = (
            get_shap_explanation(
                applicant
            )
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        response = {

            "prediction":
                int(prediction),

            "probability_good":
                round(
                    probability_good,
                    4
                ),

            "probability_bad":
                round(
                    probability_bad,
                    4
                ),

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "decision":
                decision,

            "top_risk_factors":
                shap_explanation

        }


        print("=" * 60)
        print("CREDIT ASSESSMENT RESULT")
        print("=" * 60)
        print(response)
        print("=" * 60)


        return jsonify(
            response
        )


    except Exception as e:

        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(str(e))
        print("=" * 60)


        return jsonify({

            "error":
                str(e),

            "error_type":
                type(e).__name__

        }), 500


# ============================================================
# Health check
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "service":
            "Credit Risk Assessment API"

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
