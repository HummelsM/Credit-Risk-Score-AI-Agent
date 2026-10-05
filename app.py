from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
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
# Categorical fields used by preprocessing pipeline
# ============================================================

CATEGORICAL_FIELDS = [
    "Checking_Status",
    "Credit_History",
    "Purpose",
    "Savings_Status",
    "Employment",
    "Relationship_Status",
    "Other_Parties",
    "Living_Status",
    "Job",
    "Telephone",
    "Foreign_Worker",
    "Property_Type",
    "Age_Group",
    "Other_Payment_Plans"
]


# ============================================================
# Human-readable feature names
# ============================================================

FEATURE_LABELS = {
    "Checking_Status": "Checking account status",
    "Duration": "Loan duration",
    "Credit_History": "Credit history",
    "Purpose": "Loan purpose",
    "Credit_Amount": "Credit amount",
    "Savings_Status": "Savings level",
    "Employment": "Employment history",
    "Installment_Commitment": "Installment commitment",
    "Relationship_Status": "Personal status",
    "Other_Parties": "Other liable parties",
    "Residence_Since": "Length of residence",
    "Property_Type": "Property ownership",
    "Age": "Applicant age",
    "Age_Group": "Applicant age group",
    "Other_Payment_Plans": "Other payment plans",
    "Living_Status": "Housing status",
    "Existing_Credits": "Existing credits",
    "Job": "Employment type",
    "Dependents": "Number of dependents",
    "Telephone": "Telephone status",
    "Foreign_Worker": "Foreign worker status",
    "Monthly_Burden": "Monthly repayment burden",
    "Multiple_Credits": "Multiple existing credits"
}


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
    # Prevent invalid duration
    # --------------------------------------------------------

    if (applicant["Duration"] <= 0).any():

        raise ValueError(
            "Duration must be greater than zero."
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
# Convert transformed feature name back to original feature
# ============================================================

def get_original_feature_name(transformed_name):

    # --------------------------------------------------------
    # Numeric/remainder features
    #
    # Example:
    # remainder__Duration
    # becomes:
    # Duration
    # --------------------------------------------------------

    if transformed_name.startswith(
        "remainder__"
    ):

        return transformed_name.replace(
            "remainder__",
            "",
            1
        )

    # --------------------------------------------------------
    # One-hot categorical features
    #
    # Example:
    # cat__Checking_Status_<0
    #
    # becomes:
    # Checking_Status
    # --------------------------------------------------------

    if transformed_name.startswith(
        "cat__"
    ):

        clean_name = transformed_name.replace(
            "cat__",
            "",
            1
        )

        # Important:
        # Some feature names contain underscores,
        # so we match against known original column names
        # rather than simply splitting on "_".

        sorted_fields = sorted(
            CATEGORICAL_FIELDS,
            key=len,
            reverse=True
        )

        for field in sorted_fields:

            prefix = field + "_"

            if clean_name.startswith(
                prefix
            ):

                return field

        return clean_name

    return transformed_name


# ============================================================
# Human-readable applicant value
# ============================================================

def get_display_value(
    applicant,
    feature
):

    if feature not in applicant.columns:

        return None

    value = applicant.iloc[0][feature]

    if pd.isna(value):

        return None

    # --------------------------------------------------------
    # Convert numpy / pandas values into JSON-safe values
    # --------------------------------------------------------

    if isinstance(
        value,
        np.generic
    ):

        value = value.item()

    # --------------------------------------------------------
    # Format engineered fields
    # --------------------------------------------------------

    if feature == "Monthly_Burden":

        return round(
            float(value),
            2
        )

    if feature == "Multiple_Credits":

        return (
            "Yes"
            if int(value) == 1
            else "No"
        )

    return str(value)


# ============================================================
# Generate plain-language explanation for one feature
# ============================================================

def create_feature_explanation(
    feature,
    display_name,
    value,
    risk_direction
):

    # --------------------------------------------------------
    # Business-friendly phrases
    # --------------------------------------------------------

    if risk_direction == "increased":

        phrase = (
            f"{display_name} contributed to "
            f"a higher estimated credit risk"
        )

    elif risk_direction == "decreased":

        phrase = (
            f"{display_name} contributed to "
            f"a lower estimated credit risk"
        )

    else:

        phrase = (
            f"{display_name} had little influence "
            f"on the assessment"
        )

    # --------------------------------------------------------
    # Add applicant value where useful
    # --------------------------------------------------------

    if value is not None:

        phrase += f" (value: {value})"

    return phrase + "."


# ============================================================
# SHAP explanation
# ============================================================

def get_shap_explanation(applicant):

    # --------------------------------------------------------
    # Transform applicant exactly as model sees it
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
    # Handle different SHAP return formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        # For some binary classifiers SHAP may return
        # one array per class.
        #
        # Target encoding:
        # good = 1
        # bad  = 0
        #
        # We want the explanation for class 1 ("good")
        # because XGBoost binary output is generally
        # represented in terms of the positive class.

        if len(shap_values) > 1:

            shap_values = shap_values[1]

        else:

            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    if shap_values.ndim == 2:

        shap_values = shap_values[0]

    elif shap_values.ndim == 3:

        shap_values = shap_values[0, :, 0]

    # --------------------------------------------------------
    # Feature names after preprocessing
    # --------------------------------------------------------

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(
        transformed_feature_names
    ) != len(
        shap_values
    ):

        raise ValueError(
            "SHAP feature count does not match "
            "preprocessed feature count."
        )

    # --------------------------------------------------------
    # Build transformed SHAP table
    # --------------------------------------------------------

    shap_df = pd.DataFrame({
        "transformed_feature":
            transformed_feature_names,

        "shap_value":
            shap_values
    })

    # --------------------------------------------------------
    # Map one-hot features back to original columns
    # --------------------------------------------------------

    shap_df["original_feature"] = (
        shap_df[
            "transformed_feature"
        ].apply(
            get_original_feature_name
        )
    )

    # --------------------------------------------------------
    # Aggregate one-hot encoded categories
    #
    # Example:
    #
    # Checking_Status_<0
    # Checking_Status_no checking
    # Checking_Status_0<=X<200
    #
    # are combined back into:
    #
    # Checking_Status
    # --------------------------------------------------------

    grouped = (
        shap_df
        .groupby(
            "original_feature",
            as_index=False
        )
        .agg({
            "shap_value": "sum"
        })
    )

    grouped[
        "absolute_shap"
    ] = grouped[
        "shap_value"
    ].abs()

    grouped = grouped.sort_values(
        "absolute_shap",
        ascending=False
    )

    # ========================================================
    # IMPORTANT SHAP INTERPRETATION
    #
    # Target:
    #
    # good = 1
    # bad  = 0
    #
    # Positive SHAP:
    # pushes prediction toward class 1 = GOOD
    # therefore DECREASES bad-credit risk.
    #
    # Negative SHAP:
    # pushes prediction away from class 1
    # toward BAD credit.
    # therefore INCREASES estimated risk.
    # ========================================================

    explanations = []

    for _, row in grouped.iterrows():

        feature = row[
            "original_feature"
        ]

        shap_value = float(
            row["shap_value"]
        )

        impact = abs(
            shap_value
        )

        # Ignore effectively zero contributions
        if impact < 0.0001:
            continue

        if shap_value < 0:

            risk_direction = "increased"

        elif shap_value > 0:

            risk_direction = "decreased"

        else:

            risk_direction = "neutral"

        display_name = FEATURE_LABELS.get(
            feature,
            feature.replace(
                "_",
                " "
            ).title()
        )

        value = get_display_value(
            applicant,
            feature
        )

        explanation = (
            create_feature_explanation(
                feature=feature,
                display_name=display_name,
                value=value,
                risk_direction=risk_direction
            )
        )

        explanations.append({
            "feature":
                feature,

            "display_name":
                display_name,

            "value":
                value,

            "impact":
                round(
                    impact,
                    4
                ),

            "risk_direction":
                risk_direction,

            "explanation":
                explanation
        })

    # --------------------------------------------------------
    # Top overall factors
    # --------------------------------------------------------

    top_factors = explanations[:5]

    # --------------------------------------------------------
    # Factors increasing bad-credit risk
    # --------------------------------------------------------

    risk_increasing = [
        item
        for item in explanations
        if item[
            "risk_direction"
        ] == "increased"
    ][:5]

    # --------------------------------------------------------
    # Factors reducing bad-credit risk
    # --------------------------------------------------------

    risk_reducing = [
        item
        for item in explanations
        if item[
            "risk_direction"
        ] == "decreased"
    ][:5]

    return {
        "top_factors":
            top_factors,

        "risk_increasing_factors":
            risk_increasing,

        "risk_reducing_factors":
            risk_reducing
    }


# ============================================================
# Create assessment summary
# ============================================================

def create_assessment_summary(
    risk_score,
    risk_level,
    decision,
    shap_result
):

    increasing = shap_result[
        "risk_increasing_factors"
    ]

    reducing = shap_result[
        "risk_reducing_factors"
    ]

    # --------------------------------------------------------
    # Opening sentence
    # --------------------------------------------------------

    summary = (
        f"The applicant received a risk score of "
        f"{risk_score:.2f}, corresponding to a "
        f"{risk_level.lower()} risk classification "
        f"and a recommended decision of "
        f"{decision}."
    )

    # --------------------------------------------------------
    # Main risk drivers
    # --------------------------------------------------------

    if increasing:

        names = [
            item["display_name"].lower()
            for item in increasing[:3]
        ]

        if len(names) == 1:

            risk_text = names[0]

        elif len(names) == 2:

            risk_text = (
                f"{names[0]} and "
                f"{names[1]}"
            )

        else:

            risk_text = (
                f"{names[0]}, "
                f"{names[1]}, and "
                f"{names[2]}"
            )

        summary += (
            " The main factors contributing "
            "to higher estimated risk were "
            f"{risk_text}."
        )

    # --------------------------------------------------------
    # Protective factors
    # --------------------------------------------------------

    if reducing:

        names = [
            item["display_name"].lower()
            for item in reducing[:2]
        ]

        if len(names) == 1:

            protective_text = names[0]

        else:

            protective_text = (
                f"{names[0]} and "
                f"{names[1]}"
            )

        summary += (
            " Factors contributing to lower "
            "estimated risk included "
            f"{protective_text}."
        )

    return summary


# ============================================================
# Debug endpoint
# ============================================================

@app.route(
    "/debug",
    methods=["POST"]
)
def debug():

    data = request.get_json(
        silent=True
    )

    return jsonify({
        "success": True,

        "received_data":
            data,

        "received_fields":
            (
                list(
                    data.keys()
                )
                if isinstance(
                    data,
                    dict
                )
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
        print(
            request.content_type
        )

        print("-" * 60)

        print("Received data:")
        print(
            data
        )

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
        # Check object type
        # ----------------------------------------------------

        if not isinstance(
            data,
            dict
        ):

            return jsonify({
                "error":
                    "JSON payload must be an object.",

                "received_type":
                    type(
                        data
                    ).__name__
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
                    list(
                        data.keys()
                    ),

                "received_data":
                    data
            }), 400

        # ----------------------------------------------------
        # Check numeric fields
        # ----------------------------------------------------

        invalid_numeric_fields = []

        for field in NUMERIC_FIELDS:

            try:

                float(
                    data[field]
                )

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
                        field:
                            data.get(
                                field
                            )
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

        probabilities = (
            pipeline.predict_proba(
                applicant
            )[0]
        )

        # ----------------------------------------------------
        # Probability values
        #
        # Target encoding:
        #
        # bad  = 0
        # good = 1
        # ----------------------------------------------------

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
        # Decision rules
        #
        # These are demonstration thresholds,
        # not real-world underwriting standards.
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

        shap_result = (
            get_shap_explanation(
                applicant
            )
        )

        # ----------------------------------------------------
        # Plain-language summary
        # ----------------------------------------------------

        assessment_summary = (
            create_assessment_summary(
                risk_score=
                    risk_score,

                risk_level=
                    risk_level,

                decision=
                    decision,

                shap_result=
                    shap_result
            )
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = {

            "prediction":
                int(
                    prediction
                ),

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

            "assessment_summary":
                assessment_summary,

            "top_risk_factors":
                shap_result[
                    "top_factors"
                ],

            "risk_increasing_factors":
                shap_result[
                    "risk_increasing_factors"
                ],

            "risk_reducing_factors":
                shap_result[
                    "risk_reducing_factors"
                ]
        }

        # ----------------------------------------------------
        # Log result
        # ----------------------------------------------------

        print("=" * 60)
        print("CREDIT ASSESSMENT RESULT")
        print("=" * 60)

        print(
            response
        )

        print("=" * 60)

        return jsonify(
            response
        )

    except Exception as e:

        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            str(
                e
            )
        )

        print("=" * 60)

        return jsonify({
            "error":
                str(
                    e
                ),

            "error_type":
                type(
                    e
                ).__name__
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
