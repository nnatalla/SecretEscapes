from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERICAL_FEATURES = [
    "lead_time",
    "adr",
    "total_of_special_requests",
    "booking_changes",
    "previous_cancellations",
    "days_in_waiting_list",
    "required_car_parking_spaces",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_LEAKY_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
]


def get_feature_columns():
    return {
        "numerical": NUMERICAL_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    }


def build_feature_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    max_features="sqrt",
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )