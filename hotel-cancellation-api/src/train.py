import json
import logging
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import (
    EXCLUDED_LEAKY_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_pipeline,
    get_feature_columns,
)

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "hotels.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"


def download_data() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        logger.info("Data already exists at %s", DATA_PATH)
        return
    logger.info("Downloading dataset from %s", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    size_mb = DATA_PATH.stat().st_size / 1_048_576
    logger.info("Downloaded %.2f MB to %s", size_mb, DATA_PATH)


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    before = len(df)
    df = df[(df["adr"] >= 0) & (df["adr"] <= 5000)]
    logger.info("Dropped %d adr outliers", before - len(df))

    cols_to_drop = [c for c in EXCLUDED_LEAKY_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped leaky columns: %s", cols_to_drop)

    required_cols = FEATURE_COLUMNS + ["is_canceled"]
    df = df[required_cols].dropna()
    logger.info("After column selection and dropna: %d rows remain", len(df))

    X = df[FEATURE_COLUMNS]
    y = df["is_canceled"]

    counts = y.value_counts()
    rate = 100 * counts.get(1, 0) / len(y)
    logger.info(
        "Class distribution — cancelled: %d (%.1f%%), not cancelled: %d (%.1f%%)",
        counts.get(1, 0),
        rate,
        counts.get(0, 0),
        100 - rate,
    )

    return X, y


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info("ROC-AUC : %.4f", roc_auc)
    logger.info("F1 weighted: %.4f", f1)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["kept", "cancelled"]))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    return {"roc_auc": round(roc_auc, 4), "f1_weighted": round(f1, 4)}


def save_artifacts(pipeline, metrics: dict, n_train: int, n_test: int) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    logger.info("Model saved to %s", MODEL_PATH)

    metadata = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_estimators": 300,
        "train_samples": n_train,
        "test_samples": n_test,
        "test_roc_auc": metrics["roc_auc"],
        "test_f1_weighted": metrics["f1_weighted"],
        "feature_columns": get_feature_columns(),
        "excluded_leaky_columns": EXCLUDED_LEAKY_COLUMNS,
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata saved to %s", METADATA_PATH)


def main() -> None:
    download_data()

    X, y = load_and_clean(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    logger.info("Training pipeline...")
    pipeline = build_feature_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    metrics = evaluate(pipeline, X_test, y_test)
    save_artifacts(pipeline, metrics, len(X_train), len(X_test))
    logger.info("Done. Run 'make run-dev' to start the API.")


if __name__ == "__main__":
    main()