from pathlib import Path
import joblib
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def test_artifacts_exist():
    required = [
        "lr_model.joblib",
        "svm_model.joblib",
        "nb_model.joblib",
        "shadow_model.joblib",
        "shadow_metadata.joblib",
    ]
    for name in required:
        assert (ARTIFACTS_DIR / name).exists(), f"Missing artifact: {name}"


def test_artifacts_load():
    lr_model = joblib.load(ARTIFACTS_DIR / "lr_model.joblib")
    svm_model = joblib.load(ARTIFACTS_DIR / "svm_model.joblib")
    nb_model = joblib.load(ARTIFACTS_DIR / "nb_model.joblib")
    shadow_model = joblib.load(ARTIFACTS_DIR / "shadow_model.joblib")
    shadow_metadata = joblib.load(ARTIFACTS_DIR / "shadow_metadata.joblib")

    assert lr_model is not None
    assert svm_model is not None
    assert nb_model is not None
    assert shadow_model is not None
    assert shadow_metadata is not None


def test_main_models_output_shapes():
    lr_model = joblib.load(ARTIFACTS_DIR / "lr_model.joblib")
    svm_model = joblib.load(ARTIFACTS_DIR / "svm_model.joblib")
    nb_model = joblib.load(ARTIFACTS_DIR / "nb_model.joblib")

    sample = pd.Series([
        "I forgot my password and cannot log into my patient portal."
    ])

    lr_proba = lr_model.predict_proba(sample)
    svm_proba = svm_model.predict_proba(sample)
    nb_proba = nb_model.predict_proba(sample)

    assert lr_proba.shape[0] == 1
    assert svm_proba.shape[0] == 1
    assert nb_proba.shape[0] == 1

    assert lr_proba.shape[1] == 6
    assert svm_proba.shape[1] == 6
    assert nb_proba.shape[1] == 6


def test_probabilities_sum_to_one_for_main_model():
    lr_model = joblib.load(ARTIFACTS_DIR / "lr_model.joblib")

    sample = pd.Series([
        "I need help getting a refill for my prescription."
    ])

    probs = lr_model.predict_proba(sample)[0]
    assert abs(probs.sum() - 1.0) < 1e-6


def test_shadow_metadata_has_expected_fields():
    shadow_metadata = joblib.load(ARTIFACTS_DIR / "shadow_metadata.joblib")

    assert "feature_names" in shadow_metadata
    assert "risk_threshold_default" in shadow_metadata
    assert "classes" in shadow_metadata

    assert isinstance(shadow_metadata["feature_names"], list)