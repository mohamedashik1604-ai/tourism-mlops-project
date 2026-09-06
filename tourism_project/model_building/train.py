"""
Model training script for the Tourism Wellness Package project.
Trains and tunes three candidate algorithms (XGBoost, Random Forest,
Gradient Boosting), logs each to MLflow, compares them on recall
(prioritized due to class imbalance), and saves the best-performing
model for deployment.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import mlflow

TRAIN_X_PATH = "Xtrain.csv"
TEST_X_PATH = "Xtest.csv"
TRAIN_Y_PATH = "ytrain.csv"
TEST_Y_PATH = "ytest.csv"

MODEL_OUTPUT_PATH = "tourism_project/deployment/model.joblib"

NUMERIC_COLS = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_data():
    X_train = pd.read_csv(TRAIN_X_PATH)
    X_test = pd.read_csv(TEST_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH).squeeze()
    y_test = pd.read_csv(TEST_Y_PATH).squeeze()
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    return make_column_transformer(
        (StandardScaler(), NUMERIC_COLS),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    )


def get_candidates():
    """Returns a dict of {name: (estimator, param_grid)} for GridSearchCV."""
    return {
        "XGBoost": (
            xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=4, random_state=42),
            {
                "xgbclassifier__n_estimators": [100, 200],
                "xgbclassifier__max_depth": [3, 5],
                "xgbclassifier__learning_rate": [0.05, 0.1],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "randomforestclassifier__n_estimators": [100, 200],
                "randomforestclassifier__max_depth": [5, 10, None],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "gradientboostingclassifier__n_estimators": [100, 200],
                "gradientboostingclassifier__max_depth": [3, 5],
                "gradientboostingclassifier__learning_rate": [0.05, 0.1],
            },
        ),
    }


def train_and_track():
    X_train, X_test, y_train, y_test = load_data()
    candidates = get_candidates()

    mlflow.set_experiment("tourism_wellness_package")

    results = {}
    fitted_models = {}

    for name, (estimator, param_grid) in candidates.items():
        print(f"\nTuning {name}...")
        pipeline = make_pipeline(build_preprocessor(), estimator)

        with mlflow.start_run(run_name=name):
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="recall",
                cv=3,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            fitted_models[name] = best_model

            # Log every tuned hyperparameter
            mlflow.log_param("model_type", name)
            for param_name, param_value in grid_search.best_params_.items():
                mlflow.log_param(param_name, param_value)

            y_pred = best_model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            results[name] = metrics
            print(f"{name} best params: {grid_search.best_params_}")
            print(f"{name} test recall: {metrics['recall']:.4f}")

    # --- Compare and select the best model by recall ---
    comparison_df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(comparison_df)
    print("=" * 60)

    best_name = comparison_df["recall"].idxmax()
    best_model = fitted_models[best_name]
    print(f"\nSelected model: {best_name} (highest test recall)")

    y_pred_best = best_model.predict(X_test)
    print("=" * 60)
    print(f"CLASSIFICATION REPORT — {best_name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_best))
    print("=" * 60)

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Best model ({best_name}) saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_track()
