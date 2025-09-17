# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# import optuna

# print("=== Phase 2: Modeling with XGBoost ===")

# df = pd.read_csv("data/processed/telco_churn_processed.csv")

# # target must be numeric 0/1
# if df["Churn"].dtype == "object":
#     df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

# assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
# assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

# X = df.drop(columns=["Churn"])
# y = df["Churn"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# THRESHOLD = 0.4

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 800),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
#         "random_state": 42,
#         "n_jobs": -1,
#         "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
#         "eval_metric": "logloss",
#     }
#     model = XGBClassifier(**params)
#     model.fit(X_train, y_train)
#     proba = model.predict_proba(X_test)[:, 1]
#     y_pred = (proba >= THRESHOLD).astype(int)
#     from sklearn.metrics import recall_score
#     return recall_score(y_test, y_pred, pos_label=1)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)
# print("Best Params:", study.best_params)
# print("Best Recall:", study.best_value)






# phase2_modeling.py
# phase2_modeling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import recall_score

print("=== Phase 2: Modeling with XGBoost ===")

# --- Load processed data ---
df = pd.read_csv("data/processed/telco_churn_processed.csv")

# --- Ensure target is numeric 0/1 ---
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

# --- Split features and target ---
X = df.drop(columns=["Churn"])
y = df["Churn"]

# --- Convert binary columns to numeric ---
binary_mapping = {
    "gender": {"Male": 1, "Female": 0},
    "Partner": {"Yes": 1, "No": 0},
    "Dependents": {"Yes": 1, "No": 0},
    "PhoneService": {"Yes": 1, "No": 0},
    "PaperlessBilling": {"Yes": 1, "No": 0},
}

for col, mapping in binary_mapping.items():
    X[col] = X[col].map(mapping).astype(int)

# --- One-hot encode multi-category columns ---
multi_cat_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
X = pd.get_dummies(X, columns=multi_cat_cols, drop_first=False)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

# --- Optuna objective function ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return recall_score(y_test, y_pred, pos_label=1)

# --- Run Optuna study ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("✅ Best Params:", study.best_params)
print("✅ Best Recall:", study.best_value)
