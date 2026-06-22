
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1) Load dataset
# -----------------------------
df = pd.read_csv("cars_dataset.csv")

# -----------------------------
# 2) Features and target
# -----------------------------
X = df[["brand", "model", "year", "km_driven"]]
y = df["price"]

# -----------------------------
# 3) Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4) Preprocessing
# -----------------------------
categorical_features = ["brand", "model"]
numerical_features = ["year", "km_driven"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# -----------------------------
# 5) Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

best_model_name = None
best_model = None
best_score = float("-inf")

print("Training models...\n")

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model: {name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print("-" * 40)

    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_model = pipeline

# -----------------------------
# 6) Save best model
# -----------------------------
joblib.dump(best_model, "car_price_model.pkl")

print(f"\nBest model: {best_model_name}")
print(f"Best R2 Score: {best_score:.4f}")
print("Saved as car_price_model.pkl")
