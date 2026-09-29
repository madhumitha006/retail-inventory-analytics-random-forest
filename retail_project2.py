from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "retail_sales_clean.csv")

# The current day's Units_Sold and Closing_Stock are excluded from model inputs.
features = [
    "Store_ID", "Category", "Opening_Stock", "Price",
    "Promotion", "Holiday", "Historical_Avg_Units_Sold"
]
X, y = df[features], df["Fast_Moving"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Store_ID", "Category"]),
    ("num", "passthrough", [
        "Opening_Stock", "Price", "Promotion", "Holiday",
        "Historical_Avg_Units_Sold"
    ])
])

model = Pipeline([
    ("preprocess", preprocess),
    ("rf", RandomForestClassifier(
        n_estimators=300, random_state=42,
        class_weight="balanced", min_samples_leaf=2
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Project 2 files are ready.")
print(f"Random Forest Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, pred, zero_division=0):.3f}")
print(f"F1: {f1_score(y_test, pred, zero_division=0):.3f}")
print("Use data/retail_sales_clean.csv and results/random_forest_predictions.csv for analysis.")
