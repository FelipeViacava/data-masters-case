import pandas as pd
from resources.prep import build_prep_2
from resources.train_evaluate import build_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("Training RandomForestClassifier...")

train = pd.read_csv("data/train.csv")

rf = Pipeline(
    steps=[
        ("preprocessor", build_prep_2()),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_estimators=500,
                class_weight="balanced"
            )
        )
    ]
)

rf_grid = {
    "classifier__max_depth": [4, 8, 16, 32],
    "classifier__max_features": [8, 16, 32, 64],
}

rf_model = build_model(
    path = "models/rf.pkl",
    train_df = train,
    model = rf,
    param_grid = rf_grid,
    target = "TARGET",
    njobs = 8,
    verbose = True
)

print("RandomForestClassifier trained. Model saved to models/rf.pkl")