import pandas as pd
from resources.prep import build_prep_nan
from resources.train_evaluate import build_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

rf = Pipeline(
    steps=[
        ("preprocessor", build_prep_nan()),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_estimators=500,
                class_weight="balanced_subsample"
            )
        )
    ]
)

rf_grid = {
    "classifier__max_depth": [4, 8, 16, 32],
    "classifier__max_features": [8, 16, 32, 64]
}

rf_model = build_model(
    train = True,
    path = "models/rf.pkl",
    train_df = train,
    model = rf,
    param_grid = rf_grid,
    target = "TARGET",
    njobs = 6,
    verbose = True
)