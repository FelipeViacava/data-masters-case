import pandas as pd
from resources.prep import build_prep
from resources.train_evaluate import build_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

print("Training HistGradientBoostingClassifier...")

df = pd.read_csv("data/train.csv")

hgb = Pipeline(
    steps=[
        ("preprocessor", build_prep()),
        (
            "classifier",
            HGB(
                random_state=42,
                class_weight="balanced",
                categorical_features=["var36", "var21"]
            )
        )
    ]
)

hgb_grid = {
    "classifier__learning_rate": [.01, .03, .1],
    "classifier__max_iter": [100, 150, 200],
    "classifier__max_depth": [3, 4, 5, 6],
    "classifier__l2_regularization": [1, 3, 10]
}

hgbc_model = build_model(
    path = "models/hgb.pkl",
    train_df = df,
    model = hgb,
    param_grid = hgb_grid,
    target = "TARGET",
    njobs = 8,
    verbose = True
)

print("Done training HistGradientBoostingClassifier. Model saved to models/hgb.pkl")