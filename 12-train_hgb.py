import pandas as pd
from resources.prep import build_prep
from resources.train_evaluate import build_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

hgbc = Pipeline(
    steps=[
        ("preprocessor", build_prep()),
        (
            "classifier",
            HGBC(
                random_state=42,
                class_weight="balanced",
                max_leaf_nodes=None,
                categorical_features=["var36", "var21"]
            )
        )
    ]
)

hgbc_grid = {
    "classifier__learning_rate": [.001, .01, .1, 1],
    "classifier__max_iter": [50, 75, 100, 150],
    "classifier__max_depth": [2, 4, 6, 8, 10]
}

hgbc_model = build_model(
    train = True,
    path = "models/hgbc.pkl",
    train_df = train,
    model = hgbc,
    param_grid = hgbc_grid,
    target = "TARGET",
    njobs = 6,
    verbose = True
)