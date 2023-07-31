import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.optimize import minimize_scalar
import pandas as pd
import numpy as np

class CustomModel:
    def __init__(self,
                 dataframe,
                 pipeline,
                 target_column,
                 param_grid=None,
                 test_size=0.2,
                 use_grid_search=True):
        self.data = dataframe
        self.pipeline = pipeline
        self.target_column = target_column
        self.param_grid = param_grid
        self.test_size = test_size
        self.use_grid_search = use_grid_search

    def _split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=42
        )

    def _custom_metric(self, threshold):
        predictions = (self.y_val_proba[:, 1] > threshold).astype(int)
        tp = np.sum((predictions == 1) & (self.y_val == 1))
        fp = np.sum((predictions == 1) & (self.y_val == 0))
        n = len(self.y_val)
        return (-10 * fp + 90 * tp) / n

    def _optimize_threshold(self):
        result = minimize_scalar(self._custom_metric, bounds=(0, 1), method='bounded')
        self.optimal_threshold = result.x

    def fit(self):
        # Split the data
        self._split_data()

        if self.use_grid_search:
            # Perform GridSearchCV to find the best parameters that maximize AUC
            grid_search = GridSearchCV(self.pipeline, self.param_grid, scoring='roc_auc', cv=5)
            grid_search.fit(self.X_train, self.y_train)
            # Use the best estimator found by GridSearchCV
            self.pipeline.set_params(**grid_search.best_params_)

        # Fit the model with the training set
        self.pipeline.fit(self.X_train, self.y_train)

        # Predict the probabilities on the validation set
        self.y_val_proba = self.pipeline.predict_proba(self.X_val)

        # Optimize the threshold using the validation set
        self._optimize_threshold()

        # Concatenate training and validation sets
        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])

        # Refit the model with the entire dataset
        self.pipeline.fit(X_full, y_full)

    def predict(self, X):
        probabilities = self.pipeline.predict_proba(X)
        return (probabilities[:, 1] > self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_optimal_threshold(self):
        return self.optimal_threshold