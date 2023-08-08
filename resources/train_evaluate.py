# --- Function Building --- #
from typing import Union

# --- Threshold Optimization --- #
from scipy.optimize import minimize_scalar

# --- Data Manipulation --- #
import numpy as np
import pandas as pd

# --- sklearn utils --- #
from sklearn.pipeline import Pipeline
from sklearn.model_selection import \
    train_test_split, \
    GridSearchCV, \
    StratifiedKFold

# --- sklearn metrics --- #
from sklearn.inspection import permutation_importance
from sklearn.metrics import \
    roc_auc_score, \
    confusion_matrix, \
    accuracy_score, \
    precision_score, \
    recall_score, \
    f1_score

# --- Object serialization --- #
import pickle

class TrainEvaluate:
    """
    This class can be used to train, validate and test sklearn Pipeline objects.
    """
    def __init__(self, model: Pipeline, param_grid: dict, target: str,
                 njobs: int = 8, verbose: bool = True) -> None:
        """
        model: sklearn Pipeline with the model.
        param_grid: Dictionary of parameters to search over.
        target: Name of the column to predict.
        save_model: Wheter to save the model or not.
        save_name: Name of the file to save the model.
        njobs: Number of jobs to run in parallel.
        verbose: Wheter to print the progress or not.
        Initialize the class with the model, param_grid, and target variable.
        """
        self.model = model
        self.param_grid = param_grid
        self.target = target
        self.njobs = njobs
        self.verbose = verbose
        pass

    def _validation_split(self, df: pd.DataFrame) -> tuple:
        """
        df: Pandas DataFrame with the data.
        Split the data into train and validation sets.
        """
        y = df[self.target]
        X = df.drop(self.target, axis=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42
        )
        return (X_train, X_val, y_train, y_val)
    
    def _grid_search(self, X_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series]) -> GridSearchCV:
        """
        X_train: Pandas DataFrame with the training data.
        y_train: Pandas Series with the training target.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring="roc_auc",
            n_jobs=self.njobs,
            cv=skf
        )
        grid_search = grid_search.fit(X_train, y_train)
        return grid_search
    
    def _profit(self, y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
                y_pred: Union[np.ndarray, pd.DataFrame, pd.Series]) -> float:
        """
        y_true: Pandas Series with the true target.
        y_pred: Pandas Series with the predicted target.
        Calculate the profit metric of the model.
        """
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        profit = (90 * tp - 10 * fp)
        return profit
    
    def _threshold_tuning(self, X_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]) -> float:
        """
        X_val: Pandas DataFrame with the validation data.
        y_val: Pandas Series with the validation target.
        Find the threshold that maximizes the profit metric.
        """
        y_proba = self.best_model_.predict_proba(X_val)[:, 1]

        def profit_treshold(x: float) -> float:
            """
            x: Threshold to test.
            Returns negative of the profit metric.
            """
            y_pred = (y_proba >= x).astype(int)
            scalar = -self._profit(y_val, y_pred)
            return scalar
        
        threshold = minimize_scalar(
            profit_treshold,
            bounds=(0, 1),
            method="bounded"
        )
        self.threshold = threshold.x
        return threshold.x
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        df: Pandas DataFrame with the data.
        path: Path to a fitted model.
        Splits data between train and validation, performs GridSearchCV,
        adjusts the threshold based on profit metric on the validation set,
        and fits the model on the original data.
        """
        if self.verbose:
            print("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = self._validation_split(df)
        if self.verbose:
            print("Done!")
            print("Performing GridSearchCV...")
        self.best_model_ = self._grid_search(X_train, y_train).best_estimator_
        if self.verbose:
            print("Done!")
            print("Adjusting threshold based on validation set...")
        self.threshold = self._threshold_tuning(X_val, y_val)
        if self.verbose:
            print("Done!")
            print("Fitting model on the whole dataset...")
        self.best_model_ = self.best_model_.fit(df.drop(self.target, axis=1), df[self.target])
        if self.verbose:
            print("Done!")

        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        df: Pandas DataFrame with the data.
        Predicts the target variable using the best model.
        """
        return self.best_model_.predict_proba(df)[:, 1]
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        df: Pandas DataFrame with the data.
        Predicts the target variable using the best model and the threshold.
        """
        y_proba = self.predict_proba(df)
        y_pred = (y_proba >= self.threshold).astype(int)
        return y_pred
    
    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        df: Pandas DataFrame with the test data.
        Evaluates the model on the data.
        """
        X_test = df.drop(self.target, axis=1)
        y_true = df[self.target]
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        tp, fp, tn, fn = confusion_matrix(y_true, y_pred).ravel()

        self.business_metrics = {
            "Profit (Total)": self._profit(y_true, y_pred),
            "Profit (per Customer)": self._profit(y_true, y_pred) / len(y_true),
            "False Negative Loss (Total)": 90 * fn,
            "False Negative Loss (per Customer)": 90 * fn / len(y_true),
            "False Positive Loss (Total)": 10 * fp,
            "False Positive Loss (per Customer)": 10 * fp / len(y_true),
        }

        self.classification_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_proba),
            "Classification Threshold": self.threshold
        }

        return self

    def _predict_profit(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        X: Pandas DataFrame with the data.
        y: Pandas Series with the target.
        Predicts the profit metric using the best model and custom threshold.
        """
        y_pred = self.predict(X)
        return self._profit(y, y_pred)
    
    def get_feature_importances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: Pandas DataFrame with the data.
        Implements permutation feature importances on the data using the best model and custom threshold.
        """
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        result = permutation_importance(
            self,
            X,
            y,
            scoring=self._predict_profit,
            n_repeats=5,
            random_state=42,
            n_jobs=self.njobs
        )
        feature_importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": result.importances_mean
        })
        self.feature_importances = feature_importances.sort_values("Importance", ascending=False)
        return self.feature_importances
    
    def _apply_rank(self, x: float) -> int:
        """
        x: Probability of insatisfaction.
        Applies the rank (1 to 5) to the probability of insatisfaction.
        """
        thresholds = [c * self.threshold / 4 for c in range(5)][::-1]
        for rank, threshold in enumerate(thresholds):
            if rank >= threshold:
                return rank + 1
        return 5
    
    def rank_customers(self, df: pd.DataFrame) -> pd.Series:
        """
        df: Pandas DataFrame with the data.
        Ranks the customers by their probability of insatisfaction.
        """
        df_ = df.copy()
        X = df_.drop(self.target, axis=1)
        y = df_[self.target]
        df_["rank"] = self.predict_proba(X)
        return df_["rank"].apply(self._apply_rank)
    
def build_model(train: bool = False, path: str = None,
                train_df: pd.DataFrame = None, model: Pipeline = None,
                param_grid: dict = None, target: str = None,
                njobs: int = 8, verbose: bool = True) -> TrainEvaluate:
    """
    train: Wheter to train the model or not.
    path: Path to a fitted model.
    train_df: Pandas DataFrame with the training data.
    test_df: Pandas DataFrame with the test data.
    model: sklearn Pipeline with the model.
    param_grid: Dictionary of parameters to search over.
    target: Name of the column to predict.
    njobs: Number of jobs to run in parallel.
    verbose: Wheter to print the progress or not.
    Builds a TrainEvaluate object.
    """
    if train:
        train_evaluate = TrainEvaluate(model, param_grid, target, njobs, verbose)
        train_evaluate = train_evaluate.fit(train_df)
        with open(path, "wb") as f:
            pickle.dump(train_evaluate, f)
    else:
        with open(path, "rb") as f:
            train_evaluate = pickle.load(f)
    return train_evaluate