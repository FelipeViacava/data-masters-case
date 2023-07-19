from sklearn.base import BaseEstimator, TransformerMixin

class DropConstantColumns(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.compose.ColumnTransformer object.
    """
    def __init__(self, print_cols=False):
        """
        print_cols: default = False. Determine whether the fit function should print the constant columns' names.
        """
        self.print_cols = print_cols
        pass

    def fit(self, X, y=None):
        """
        X: dataset whose constant columns should be removed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        self.constant_cols = [
            col
            for col in X.columns
            if X[col].nunique() == 1
        ]
        if self.print_cols:
            print(f"{len(self.constant_cols)} constant columns were found")
        return self
    
    def transform(self, X):
        """
        X: dataset whose constant columns should be removed.
        Returns dataset without the constant columns found in the fit function.
        """  
        X_ = X.copy()
        return X_.drop(self.constant_cols, axis=1)

class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.compose.ColumnTransformer object.
    """
    def __init__(self, print_cols=False):
        """
        print_cols: default = False. Determine whether the fit function should print the duplicate columns' names.
        """
        self.print_cols = print_cols
        pass

    def fit(self, X, y=None):
        """
        X: dataset whose duplicate columns should be removed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        regular_columns = []
        duplicate_columns = []
        for col0 in X.columns:
            if col0 not in duplicate_columns:
                regular_columns.append(col0)
            for col1 in X.columns:
                if (col0 != col1):
                    if X[col0].equals(X[col1]):
                        if col1 not in regular_columns:
                            duplicate_columns.append(col1)
        self.duplicate_cols = duplicate_columns
        if self.print_cols:
            print(f"{len(duplicate_columns)} duplicate columns were found.")
        return self
    
    def transform(self, X):
        """
        X: dataset whose duplicate columns should be removed.
        Returns dataset without the duplicate columns found in the fit function.
        """ 
        X_ = X.copy()
        return X_.drop(self.duplicate_cols,axis=1)