from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    This class should be used as a step in a sklearn pipeline
    """
    def __init__(self, prefix, value):
        self.prefix = prefix
        self.value = value
        pass

    def fit(self, X, y=None):
        self.prefix_cols = [
            col
            for col in X.columns
            if (
                    (col.startswith(self.prefix))
                    & (col not in self.ignore)
            )
        ]
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[self.prefix_cols] = X_[self.prefix_cols] \
            .applymap(lambda x: self.value if x == None else x)
        return X_

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
        sorted_cols = sorted(X.columns)
        for col0 in sorted_cols:
            if col0 not in duplicate_columns:
                regular_columns.append(col0)
            for col1 in sorted_cols:
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
        return X_.drop(self.duplicate_cols, axis=1)

class CustomSum(BaseEstimator, TransformerMixin):
    def __init__(self, prefix="", ignore=[], fake_value=None):
        """
        prefix: subset of variables to sum.
        ignore: list of columns to ignore
        """
        self.prefix = prefix
        self.ignore = ignore
        self.fake_value = fake_value
        pass

    def fit(self, X, y=None):
        """
        X: dataset whose "prefix" variables should be summed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        self.prefix_cols = [
            col
            for col in X.columns
            if (
                    (col.startswith(self.prefix))
                    & (col not in self.ignore)
            )
        ]
        return self
    
    def transform(self, X):
        """
        X: dataset whose "prefix" variables should be summed.
        Returns dataset without the constant columns found in the fit function.
        """  
        X_ = X.copy()
        X_[f"sum_of_{self.prefix}"] = X_[self.prefix_cols] \
            .applymap(lambda x: None if x == self.fake_value else x) \
            .sum(axis=1)
        return X_

class AddNonZeroCount(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.compose.ColumnTransformer object.
    """
    def __init__(self, prefix="", ignore=[], fake_value=None):
        """
        prefix: subset of variables for non-zero count.
        ignore: list of columns to ignore
        """
        self.prefix = prefix
        self.ignore = ignore
        self.fake_value = fake_value
        pass

    def fit(self, X, y=None):
        """
        X: dataset whose "prefix" variables different than 0 should be counted.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        self.prefix_cols = [
            col
            for col in X.columns
            if (
                    (col.startswith(self.prefix))
                    & (col not in self.ignore)
            )
        ]
        return self
    
    def transform(self, X):
        """
        X: dataset whose "prefix" variables different than 0 should be counted.
        Returns dataset without the constant columns found in the fit function.
        """  
        X_ = X.copy()
        X_[f"non_zero_count_{self.prefix}"] = X_[self.prefix_cols] \
            .applymap(lambda x:
                1
                if (
                    (x != 0)
                    & (x != self.fake_value)
                )
                else 0
            ) \
            .sum(axis=1)
        return X_
    
class AddNoneCount(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.compose.ColumnTransformer object.
    """
    def __init__(
            self,
            prefix="",
            fake_value=None,
            ignore=[],
            drop_constant=False,
        ):
        """
        prefix: subset of variables for none count starting with this string.
        fake_value: values inserted to replace None.
        ignore: list of columns with prefix to ignore.
        replace_none: whether columns with
        """
        self.prefix = prefix
        self.fake_value = fake_value
        self.ignore = ignore
        self.drop_constant = drop_constant
        pass

    def fit(self, X, y=None):
        """
        X: dataset whose "prefix" variables' null values should be counted.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        self.prefix_cols = [
            col
            for col in X.columns
            if (
                    (col.startswith(self.prefix))
                    & (col not in self.ignore)
            )
        ]
        if self.drop_constant:
            X_ = X.copy()
            X_[self.prefix_cols] = X_[self.prefix_cols] \
                .applymap(lambda x: None if x == self.fake_value else x)
            self.constant_cols = [
                col for col in X_.columns if X_[col].nunique() == 1
            ]
        return self
    
    def transform(self, X):
        """
        X: dataset to apply transformation on.
        Returns dataset without the constant columns found in the fit function.
        """  
        X_ = X.copy()
        X_[f"none_count_{self.prefix}"] = X_[self.prefix_cols] \
            .applymap(lambda x: None if x == self.fake_value else x) \
            .isnull() \
            .sum(axis=1)
        if self.drop_constant:
            X_ = X_.drop(self.constant_cols, axis=1)
        return X_
    
class AvgOverNonZero(BaseEstimator, TransformerMixin):
    def __init__(self, prefix):
        self.sum_col = f"sum_of_{prefix}",
        self.count_col = f"non_zero_count_{prefix}",
        self.prefix = prefix
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[f"avg_{self.prefix}"] = X_[f"sum_of_{self.prefix}"] / (X_[f"non_zero_count_{self.prefix}"]+1)
        return X_
    
class NotZeroFlag(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        pass
    