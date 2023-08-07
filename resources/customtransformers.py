from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
import pandas as pd
import numpy as np

class DropConstantColumns(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    It drops constant columns from a pandas dataframe object.
    Important: the constant columns are found in the fit function and dropped in the transform function.
    """
    def __init__(self, print_cols: bool = False, also: list[str] = []) -> None:
        """
        print_cols: default = False. Determine whether the fit function should print the constant columns' names.
        ignore: list of columns to ignore.
        Initiates the class.
        """
        self.print_cols = print_cols
        self.also = also
        pass

    def fit(self, X: pd.DataFrame , y: None = None) -> None:
        """
        X: dataset whose constant columns should be removed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be removed in the transform function.
        """
        self.constant_cols = [
            col
            for col in X.columns
            if (
                (X[col].nunique() == 1)
                & (col in self.also)
            )
        ]
        if self.print_cols:
            print(f"{len(self.constant_cols)} constant columns were found")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset whose constant columns should be removed.
        Returns dataset without the constant columns found in the fit function.
        """
        return X.copy().drop(self.constant_cols, axis=1)

class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    It drops duplicate columns from a pandas dataframe object.
    Important: the duplicate columns are found in the fit function and dropped in the transform function.
    """
    def __init__(self, print_cols: bool = False, ignore: list[str] = []) -> None:
        """
        print_cols: default = False. Determine whether the fit function should print the duplicate columns' names.
        ignore: list of columns to ignore.
        Initiates the class.
        """
        self.print_cols = print_cols
        self.ignore = ignore
        pass

    def fit(self, X: pd.DataFrame, y: None = None) -> None:
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset whose duplicate columns should be removed.
        Returns dataset without the duplicate columns found in the fit function.
        """ 
        X_ = X.copy()
        return X_.drop(self.duplicate_cols, axis=1)

class AddNonZeroCount(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    """
    def __init__(self, prefix: str = "", ignore: list[str] = []) -> None:
        """
        prefix: prefix of the columns to be summed.
        ignore: list of columns to ignore.
        fake_value: value to be replaced with None.
        Initiates de class.
        """
        self.prefix = prefix
        self.ignore = ignore
        pass

    def fit(self, X: pd.DataFrame, y: None = None) -> None:
        """
        X: dataset whose "prefix" variables different than 0 should be counted.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns whose not 0 values should be counted in the transform function.
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset whose "prefix" variables' not 0 values should be counted.
        Returns dataset with new column with the count of the "prefix" variables' not 0 values.
        """  
        X_ = X.copy()
        X_[f"non_zero_count_{self.prefix}"] = X_[self.prefix_cols] \
            .applymap(lambda x: 0 if ((x == 0) | (x == None)) else 1) \
            .sum(axis=1)
        return X_

class CustomSum(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    It sums columns from a pandas dataframe object based on the columns prefix.
    """
    def __init__(self, prefix: str = "", ignore: list[str] = []) -> None:
        """
        prefix: prefix of the columns to be summed.
        ignore: list of columns to ignore.
        fake_value: value to be replaced with None.
        Initiates de class.
        """
        self.prefix = prefix
        self.ignore = ignore
        pass

    def fit(self, X: pd.DataFrame, y: None = None) -> None:
        """
        X: dataset whose columns with "prefix" should be summed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be summed in the transform function.
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset whose "prefix" variables should be summed.
        Returns dataset with new column with the sum of the "prefix" variables.
        """  
        X_ = X.copy()
        X_[f"sum_of_{self.prefix}"] = X_[self.prefix_cols] \
            .sum(axis=1)
        return X_

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in a sklearn.pipeline.Pipeline object.
    It imputes values in a pandas dataframe object based on the columns prefix.
    """
    def __init__(self, prefix: str, to_replace: Union[int, float, str],
                 replace_with: Union[int, float, str] = np.nan, ignore: list[str] = []) -> None:
        """
        prefix: prefix of the columns to be imputed.
        to_replace: value to be replaced.
        replace_with: value to replace "to_replace" with.
        ignore: list of columns to ignore.
        Initiates de class.
        """
        self.prefix = prefix
        self.to_replace = to_replace
        self.replace_with = replace_with
        self.ignore = ignore
        pass

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: None = None) -> None:
        """
        X: dataset whose columns with "prefix" should be imputed.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns to be imputed in the transform function.
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
    
    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        X: dataset whose columns with "prefix" should be imputed.
        Returns dataset with the imputed columns.
        """
        X_ = X.copy()
        X_[self.prefix_cols] = X_[self.prefix_cols] \
            .replace(self.to_replace, self.replace_with)
        return X_
 
class AddNoneCount(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    It counts the number of None values in a pandas dataframe object based on the columns prefix.
    """
    def __init__(self, prefix: str = "", ignore: list[str] = []) -> None:
        """
        prefix: subset of variables for none count starting with this string.
        fake_value: values inserted to replace None.
        ignore: list of columns with prefix to ignore.
        drop_constant: whether to drop columns that would become constant without missing features or not.
        """
        self.prefix = prefix
        self.ignore = ignore
        pass

    def fit(self, X: pd.DataFrame, y: None = None) -> None:
        """
        X: dataset whose "prefix" variables' null values should be counted.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the names of the columns whose null values should be counted in the transform function.
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset to apply transformation on.
        Returns dataset with new column with the count of the "prefix" variables' null values.
        """  
        X_ = X.copy()
        X_[f"none_count_{self.prefix}"] = X_[self.prefix_cols] \
            .isnull() \
            .sum(axis=1)
        return X_
    
class CustomEncoder(BaseEstimator, TransformerMixin):
    """
    This class is made to work as a step in sklearn.pipeline.Pipeline object.
    It encodes categorical variables in a pandas dataframe based on the categories mean of the target variable.
    Unknown values must be defined by the user.
    """
    def __init__(self, colname: str) -> None:
        """
        labels: dictionary with the labels to be replaced.
        colname: name of the column to be encoded.
        Initiates de class.
        """
        self.colname = colname
        pass

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        X: dataset whose column should be encoded.
        y: Shouldn't be used. Only exists to prevent raise Exception due to accidental input in a pipeline.
        Creates class atributte with the dictionary to be used in the transform function.
        """
        X_ = X.copy().assign(TARGET=y)

        grouped_X_ = X_ \
            .groupby(self.colname) \
            .agg({"TARGET": "mean"}) \
            .sort_values("TARGET", ascending=True)
        
        groups = grouped_X_.index

        self.labels ={
            groups[i]: i
            for i in range(len(groups))
        }

        self.most_frequent = X_[self.colname].mode()[0]
        return self
    
    def _apply_map(self, x: Union[int, str]) -> int:
        """
        x: value to be replaced.
        Returns the value to replace "x" with.
        """
        if x in self.labels.keys():
            return self.labels[x]
        else:
            return self.labels[self.most_frequent]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: dataset whose column should be encoded.
        Returns dataset with the encoded column.
        """
        X_ = X.copy()
        X_[self.colname] = X_[self.colname] \
            .apply(self._apply_map)
        return X_