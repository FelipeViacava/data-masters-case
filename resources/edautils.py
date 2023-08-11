import pandas as pd

def neg_pos_zero(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Returns a dataframe with the percentage of negative, positive and zero values in the given columns.
    :param df: Dataframe to analyze.
    :param cols: Columns to analyze.
    :return: Dataframe with the percentage of negative, positive and zero values in the given columns.
    """
    return df[cols] \
        .applymap(lambda x: 100 / len(df) if x < 0 else 0) \
        .sum() \
        .reset_index() \
        .rename(
            mapper={
                "index": "Column",
                0: "Negative values (%)"
            },
            axis=1
        ) \
        .merge(
            df[cols] \
                .applymap(lambda x: 100 / len(df) if x > 0 else 0) \
                .sum() \
                .reset_index() \
                .rename(
                    mapper={
                        "index": "Column",
                        0: "Positive values (%)"
                    },
                    axis=1
                ),
            on="Column",
            how="outer"
        ) \
        .merge(
            df[cols] \
                .applymap(lambda x: 100 / len(df) if x == 0 else 0) \
                .sum() \
                .reset_index() \
                .rename(
                    mapper={
                        "index": "Column",
                        0: "Zero values (%)"
                    },
                    axis=1
                ),
            on="Column",
            how="outer"
        ) \
        .sort_values("Negative values (%)")