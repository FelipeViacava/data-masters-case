import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def histplot(df,var,nbins=25):
    df[var] \
        .plot \
        .hist(bins=nbins)

    plt.title(f"Distribution of {var}")
    plt.xlabel(var)
    plt.show()

def lineplot(df,var):
    pd.crosstab(df[var], df["TARGET"]) \
        .div(
            pd.crosstab(df[var], df["TARGET"]).sum(axis=1),
            axis=0
        ) \
        [1] \
        .plot \
        .line()

    plt.title(f"Proportion of TARGET=1 by {var}")
    plt.show()

def violinplot(df,var):
    sns.violinplot(
        df,
        x="TARGET",
        y=var
    )

    plt.title(f"Distribution of {var} by TARGET")
    plt.show()