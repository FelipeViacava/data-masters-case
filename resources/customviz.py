import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def binaryhistplot(x: pd.Series, hue: pd.Series, nbins=25) -> None:
    sns.histplot(
        x=x,
        hue=hue,
        element="step",
        stat="density",
        common_norm=False,
        common_bins=True,
        bins=nbins
    )
    plt.title(f"Distribution of {x.name} var by TARGET")
    plt.show()