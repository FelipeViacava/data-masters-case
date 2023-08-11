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

def expl_var(evr) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(evr)+1), evr)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.show()

def plot_components(components, feature_names) -> None:
    plt.figure(figsize=(12, 6))
    for i, component in enumerate(components):
        plt.subplot(2, 2, i + 1)
        plt.bar(feature_names, component)
        plt.title(f"Component {i+1}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
