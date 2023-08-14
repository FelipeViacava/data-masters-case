import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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
    cumulative_variance = np.cumsum(evr)
    index_80_percent = np.where(cumulative_variance >= 0.8)[0][0]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(range(1, len(evr)+1), evr, alpha=0.8, align='center')
    ax1.set_ylabel('Explained Variance Ratio', color='b')
    ax1.set_xlabel('Principal Component')
    for label in ax1.get_yticklabels():
        label.set_color("b")

    ax2 = ax1.twinx()
    ax2.step(range(1, len(evr)+1), cumulative_variance, where='mid', label='Cumulative Explained Variance', color='g')
    ax2.axvline(x=index_80_percent + 1, color='r', linestyle='--', label='80% of Explained Variance')
    ax2.set_ylabel('Cumulative Explained Variance', color='g')
    for label in ax2.get_yticklabels():
        label.set_color("g")

    ax1.set_ylim([0, max(evr)*1.1])
    ax2.set_ylim([0, 1])

    ax1.set_xlim([0, 30])
    ax2.set_xlim([0, 30])

    plt.tight_layout()
    plt.show()
    print(f"80% of variance is explained by {index_80_percent + 1} components")
    
def plot_components(components, feature_names) -> None:
    plt.figure(figsize=(12, 6))
    for i, component in enumerate(components):
        plt.subplot(2, 2, i + 1)
        plt.bar(feature_names, component)
        plt.title(f"Component {i+1}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()