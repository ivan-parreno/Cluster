import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def get_aligned_confusion_matrix(df: pd.DataFrame, true_column: str, predicted_columns: str) -> pd.DataFrame:
    categorizer = dict(zip(df[true_column].cat.codes, df[true_column]))

    cm = confusion_matrix(df[true_column].cat.codes, df[predicted_columns])
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))

    aligned_clusters = [categorizer[mapping[c]] for c in df[predicted_columns]]
    cm_aligned = confusion_matrix(df[true_column], aligned_clusters)
    cm_aligned = pd.DataFrame(cm_aligned)
    cm_aligned.index = cm_aligned.index.map(categorizer)
    return cm_aligned


def show_confusion_matrix(cm: pd.DataFrame, title: str, ax: plt.Axes | None = None) -> None:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Species")
    ax.set_xlabel("Predicted Species")


def scatter_comparison(
    df: pd.DataFrame,
    original_column: str,
    predicted_column: str,
    predicted_bivariate_column: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    sns.scatterplot(
        ax=ax[0], data=df, x="flipper_length_mm", y="bill_length_mm", hue=original_column, palette="Set2", s=80
    )
    sns.scatterplot(
        ax=ax[1], data=df, x="flipper_length_mm", y="bill_length_mm", hue=predicted_column, palette="Set2", s=80
    )
    sns.scatterplot(
        ax=ax[2],
        data=df,
        x="flipper_length_mm",
        y="bill_length_mm",
        hue=predicted_bivariate_column,
        palette="Set2",
        s=80,
    )

    ax[0].set_xlabel("Flipper Length (mm)")
    ax[0].set_ylabel("Bill Length (mm)")
    ax[0].set_title("Original Species")
    ax[0].legend(title="Species")

    ax[1].set_xlabel("Flipper Length (mm)")
    ax[1].set_ylabel("Bill Length (mm)")
    ax[1].set_title(f"{title} Clustering")
    ax[1].legend(title="Species")

    ax[2].set_xlabel("Flipper Length (mm)")
    ax[2].set_ylabel("Bill Length (mm)")
    ax[2].set_title(f"{title} Clustering (BiVariate)")
    ax[2].legend(title="Species")
