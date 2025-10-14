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
