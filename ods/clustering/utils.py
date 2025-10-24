import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix


def get_aligned_confusion_matrix(df: pd.DataFrame, true_column: str, predicted_columns: str) -> pd.DataFrame:
    cm = confusion_matrix(df[true_column].cat.codes, df[predicted_columns])
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned_cm = cm[:, col_ind][row_ind, :]
    return aligned_cm


def align_with_species(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        cm = confusion_matrix(df["species"].cat.codes, df[column])
        row_ind, col_ind = linear_sum_assignment(-cm)
        assert list(row_ind) == [0, 1, 2]
        indexer = list(col_ind).index
        df[column] = df[column].map(indexer)


def show_confusion_matrix(cm: pd.DataFrame, title: str, ax: plt.Axes | None = None) -> None:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Species")
    ax.set_xlabel("Predicted Species")


def scatter_comparison(
    df: pd.DataFrame,
    original_column: str,
    predicted_column: str,
    predicted_scaled_column: str,
    predicted_bivariate_column: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    sns.scatterplot(
        ax=ax[0], data=df, x="flipper_length_mm", y="bill_length_mm", hue=original_column, palette="Set2", s=80
    )

    sns.scatterplot(
        ax=ax[1],
        data=df,
        x="flipper_length_mm",
        y="bill_length_mm",
        hue=predicted_column,
        palette="Set2",
        s=80,
        style=["Correct" if correct else "Wrong" for correct in (df[predicted_column] == df[original_column].cat.codes)],
        markers={"Correct": "o", "Wrong": "X"}
    )
    sns.scatterplot(
        ax=ax[2],
        data=df,
        x="flipper_length_mm",
        y="bill_length_mm",
        hue=predicted_scaled_column,
        palette="Set2",
        s=80,
        style=["Correct" if correct else "Wrong" for correct in (df[predicted_scaled_column] == df[original_column].cat.codes)],
        markers={"Correct": "o", "Wrong": "X"}
    )
    sns.scatterplot(
        ax=ax[3],
        data=df,
        x="flipper_length_mm",
        y="bill_length_mm",
        hue=predicted_bivariate_column,
        palette="Set2",
        s=80,
        style=["Correct" if correct else "Wrong" for correct in (df[predicted_bivariate_column] == df[original_column].cat.codes)],
        markers={"Correct": "o", "Wrong": "X"}
    )

    ax[0].set_xlabel("Flipper Length (mm)")
    ax[0].set_ylabel("Bill Length (mm)")
    ax[0].set_title("Original Species")
    ax[0].legend(title="Species")

    ax[1].set_xlabel("Flipper Length (mm)")
    ax[1].set_ylabel("Bill Length (mm)")
    ax[1].set_title(f"{title} Clustering (MultiVariate)")
    ax[1].legend(title="Species")

    ax[2].set_xlabel("Flipper Length (mm)")
    ax[2].set_ylabel("Bill Length (mm)")
    ax[2].set_title(f"{title} Clustering (MultiVariate Scaled)")
    ax[2].legend(title="Species")

    ax[3].set_xlabel("Flipper Length (mm)")
    ax[3].set_ylabel("Bill Length (mm)")
    ax[3].set_title(f"{title} Clustering (BiVariate)")
    ax[3].legend(title="Species")

    return fig

def _penguin_generator_species(original_penguins: pd.DataFrame, num_penguins: int) -> pd.DataFrame:
    synthetic = pd.DataFrame()
    for col in original_penguins.columns:
        if original_penguins[col].dtype in ["object", "bool", "category"]:
            probs = original_penguins[col].value_counts(normalize=True)
            synthetic[col] = np.random.choice(probs.index, size=num_penguins, p=probs.values)
        else:
            # Fit KDE to continuous numeric data
            data = original_penguins[col].values
            kde = gaussian_kde(data)
            synthetic[col] = kde.resample(num_penguins).flatten()
    return synthetic


def penguin_generator(original_penguins: pd.DataFrame, penguins_per_species: int) -> pd.DataFrame:
    species = original_penguins["species"].unique()
    fake_penguins = []
    for specie in species:
        species_penguins = original_penguins[original_penguins["species"] == specie]
        fake_penguins.append(_penguin_generator_species(species_penguins, penguins_per_species))
    return pd.concat(fake_penguins, ignore_index=True)