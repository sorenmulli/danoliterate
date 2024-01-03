# %%
"""
This file is a jupytext notebook. Install jupytext and jupyter lab and right-click on the file -> Open as Notebook in JL
"""
# pylint: skip-file
# mypy: ignore-errors

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy

from danoliterate.evaluation.analysis.dimensions import Dimension

# %%
from danoliterate.evaluation.artifact_integration import get_scores_wandb

# %%
P = Path("/home/sorenmulli/Nextcloud/cand4/thesis/imgs/plots")

# %%
plt.rcParams["font.family"] = "serif"

# %%
scores = get_scores_wandb("nlgenda", "sorenmulli")
len(scores.scorings)

# %%
from danoliterate.evaluation.leaderboard.metric_parsing import default_choices, extract_metrics
from danoliterate.evaluation.leaderboard.table import build_leaderboard_table, get_table_values

# %% [markdown]
# # Main leaderboard analysis

# %%
chosen_metrics = default_choices(extract_metrics(scores, Dimension.CAPABILITY, "standard"))
table = get_table_values(chosen_metrics)
ld, _ = build_leaderboard_table(chosen_metrics, show_missing=False)
index = ld[ld.columns[0]]
table = table.loc[index.index]
table

# %%
index

# %%
plt.figure(figsize=(10, 7))
model_tab = table.loc[index[index.astype(int) > 10].index]
sns.heatmap(
    model_tab.T.corr() * 100,
    annot=True,
    fmt=".0f",
    cmap=plt.cm.Spectral,
    cbar_kws={"label": "Pearson Corr. [%]"},
)
plt.title("How Models Correlate Across Scenarios", fontsize=15)
plt.tight_layout()
plt.savefig(P / "model-corr.pdf")
plt.show()

# %%
plt.figure(figsize=(10, 7))

cmap = plt.cm.Spectral
colors = cmap(np.arange(cmap.N // 2, cmap.N))
mirrored_cmap = LinearSegmentedColormap.from_list("", np.vstack([colors[::-1], colors]))

corr = table.corr()
pdist = hierarchy.distance.pdist(corr.abs())
linkage = hierarchy.linkage(pdist, method="complete")
order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, pdist))
ordered_corr = corr.iloc[order, :].iloc[:, order]

sns.heatmap(
    ordered_corr * 100,
    annot=True,
    fmt=".0f",
    cmap=mirrored_cmap,
    cbar_kws={"label": "Pearson Corr. [%]"},
)
plt.title("How Scenarios Correlate", fontsize=15)

plt.tight_layout()
plt.savefig(P / "scenario-corr.pdf")

plt.show()

# %%
INTERESTING_SUBSET = (
    "OpenAI GPT 4",
    "OpenAI GPT 3.5 Turbo",
    "Google Gemini Pro",
    "LlaMa 2 13B Chat",
    "Mistral 7B Instruct (v0.2)",
    "Danoliterate Mistral 7B",
    "OpenAI Davinci 002",
)

# %%
dfs = []
for model, result in chosen_metrics["Nordjylland News"].items():
    if model in INTERESTING_SUBSET:
        df = pd.DataFrame(
            {"idx": result.example_results.keys(), model: result.example_results.values()}
        )
        df = df.set_index("idx")
        dfs.append(df)
nn_df = pd.concat(dfs, axis=1)
nn_df.corr()

# %%
plt.figure(figsize=(10, 5))
corr = nn_df.corr()

order = [model for model in index.index if model in nn_df.columns]

ordered_corr = corr.loc[order, :].loc[:, order]
plt.subplot(121)
sns.heatmap(
    ordered_corr * 100,
    annot=True,
    fmt=".0f",
    cbar=False,
    cmap=plt.cm.Spectral,
    vmin=-100,
    vmax=100,
)
plt.subplot(122)
for col in nn_df.columns:
    sns.kdeplot(nn_df[col], label=col)
plt.xlabel("Summary BERT score")
plt.legend(fontsize=7)

plt.suptitle("How Models Covary on Nordjylland News BERT-scores")
plt.tight_layout()
plt.savefig(P / "nn-corr.pdf")
plt.show()

# %%
order

# %%
plt.figure(figsize=(15, 15))

for i, dataset in enumerate(
    ("Citizenship Test", "HyggeSwag", "Da. Cloze Self Test", "Da. Gym 2000")
):
    dfs = []
    for model, result in chosen_metrics[dataset].items():
        if model in INTERESTING_SUBSET:
            df = pd.DataFrame(
                {
                    "idx": result.example_results.keys(),
                    "Ground Truth": [true for true, _ in result.example_results.values()],
                    model: [pred for _, pred in result.example_results.values()],
                }
            )
            df = df.set_index("idx")
            dfs.append(df)
    ct_df = pd.concat(dfs, axis=1)

    true_columns = ct_df.filter(like="Ground Truth").columns
    if not all(ct_df[true_columns].eq(ct_df[true_columns[0]], axis=0).all()):
        raise ValueError("Not all 'true' columns are identical")
    ct_df = ct_df.loc[:, ~ct_df.columns.duplicated(keep="first")]

    corr = ct_df.corr(lambda x, y: (x == y).mean())
    mt_order = ["Ground Truth", *order]
    ordered_corr = corr.loc[mt_order, :].loc[:, mt_order]
    plt.subplot(221 + i)
    sns.heatmap(
        ordered_corr * 100,
        annot=True,
        fmt=".0f",
        cmap=plt.cm.Spectral,
        vmin=-100,
        vmax=100,
        cbar=False,
    )
    plt.title(dataset)
plt.suptitle("Model Agreement Rates [%] on Multiple Choice Tasks", fontsize=15)

plt.tight_layout()
plt.savefig(P / "mc-agree.pdf")

plt.show()

# %% [markdown]
# # Qualitative error analysis

# %%
