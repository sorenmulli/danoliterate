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

# %%
freegen = default_choices(extract_metrics(scores, Dimension.CAPABILITY, "free-generation"))


# %%
freegen

# %%
