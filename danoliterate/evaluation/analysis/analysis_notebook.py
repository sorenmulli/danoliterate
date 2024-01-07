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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
corr_matrix = table.T.corr()
corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()

# %%
plt.figure(figsize=(10, 7))
model_tab = table.loc[index[index.astype(int) >= 20].index]
sns.heatmap(
    model_tab.T.corr() * 100,
    annot=True,
    fmt=".0f",
    cmap=plt.cm.Spectral,
    cbar_kws={"label": "Pearson Corr. [%]"},
        vmin=-100,
    vmax=100,
)
plt.title("How Models Correlate Across Scenarios", fontsize=15)
plt.tight_layout()
plt.savefig(P / "model-corr.pdf")
plt.show()

# %%
df_standardized = StandardScaler().fit_transform(table.T)
pca = PCA()
pcs = pca.fit_transform(df_standardized)
explained_variance = pca.explained_variance_ratio_*100
plt.figure()
plt.plot(range(0, len(explained_variance) + 1), [0, *np.cumsum(explained_variance)])
plt.xlabel('Number of Components')
plt.ylabel('Total Variance Prop. [%]')
plt.title('Scenario Variance is Explained by Model PCs')
plt.grid(True)
plt.savefig(P / "pca-model-var.pdf")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Import the colormap

# Assume loadings and feature_names are defined as per your context
loadings = pca.components_  # Shape n_components x n_features
feature_names = np.array(table.T.columns)  # Ensure feature_names is a numpy array for advanced indexing

# Constants
num_pc_to_display = 3  # or any other number of components you are interested in
num_top_features = 28  # Number of top features to display

for i in range(num_pc_to_display):
    # Sorting the loadings of the i-th PC by absolute value while keeping track of the indices
    sorted_indices = np.argsort(np.abs(loadings[i]))[::-1]
    top_indices = sorted_indices[:num_top_features]

    # Selecting the top loadings and the corresponding feature names
    top_loadings = loadings[i][top_indices]
    top_feature_names = feature_names[top_indices]

    # Normalize the absolute values of the loadings between 0 and 1 for the colormap
    norm = plt.Normalize(np.min(top_loadings), np.max(top_loadings))
    colors = plt.cm.Spectral(norm(top_loadings))  # Get colors from the colormap

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 7))
    y_pos = np.arange(num_top_features)
    ax.barh(y_pos, top_loadings, align='center', color=colors)  # Add color to the bars
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Loading')
    ax.set_title(f'Principal Component {i+1}')
    plt.tight_layout()
    # Update the path as needed
    plt.savefig(P / f"pca-model-load-{i+1}.pdf")  
    plt.show()  

# %%
corr_matrix = table.corr().abs()
corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()

# %%
plt.figure(figsize=(10, 7))

cmap = plt.cm.Spectral
colors = cmap(np.arange(cmap.N // 2, cmap.N))
mirrored_cmap = LinearSegmentedColormap.from_list("", np.vstack([colors[::-1], colors]))

corr = table.corr()
pdist = hierarchy.distance.pdist(corr.abs())
linkage = hierarchy.linkage(pdist, method="complete")
order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, pdist))
#corr = table.corr("spearman")
ordered_corr = corr.iloc[order, :].iloc[:, order]

sns.heatmap(
    ordered_corr * 100,
    annot=True,
    fmt=".0f",
    cmap=mirrored_cmap,
    #cbar_kws={"label": "Spearman Rank Corr. [%]"},
    cbar_kws={"label": "Pearson Corr. [%]"},
    vmin=-100,
    vmax=100,

)
plt.title("How Scenario Results Correlate", fontsize=15)
#plt.title("How Scenario Ranks Correlate", fontsize=15)

plt.tight_layout()
plt.savefig(P / "scenario-corr.pdf")
#plt.savefig(P / "scenario-rank-corr.pdf")

plt.show()

# %%
_table = table.copy()
_table["#twitterhjerne"] *= -1
df_standardized = StandardScaler().fit_transform(_table)
pca = PCA()
pcs = pca.fit_transform(df_standardized)
explained_variance = pca.explained_variance_ratio_*100
plt.figure()
plt.plot(range(0, len(explained_variance) + 1), [0, *np.cumsum(explained_variance)])
plt.xlabel('Number of Components')
plt.ylabel('Total Variance Prop. [%]')
plt.title('Model Variance Explained by Scenario PCs')
plt.grid(True)
plt.savefig(P / "pca-scenario-var.pdf")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Import the colormap

# Assume loadings and feature_names are defined as per your context
loadings = pca.components_  # Shape n_components x n_features
feature_names = np.array(table.columns)  # Ensure feature_names is a numpy array for advanced indexing

# Constants
num_pc_to_display = 3  # or any other number of components you are interested in
num_top_features = 8  # Number of top features to display

for i in range(num_pc_to_display):
    # Sorting the loadings of the i-th PC by absolute value while keeping track of the indices
    sorted_indices = np.argsort(np.abs(loadings[i]))[::-1]
    top_indices = sorted_indices[:num_top_features]

    # Selecting the top loadings and the corresponding feature names
    top_loadings = loadings[i][top_indices]
    top_feature_names = feature_names[top_indices]

    # Normalize the absolute values of the loadings between 0 and 1 for the colormap
    norm = plt.Normalize(min(np.min(top_loadings), 0), np.max(top_loadings))
    colors = plt.cm.Spectral(norm(top_loadings))  # Get colors from the colormap

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5))
    y_pos = np.arange(num_top_features)
    ax.barh(y_pos, top_loadings, align='center', color=colors)  # Add color to the bars
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Loading')
    ax.set_title(f'Principal Component {i+1}')
    plt.tight_layout()
    # Update the path as needed
    plt.savefig(P / f"pca-scenario-load-{i+1}.pdf")  
    plt.show()  

# %%
INTERESTING_SUBSET = (
    "OpenAI GPT 4",
    "OpenAI GPT 3.5 Turbo",
    "SOLAR 10.7B Instruct",
    "Google Gemini Pro",
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
from danoliterate.evaluation.artifact_integration import get_results_wandb

# %%
all_res = get_results_wandb(
    "nlgenda",
    "sorenmulli",
    "/home/sorenmulli/Nextcloud/cand4/framework/local-computations/wandb-cache.json",
)
len(all_res)

# %%

# %%
from collections import defaultdict

interesting_executions = defaultdict(dict)
for res in all_res:
    if (mname := res.metadata.model_cfg["name"]) in INTERESTING_SUBSET:
        interesting_executions[res.metadata.scenario_cfg["name"]][mname] = res


# %%
def save_results(scenario, col, name):
    df = pd.DataFrame()
    for model, res in interesting_executions[scenario].items():
        exes = [ex for ex in res.examples if ex.id_ in col.index]
        df[model] = [ex.generated_text for ex in exes]
    df = df.set_index(col.index)
    df.insert(0, "Prompt", [ex.prompt for ex in exes])
    df.insert(0, "Answer", [ex.target_answer or ex.options[ex.index_label] for ex in exes])

    df.to_csv(
        Path("/home/sorenmulli/Nextcloud/cand4/framework/local-data") / f"{scenario}-{name}.csv"
    )


# %%
for scenario, models in chosen_metrics.items():
    if scenario in {"DaNE", "#twitterhjerne"}:
        continue
    print(scenario)
    dfs = []
    for model, result in models.items():
        if "baseline" in model.lower():
            continue
        df = pd.DataFrame(
            {
                "idx": result.example_results.keys(),
                model: [
                    x if isinstance(x, float) else int(x[0] == x[1])
                    for x in result.example_results.values()
                ],
            }
        )
        df = df.set_index("idx")
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df["Mean"] = df.apply(np.mean, axis=1)
    df = df.sort_values("Mean")
    print("Top easiest")
    print(df.tail(10).Mean)
    save_results(scenario, df.tail(10)["Mean"], "easiest")

    print("Top hardest")
    print(df.head(10).Mean)
    save_results(scenario, df.head(10)["Mean"], "hardest")

    df["Std"] = df.apply(np.std, axis=1)
    df = df.sort_values("Std")

    print("Top same performance")
    print(df.head(10).Std)
    save_results(scenario, df.head(10)["Std"], "same")

    print("Top different performance")
    print(df.tail(10).Std)
    save_results(scenario, df.tail(10)["Std"], "different")
    print()

# %%
