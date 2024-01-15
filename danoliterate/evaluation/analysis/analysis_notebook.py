# %%
"""
This file is a jupytext notebook. Install jupytext and jupyter lab and right-click on the file -> Open as Notebook in JL
"""
# pylint: skip-file
# mypy: ignore-errors

# %%
# %load_ext autoreload
# %autoreload 2

import json
from collections import Counter, defaultdict

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
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
from danoliterate.evaluation.leaderboard.table import (
    build_leaderboard_table,
    format_table_for_latex,
    get_table_values,
)


# %%
def exclude_models(metric_struct, to_exclude):
    return {
        s: {m: me for m, me in models.items() if m not in to_exclude}
        for s, models in metric_struct.items()
    }


# %% [markdown]
# # Main leaderboard analysis

# %%
SCENARIO_ORDER = (
    "Citizenship Test",
    "HyggeSwag",
    "#twitterhjerne",
    "Da. Cloze Self Test",
    "Da. Gym 2000",
    "Nordjylland News",
    "DaNE",
    "Angry Tweets",
)
ex_met = extract_metrics(scores, Dimension.CAPABILITY, "standard")
chosen_metrics = default_choices(ex_met)
table = get_table_values(chosen_metrics)
ld, lower = build_leaderboard_table(chosen_metrics, show_missing=False)
ld = ld.loc[:, [ld.columns[0], *SCENARIO_ORDER]]
print(format_table_for_latex(ld, lower))
index = ld[ld.columns[0]]
table = table.loc[index.index]

# %%

# %%
extraaa = exclude_models(
    extract_metrics(scores, Dimension.CALIBRATION, "standard"),
    [model for model in ld.index if model not in INTERESTING_SUBSET3] + ["Constant Baseline"],
)

# %%
SCENARIO_ORDER = (
    "Citizenship Test",
    "HyggeSwag",
    "#twitterhjerne",
    "Da. Cloze Self Test",
    "Da. Gym 2000",
    "Nordjylland News",
    "DaNE",
    "Angry Tweets",
)
extrooo = {
    scenario: {model: metrics[1] for model, metrics in models.items()}
    for scenario, models in extraaa.items()
}
cal_ld, lower = build_leaderboard_table(
    extrooo,
    # default_choices(
    #    extraaa
    # ),
    show_missing=True,
)
cal_ld = cal_ld.loc[
    :, pd.Series([ld.columns[0], *[o for o in SCENARIO_ORDER if o in cal_ld.columns]])
]
print(format_table_for_latex(cal_ld, lower))
cal_ld

# %%
SCENARIO_ORDER = (
    "Citizenship Test",
    "HyggeSwag",
    "#twitterhjerne",
    "Da. Cloze Self Test",
    "Da. Gym 2000",
    "Nordjylland News",
    "DaNE",
    "Angry Tweets",
)
cal_ld, lower = build_leaderboard_table(
    default_choices(
        exclude_models(
            extract_metrics(scores, Dimension.CALIBRATION, "standard"), ["Constant Baseline"]
        )
    ),
    show_missing=True,
)
cal_ld = cal_ld.loc[
    :, pd.Series([ld.columns[0], *[o for o in SCENARIO_ORDER if o in cal_ld.columns]])
]
print(format_table_for_latex(cal_ld, lower))
cal_ld

# %%
SCENARIO_ORDER = (
    "Citizenship Test",
    "HyggeSwag",
    "#twitterhjerne",
    "Da. Cloze Self Test",
    "Da. Gym 2000",
    "Nordjylland News",
    "DaNE",
    "Angry Tweets",
)
cal_ld, lower = build_leaderboard_table(
    default_choices(
        exclude_models(
            extract_metrics(scores, Dimension.EFFICIENCY, "standard"),
            ["Constant Baseline", *[m for m in ld.index if "OpenAI" in m or "Google" in m]],
        )
    ),
    efficiency=True,
    show_missing=False,
)
cal_ld = cal_ld.loc[
    :, pd.Series([ld.columns[0], *[o for o in SCENARIO_ORDER if o in cal_ld.columns]])
]
print(format_table_for_latex(cal_ld, lower))

# %%
SCENARIO_ORDER = (
    "Citizenship Test",
    "HyggeSwag",
    "#twitterhjerne",
    "Da. Cloze Self Test",
    "Da. Gym 2000",
    "Nordjylland News",
    "DaNE",
    "Angry Tweets",
)
fairs = exclude_models(
    extract_metrics(scores, Dimension.TOXICITY, "standard"), ["Constant Baseline"]
)
fairs = {s: f for s, f in fairs.items() if s in ("#twitterhjerne", "Nordjylland News")}
cal_ld, lower = build_leaderboard_table(default_choices(fairs), show_missing=False)
cal_ld = cal_ld.loc[
    :, pd.Series([ld.columns[0], *[o for o in SCENARIO_ORDER if o in cal_ld.columns]])
]
print(format_table_for_latex(cal_ld, lower))

# %%
to_show = {
    "Keystroke robustness: #twitterhjerne": "#twitterhjerne",
    "Keystroke robustness: Angry Tweets": "Angry Tweets",
    "Keystroke robustness: Nordjylland News": "Nordjylland News",
}
fairs = default_choices(extract_metrics(scores, Dimension.ROBUSTNESS, "standard"))
fairs = {s: f for s, f in fairs.items() if s in to_show}
cal_ld, lower = build_leaderboard_table(fairs, show_missing=True, reverse_abso_sort=True)
cal_ld = cal_ld.loc[:, pd.Series([ld.columns[0], *[o for o in to_show]])]
cal_ld = cal_ld.rename(to_show, axis=1)
print(format_table_for_latex(cal_ld, to_show.values(), abso_num=True))

cal_ld

# %%
to_show = {
    "Female to male disparity: #twitterhjerne": "Female/male #twi.",
    "Female to male disparity: Nordjylland News": "Female/male NN",
    "Female to male disparity: Angry Tweets": "Female/male AT",
    "Muslim to Danish disparity: #twitterhjerne": "Muslim/Danish #twi",
    "Muslim to Danish disparity: Nordjylland News": "Muslim/Danish NN",
    "Muslim to Danish disparity: Angry Tweets": "Muslim/Danish AT",
}
fairs = default_choices(extract_metrics(scores, Dimension.FAIRNESS, "standard"))
fairs = {s: f for s, f in fairs.items() if s in to_show}
cal_ld, lower = build_leaderboard_table(fairs, show_missing=True, reverse_abso_sort=True)
cal_ld = cal_ld.loc[:, pd.Series([ld.columns[0], *[o for o in to_show]])]
cal_ld = cal_ld.rename(to_show, axis=1)
print(format_table_for_latex(cal_ld, to_show.values(), abso_num=True))

cal_ld

# %%
index

# %%
corr_matrix = table.T.corr()
corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()

# %%
plt.figure(figsize=(10, 7))
model_tab = table.loc[index[index.astype(int) >= 19].index]
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
explained_variance = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(explained_variance) + 1), [0, *np.cumsum(explained_variance)])
plt.xlabel("Number of Components")
plt.ylabel("Total Variance Prop. [%]")
plt.title("Scenario Variance is Explained by Model PCs")
plt.grid(True)
plt.savefig(P / "pca-model-var.pdf")
plt.show()

import matplotlib.pyplot as plt

# %%
import numpy as np
from matplotlib import cm  # Import the colormap

# Assume loadings and feature_names are defined as per your context
loadings = pca.components_  # Shape n_components x n_features
feature_names = np.array(
    table.T.columns
)  # Ensure feature_names is a numpy array for advanced indexing

# Constants
num_pc_to_display = 3  # or any other number of components you are interested in
num_top_features = len(table)  # Number of top features to display

for i in range(num_pc_to_display):
    # Sorting the loadings of the i-th PC by absolute value while keeping track of the indices
    sorted_indices = np.argsort((loadings[i]))[::-1]
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
    ax.barh(y_pos, top_loadings, align="center", color=colors)  # Add color to the bars
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Loading")
    ax.set_title(f"Principal Component {i+1}")
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
# corr = table.corr("spearman")
ordered_corr = corr.iloc[order, :].iloc[:, order]

sns.heatmap(
    ordered_corr * 100,
    annot=True,
    fmt=".0f",
    cmap=mirrored_cmap,
    # cbar_kws={"label": "Spearman Rank Corr. [%]"},
    cbar_kws={"label": "Pearson Corr. [%]"},
    vmin=-100,
    vmax=100,
)
plt.title("How Scenario Results Correlate", fontsize=15)
# plt.title("How Scenario Ranks Correlate", fontsize=15)

plt.tight_layout()
plt.savefig(P / "scenario-corr.pdf")
# plt.savefig(P / "scenario-rank-corr.pdf")

plt.show()

# %%
_table = table.copy()
_table["#twitterhjerne"] *= -1
df_standardized = StandardScaler().fit_transform(_table)
pca = PCA()
pcs = pca.fit_transform(df_standardized)
explained_variance = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(5, 2))
plt.plot(range(0, len(explained_variance) + 1), [0, *np.cumsum(explained_variance)])
plt.xlabel("Number of Components")
plt.ylabel("Total Variance Prop. [%]")
plt.title("Model Variance Explained by Scenario PCs")
plt.grid(True)
plt.savefig(P / "pca-scenario-var.pdf")
plt.show()

import matplotlib.pyplot as plt

# %%
import numpy as np
from matplotlib import cm  # Import the colormap

# Assume loadings and feature_names are defined as per your context
loadings = pca.components_  # Shape n_components x n_features
feature_names = np.array(
    table.columns
)  # Ensure feature_names is a numpy array for advanced indexing

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
    ax.barh(y_pos, top_loadings, align="center", color=colors)  # Add color to the bars
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Loading")
    ax.set_title(f"Principal Component {i+1}")
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
    "Mistral 7B",
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
    sns.kdeplot([x for x in nn_df[col] if x > 0.3], label=col)
plt.xlabel("Summary BERT score")
plt.ylim(0, 15)
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
from collections import defaultdict

interesting_executions = defaultdict(dict)
for res in all_res:
    if (mname := res.metadata.model_cfg["name"]) in [
        *INTERESTING_SUBSET,
        "Danoliterate LlaMa 2 7B",
        "LlaMa 2 7B",
        "LlaMa 2 7B Chat",
    ]:
        if (
            res.metadata.augmenter_key is None
            and res.metadata.scenario_cfg.get("type", "standard") == "standard"
        ):
            if interesting_executions[res.metadata.scenario_cfg["name"]].get(mname):
                raise
            interesting_executions[res.metadata.scenario_cfg["name"]][mname] = res

# %%
from collections import defaultdict
dfs = {s: pd.DataFrame() for s in SCENARIO_ORDER}
for res in all_res:
    if (
        res.metadata.augmenter_key is None
        and res.metadata.scenario_cfg.get("type", "standard") == "standard"
    ):
        s = res.metadata.scenario_cfg["name"]
        m = res.metadata.model_cfg["name"]
        if "prompt" not in dfs[s].columns:
            dfs[s]["prompt"] = pd.Series([ex.prompt for ex in res.examples], index=[ex.id_ for ex in res.examples])
        dfs[s][m] = pd.Series([ex.generated_text for ex in res.examples], index=[ex.id_ for ex in res.examples])
for name, df in dfs.items():
    df[:5].to_csv(f"/home/sorenmulli/Nextcloud/cand4/framework/danoliterate/evaluation/leaderboard/pages/assets/{name}.csv")

# %%
ex = " Det 19-årige stortalent i speedway Mikkel B. Andersen er blevet udtaget til landsholdet af træner Hans Nielsen. I første omgang er Mikkel B. Andersen, der til daglig står i lære ved Peugeot i Bejstrup ved Fjerritslev, udtaget til den ni mand store bruttotrup, og kun fem kørere skal på banen når landsholdet 23. juli kører VM-semifinale i Vojens"
from danoliterate.evaluation.execution.augmentation import (
    DanishNameInserter,
    FemaleNameInserter,
    KeystrokeErrorAdder,
    MaleNameInserter,
    MuslimNameInserter,
)

for aug in (
    KeystrokeErrorAdder,
    MaleNameInserter,
    FemaleNameInserter,
    DanishNameInserter,
    MuslimNameInserter,
):
    print(aug()(ex))

# %%
gpt_4_at = {}
for res in all_res:
    if "GPT 4" in (mname := res.metadata.model_cfg["name"]):
        if (
            res.metadata.scenario_cfg.get("type", "standard") == "standard"
            and res.metadata.augmenter_key == "keystroke-error"
            and res.metadata.scenario_cfg["name"] == "Angry Tweets"
        ):
            gpt_4_at[mname] = res
exes = {m: [] for m in gpt_4_at}
for m_n, m_t in zip(gpt_4_at["OpenAI GPT 4"].examples, gpt_4_at["OpenAI GPT 4 Turbo"].examples):
    assert m_n.id_ == m_t.id_
    assert m_n.prompt == m_t.prompt
    assert m_n.options[m_n.index_label] == m_t.options[m_t.index_label]
    print(m_n.prompt, m_n.generated_text, m_t.generated_text, m_n.options[m_n.index_label])
    break
abe = pd.Series([m_n.generated_text for m_n in gpt_4_at["OpenAI GPT 4"].examples])
print(abe.value_counts())
abe = pd.Series([m_n.generated_text for m_n in gpt_4_at["OpenAI GPT 4 Turbo"].examples])
print(abe.value_counts())

# %%
for res in all_res:
    if (
        "DaNE" in res.metadata.scenario_cfg["name"]
        and res.metadata.scenario_cfg.get("type", "standard") == "standard"
        and res.metadata.augmenter_key is None
    ):
        print(res.examples[0].prompt)
        print()
        break

# %%
foodstuffs = default_choices(extract_metrics(scores, Dimension.TOXICITY, "standard"))


# %%
for res in all_res:
    if (
        "Nordjylland" in (sn := res.metadata.scenario_cfg["name"])
        and res.metadata.scenario_cfg.get("type", "standard") == "standard"
        and res.metadata.augmenter_key is None
    ):
        mn = res.metadata.model_cfg["name"]
        if "Google" not in mn:
            continue
        if (exes := foodstuffs[sn].get(mn)) is not None:
            for ex in res.examples:
                if (p := exes.example_results[ex.id_]) > 0.05:
                    print(ex.id_, p, mn, ex.generated_text)

# %%
interesting_executions["HyggeSwag"]["Danoliterate Mistral 7B"].examples[1].generated_text


# %%
def save_results(scenario, col, name):
    df = pd.DataFrame()
    for model, res in interesting_executions[scenario].items():
        all_exes = {ex.id_: ex for ex in res.examples}
        exes = [all_exes[idx] for idx in col.index]
        df[model] = [ex.generated_text for ex in exes]
    df = df.set_index(col.index)
    df.insert(0, "Prompt", [ex.prompt for ex in exes])
    df.insert(
        0,
        "Answer",
        [
            ex.target_answer or (ex.options[ex.index_label] if ex.index_label is not None else "")
            for ex in exes
        ],
    )

    df.to_csv(
        Path("/home/sorenmulli/Nextcloud/cand4/framework/local-data") / f"{scenario}-{name}.csv"
    )


# %%
for scenario, models in chosen_metrics.items():
    if scenario not in {"#twitterhjerne"}:  # , "Hyggeswag", "#twitterhjerne"}:
        continue
    print(scenario)
    dfs = []
    for model, result in models.items():
        if int(index[model]) < 25:
            continue
        df = pd.DataFrame(
            {
                "idx": result.example_results.keys(),
            }
        )
        df[model] = pd.Series(
            [
                x if isinstance(x, float) else float(x[0] == x[1])
                for x in result.example_results.values()
            ]
        )
        for idx, val in result.example_results.items():
            if not isinstance(val, float):
                if int(val[0]) == 0:
                    df[df.idx == idx] = float("nan")
        df = df.dropna()
        df = df.set_index("idx")
        dfs.append(df)
    df = pd.concat(dfs, axis=1)

    df["Mean"] = df.apply(np.mean, axis=1)
    df = df.sort_values("Mean")
    print("Top easiest")
    print(df.tail(10).Mean)
    save_results(scenario, df.tail(10)["Mean"], "easiest")

    print("Top hardest")
    print(df.head(45).Mean)
    save_results(scenario, df.head(45)["Mean"], "hardest")

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
INTERESTING_SUBSET2 = (
    "OpenAI GPT 3.5 Turbo",
    "Google Gemini Pro",
    "Mistral 7B Instruct (v0.2)",
    "Mistral 7B",
    "Danoliterate Mistral 7B",
    "LlaMa 2 7B",
    "Danoliterate LlaMa 2 7B",
    "Constant Baseline",
)

# %%
main_table = ld.loc[pd.Series(INTERESTING_SUBSET2)]
main_table

# %%
freegen_nlg_metrics = {}
for s, models in extract_metrics(scores, Dimension.CAPABILITY, "free-generation").items():
    if s in {"DaNE", "#twitterhjerne", "Nordjylland News"}:
        continue
    freegen_nlg_metrics[s] = {}
    for m, metrics in models.items():
        met = [me for me in metrics if me.short_name == "Accuracy (NLG BERT similarity)"]
        if met:
            freegen_nlg_metrics[s][m] = met[0]
freegen_nlg_ld, _ = build_leaderboard_table(freegen_nlg_metrics, show_missing=True)

options_lm_metrics = {}
for s, models in extract_metrics(scores, Dimension.CAPABILITY, "standard").items():
    if s in {"DaNE", "#twitterhjerne", "Nordjylland News"}:
        continue
    options_lm_metrics[s] = {}
    for m, metrics in models.items():
        met = [me for me in metrics if me.short_name == "Accuracy (LM)"]
        if met:
            options_lm_metrics[s][m] = met[0]
options_lm_ld, _ = build_leaderboard_table(options_lm_metrics, show_missing=True)

freegen_lm_metrics = {}
for s, models in extract_metrics(scores, Dimension.CAPABILITY, "free-generation").items():
    if s in {"DaNE", "#twitterhjerne", "Nordjylland News"}:
        continue
    freegen_lm_metrics[s] = {}
    for m, metrics in models.items():
        met = [me for me in metrics if me.short_name == "Accuracy (LM)"]
        if met:
            freegen_lm_metrics[s][m] = met[0]
freegen_lm_ld, _ = build_leaderboard_table(freegen_lm_metrics, show_missing=True)

# %%
ct_versions = pd.DataFrame()
ct_versions["Options+NLG"] = ld["Citizenship Test"]
ct_versions["Options+LM"] = options_lm_ld["Citizenship Test"]
ct_versions["Free+NLG"] = freegen_nlg_ld["Citizenship Test"]
ct_versions["Free+LM"] = freegen_lm_ld["Citizenship Test"]
ct_versions = ct_versions.loc[pd.Series(INTERESTING_SUBSET2)]
print(format_table_for_latex(ct_versions, {}))
ct_versions

# %%
hs_versions = pd.DataFrame()
hs_versions["Options+NLG"] = ld["HyggeSwag"]
hs_versions["Options+LM"] = options_lm_ld["HyggeSwag"]
hs_versions["Free+NLG"] = freegen_nlg_ld["HyggeSwag"]
hs_versions["Free+LM"] = freegen_lm_ld["HyggeSwag"]
hs_versions = hs_versions.loc[pd.Series(INTERESTING_SUBSET2)]
print(format_table_for_latex(hs_versions, {}))
hs_versions

# %%
INTERESTING_SUBSET3 = (
    "OpenAI GPT 4",
    "OpenAI GPT 3.5 Turbo",
    "SOLAR 10.7B Instruct",
    "Google Gemini Pro",
    "Mistral 7B Instruct (v0.2)",
    "LlaMa 2 13B Chat",
    "OpenAI Davinci 002",
    "Danoliterate Mistral 7B",
    "Mistral 7B",
    "Danoliterate LlaMa 2 7B",
    "LlaMa 2 7B",
    "Constant Baseline",
)

# %%
th_metric_names = {
    "Prediction odd-one-out frequency (BERT similarity)": "Pred odd-one-out freq.",
    "Avg. similarity to references (BERT similarity)": "Avg. similarity",
    "Min. similarity to references (BERT similarity)": "Min, similarity",
    "Max. similarity to references (BERT similarity)": "Max. similarity",
}
th_metrics = defaultdict(dict)
for m, metrics in extract_metrics(scores, Dimension.CAPABILITY, "standard")[
    "#twitterhjerne"
].items():
    for met, name in th_metric_names.items():
        th_metrics[name][m] = [me for me in metrics if me.short_name == met][0]
th_metrics_ld, lower = build_leaderboard_table(th_metrics, show_missing=True)
th_metrics_ld = th_metrics_ld.loc[:, pd.Series(th_metric_names.values())]
th_metrics_ld = th_metrics_ld.loc[
    pd.Series([idx for idx in th_metrics_ld.index if idx in INTERESTING_SUBSET3])
]
print(format_table_for_latex(th_metrics_ld, lower))
th_metrics_ld

# %%
nn_metric_names = {
    "Similarity (BERT similarity)": "BERT similarity",
    "Similarity (ROUGE-1)": "ROUGE-1",
    "Similarity (ROUGE-L)": "ROUGE-L",
}
nn_metrics = defaultdict(dict)
for m, metrics in extract_metrics(scores, Dimension.CAPABILITY, "standard")[
    "Nordjylland News"
].items():
    for met, name in nn_metric_names.items():
        nn_metrics[name][m] = [me for me in metrics if me.short_name == met][0]
nn_metrics_ld, lower = build_leaderboard_table(nn_metrics, show_missing=True)
nn_metrics_ld = nn_metrics_ld.loc[:, pd.Series(nn_metric_names.values())]
nn_metrics_ld = nn_metrics_ld.loc[
    pd.Series([idx for idx in nn_metrics_ld.index if idx in INTERESTING_SUBSET3])
]
print(format_table_for_latex(nn_metrics_ld, lower))
nn_metrics_ld

# %%
cal_metric_names = {
    "Brier Score (LM)": "Brier Score",
    "ECE Calibration (LM)": "ECE",
    "Accuracy (LM)": "Accuracy (LM)",
}
cal_metrics = defaultdict(dict)
for m, metrics in extract_metrics(scores, Dimension.CALIBRATION, "standard")[
    "Angry Tweets"
].items():
    for met, name in cal_metric_names.items():
        try:
            cal_metrics[name][m] = [me for me in metrics if me.short_name == met][0]
        except:
            ...
for m, metrics in extract_metrics(scores, Dimension.CAPABILITY, "standard")["Angry Tweets"].items():
    for met, name in cal_metric_names.items():
        try:
            cal_metrics[name][m] = [me for me in metrics if me.short_name == met][0]
        except:
            ...
cal_metrics_ld, lower = build_leaderboard_table(cal_metrics, show_missing=True)
cal_metrics_ld = cal_metrics_ld.loc[:, pd.Series(cal_metric_names.values())]
# print(format_table_for_latex(cal_metrics_ld, lower))
# cal_metrics_ld

# %%
ct_prompt_metrics = {
    "Standard Structured Prompt": extract_metrics(scores, Dimension.CAPABILITY, "standard")[
        "Citizenship Test"
    ],
    "Simple Question": extract_metrics(scores, Dimension.CAPABILITY, "alternative-prompt")[
        "Citizenship Test"
    ],
}
ct_prompt_metrics = default_choices(ct_prompt_metrics)

ct_prompt_ld, lower = build_leaderboard_table(ct_prompt_metrics, show_missing=True)
ct_prompt_ld = ct_prompt_ld.loc[
    pd.Series([idx for idx in ct_prompt_ld.index if idx in INTERESTING_SUBSET2])
]
ct_prompt_ld = ct_prompt_ld.loc[:, pd.Series(ct_prompt_metrics.keys())]
print(format_table_for_latex(ct_prompt_ld, lower))
ct_prompt_ld

# %%
dg_prompt_metrics = {
    "Standard Danish Prompt": extract_metrics(scores, Dimension.CAPABILITY, "standard")[
        "Da. Gym 2000"
    ],
    "English Prompt Text": extract_metrics(scores, Dimension.CAPABILITY, "alternative-prompt")[
        "Da. Gym 2000"
    ],
}
dg_prompt_metrics = default_choices(dg_prompt_metrics)

dg_prompt_ld, lower = build_leaderboard_table(dg_prompt_metrics, show_missing=True)
dg_prompt_ld = dg_prompt_ld.loc[
    pd.Series([idx for idx in dg_prompt_ld.index if idx in INTERESTING_SUBSET2])
]
dg_prompt_ld = dg_prompt_ld.loc[:, pd.Series(dg_prompt_metrics.keys())]
print(format_table_for_latex(dg_prompt_ld, lower))
dg_prompt_ld

# %%
nn_prompt_metrics = {
    "Standard Simple Prompt": extract_metrics(scores, Dimension.CAPABILITY, "standard")[
        "Nordjylland News"
    ],
    "Detailed Instructions in Prompt": extract_metrics(
        scores, Dimension.CAPABILITY, "alternative-prompt"
    )["Nordjylland News"],
}
nn_prompt_metrics = default_choices(nn_prompt_metrics)

nn_prompt_ld, lower = build_leaderboard_table(nn_prompt_metrics, show_missing=True)
nn_prompt_ld = nn_prompt_ld.loc[
    pd.Series([idx for idx in nn_prompt_ld.index if idx in INTERESTING_SUBSET2])
]
nn_prompt_ld = nn_prompt_ld.loc[:, pd.Series(nn_prompt_metrics.keys())]
print(format_table_for_latex(nn_prompt_ld, lower))
nn_prompt_ld

# %%
da_prompt_metrics = {
    "1-shot": extract_metrics(scores, Dimension.CAPABILITY, "few-shot-experiment-1")["DaNE"],
    "3-shot (standard)": extract_metrics(scores, Dimension.CAPABILITY, "standard")["DaNE"],
    "5-shot": extract_metrics(scores, Dimension.CAPABILITY, "few-shot-experiment-5")["DaNE"],
}
da_prompt_metrics = default_choices(da_prompt_metrics)

da_prompt_ld, lower = build_leaderboard_table(da_prompt_metrics, show_missing=True)
da_prompt_ld = da_prompt_ld.loc[
    pd.Series([idx for idx in da_prompt_ld.index if idx in INTERESTING_SUBSET2])
]
da_prompt_ld = da_prompt_ld.loc[:, pd.Series(da_prompt_metrics.keys())]
print(format_table_for_latex(da_prompt_ld, lower))
da_prompt_ld

# %%
from pelutils.ds.plots import moving_avg

# %%
for model, name in zip(
    ("llama", "mistral", "baseline", "mistral-full"),
    (
        "Danoliterate LlaMa 2 7B",
        "Danoliterate Mistral 7B",
        "Danoliterate Baseline LLM",
        "Danoliterate Mistral 7B",
    ),
):
    state_path = (
        Path("/home/sorenmulli/Nextcloud/cand4/framework/local-data/trainer-states")
        / f"{model}.json"
    )
    with open(state_path, "r") as f:
        res = json.load(f)["log_history"]
    eval_losses = []
    train_losses = []
    for log in res:
        if (train_loss := log.get("loss")) is not None:
            train_losses.append((log["step"], train_loss))
        if (eval_loss := log.get("eval_loss")) is not None:
            eval_losses.append((log["step"], eval_loss))
    plt.figure(figsize=(10, 5) if model in {"llama", "mistral-full"} else (5, 5))
    plt.plot(*np.array(eval_losses).T, label="Evaluation Loss", linewidth=2)
    if model == "baseline":
        plt.plot(*np.array(train_losses).T, label="Batch Train Loss", alpha=1)
    else:
        plt.plot(*np.array(train_losses).T, label="Batch Train Loss", alpha=0.2)
        plt.plot(*moving_avg(*np.array(train_losses).T, neighbors=100), label="Smooth Train Loss")
    plt.title(f"{name} Loss Trajectory")
    plt.xlabel("Step")
    plt.ylabel("LM Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(P / f"{model}-loss.pdf")
    plt.show()


# %%
for split, name in zip(("val", "test"), ("Validation", "Test")):
    data = Dataset.load_from_disk(
        Path("/home/sorenmulli/Nextcloud/cand4/framework/local-data") / split
    )["source"]
    category_counts = Counter(data)
    total_count = sum(category_counts.values())
    adjusted_counts = Counter()
    for category, count in category_counts.items():
        if (count / total_count) * 100 < 1:
            adjusted_counts["Other"] += count
        else:
            adjusted_counts[category] = count

    sorted_adjusted_counts = dict(
        sorted(adjusted_counts.items(), key=lambda item: (item[0] != "Other", item[1]))
    )
    colors = plt.cm.Set3(np.linspace(0, 1, len(adjusted_counts)))
    labels = [label.capitalize() for label in sorted_adjusted_counts]
    sizes = sorted_adjusted_counts.values()

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140, colors=colors)
    plt.axis("equal")
    plt.title(f"Data Sources (n={len(data)})")
    plt.tight_layout()
    plt.savefig(P / f"pretrain-{name}-source.pdf")
    plt.show()
