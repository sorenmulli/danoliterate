import pandas as pd

from danoliterate.evaluation.results import MetricResult

INDEX_TITLE = "🏆Average Index"


def _space(val: str, spacing=5) -> str:
    return " " * (spacing - len(val)) + val


def build_leaderboard_table(
    metric_structure: dict[str, dict[str, list[MetricResult]]],
    chosen_metrics: dict[str, dict[str, MetricResult]],
    efficiency=True,
) -> pd.DataFrame:
    df = pd.DataFrame()
    metric_df = pd.DataFrame()
    lower_is_better = set()
    for scenario_name, models in metric_structure.items():
        for model_name in models:
            metric = chosen_metrics[scenario_name][model_name]

            if model_name not in df.index:
                assert isinstance(model_name, str)
                df.loc[model_name, :] = [None] * len(df.columns)
            if scenario_name not in df.columns:
                df[scenario_name] = [None] * len(df)

            agg = _space(
                str(round(metric.aggregate)) if efficiency else f"{round(metric.aggregate * 100)}"
            )
            err = f"± {metric.error * (1 if efficiency else  100):.0f}" if metric.error else ""

            df.at[model_name, scenario_name] = f"{agg}{err}"
            metric_df.at[model_name, scenario_name] = metric.aggregate

            if not metric.higher_is_better:
                lower_is_better.add(scenario_name)

    index_scores_df = metric_df.apply(
        lambda col: (
            1 - (col - col.min()) / (col.max() - col.min())
            if col.name in lower_is_better
            else (col - col.min()) / (col.max() - col.min())
        )
    )
    df[INDEX_TITLE] = [_space(str(round(score * 100))) for score in index_scores_df.mean(axis=1)]
    df = df[[INDEX_TITLE, *[col for col in df.columns if col != INDEX_TITLE]]]
    df = df.sort_values(INDEX_TITLE, ascending=False)
    return df
