import pandas as pd

from nlgenda.evaluation.results import MetricResult


def build_leaderboard_table(
    metric_structure: dict[str, dict[str, list[MetricResult]]],
    chosen_metrics: dict[str, dict[str, MetricResult]],
    efficiency=True,
) -> pd.DataFrame:
    df = pd.DataFrame()
    metric_df = pd.DataFrame()
    for scenario_name, models in metric_structure.items():
        for model_name in models:
            metric = chosen_metrics[scenario_name][model_name]

            if model_name not in df.index:
                assert isinstance(model_name, str)
                df.loc[model_name, :] = [None] * len(df.columns)
            if scenario_name not in df.columns:
                df[scenario_name] = [None] * len(df)

            agg = (
                f"{round(metric.aggregate):05d}"
                if efficiency
                else f"{round(metric.aggregate * 100):02d}"
            )
            err = f"± {metric.error * (1 if efficiency else  100):.0f}" if metric.error else ""

            df.at[model_name, scenario_name] = f"{agg}{err}"
            metric_df.at[model_name, scenario_name] = metric.aggregate
    index_scores_df = metric_df.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
    index_title = "🏆Overall Index"
    df[index_title] = [f"{round(score*100):02d}" for score in index_scores_df.mean(axis=1)]
    df = df[[index_title, *[col for col in df.columns if col != index_title]]]
    return df
