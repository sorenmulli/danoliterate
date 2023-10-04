import pandas as pd

from nlgenda.evaluation.results import Scores


def build_leaderboard_table(scores: Scores) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.DataFrame()
    helps = {}
    for scoring in scores.scorings:
        model_name = str(scoring.execution_metadata.model_cfg["name"])
        scenario_name = str(scoring.execution_metadata.scenario_cfg["name"])

        for metric in scoring.metric_results:
            scenario_name += f": {metric.short_name}"

            if model_name not in df.index:
                assert isinstance(model_name, str)
                df.loc[model_name, :] = [None] * len(df.columns)
            if scenario_name not in df.columns:
                df[scenario_name] = [None] * len(df)

            df.at[model_name, scenario_name] = f"{metric.mean * 100:.0f}" + (
                "" if metric.error is None else f" ± {metric.error * 100:.0f}"
            )
            helps[scenario_name] = metric.description
    return df, helps
