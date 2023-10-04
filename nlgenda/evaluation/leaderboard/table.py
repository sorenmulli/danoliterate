import pandas as pd

from nlgenda.evaluation.results import EvaluationResult


def build_leaderboard_table(results: list[EvaluationResult]) -> pd.DataFrame:
    df = pd.DataFrame()
    for result in results:
        model_name = result.metadata.model_cfg["name"]
        scenario_name = result.metadata.scenario_cfg["name"]
        score = result.get_score()

        if model_name not in df.index:
            assert isinstance(model_name, str)
            df.loc[model_name, :] = [None] * len(df.columns)
        if scenario_name not in df.columns:
            df[scenario_name] = [None] * len(df)
        df.at[model_name, scenario_name] = score
    return df
