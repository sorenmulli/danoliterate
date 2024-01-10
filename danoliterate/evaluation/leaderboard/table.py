from math import isnan

import numpy as np
import pandas as pd

from danoliterate.evaluation.results import MetricResult

WIN_EMOJI = "🏆"
INDEX_TITLE = WIN_EMOJI + "Avg. Index"


def _space(val: str, spacing=5) -> str:
    return " " * (spacing - len(val)) + val


def get_table_values(
    chosen_metrics: dict[str, dict[str, MetricResult]], show_missing=False
) -> pd.DataFrame:
    df = pd.DataFrame()
    for scenario_name, models in chosen_metrics.items():
        for model_name, metric in models.items():
            if model_name not in df.index:
                assert isinstance(model_name, str)
                df.loc[model_name, :] = [float("nan")] * len(df.columns)
            if scenario_name not in df.columns:
                df[scenario_name] = [float("nan")] * len(df)
            df.at[model_name, scenario_name] = metric.aggregate
    if not show_missing:
        df = df.dropna()
    return df


def _format_err(num):
    if round(num) >= 1:
        return f"{num:.0f}"
    else:
        num_str = str(num)
        if "." in num_str:
            decimal_part = num_str.split(".")[1]
            for i, digit in enumerate(decimal_part):
                if digit != "0":
                    return f"{num:.{i+1}f}"
        return str(num)


def build_leaderboard_table(
    chosen_metrics: dict[str, dict[str, MetricResult]],
    efficiency=False,
    micro=True,
    show_missing=False,
    reverse_abso_sort=False,
) -> tuple[pd.DataFrame, set[str]]:
    df = get_table_values(chosen_metrics, show_missing)
    metric_df = pd.DataFrame()
    examples = {}
    lower_is_better = set()
    for scenario_name, models in chosen_metrics.items():
        for model_name, metric in models.items():
            if model_name not in df.index:
                continue
            agg = _space(
                f"{metric.aggregate:.1f}" if efficiency else f"{round(metric.aggregate * 100)}"
            )
            err = (
                f"± {_format_err(metric.error * (1 if efficiency else 100))}"
                if metric.error
                else ""
            )

            metric_df.at[model_name, scenario_name] = metric.aggregate

            df.at[model_name, scenario_name] = f"{agg}{err}"
            examples[scenario_name] = len(metric.example_results) or 1

            if not metric.higher_is_better:
                lower_is_better.add(scenario_name)

    if reverse_abso_sort:
        metric_df = -metric_df.abs()
    index_scores_df = metric_df.apply(
        lambda col: (
            1 - (col - col.min()) / (col.max() - col.min())
            if col.name in lower_is_better and not reverse_abso_sort
            else (col - col.min()) / (col.max() - col.min())
        )
    )
    weights = np.array(list(examples.values()))
    index_means = (
        index_scores_df.apply(
            lambda x: np.ma.average(
                np.ma.MaskedArray(x, mask=np.isnan(x)),
                weights=weights,
            ),
            axis=1,
        )
        if micro
        else index_scores_df.mean(axis=1)
    )
    df[INDEX_TITLE] = [_space(str(round(score * 100))) for score in index_means]
    df = df[[INDEX_TITLE, *[col for col in df.columns if col != INDEX_TITLE]]]
    df = df.sort_values(INDEX_TITLE, ascending=False)
    return df, lower_is_better


def format_table_for_latex(table: pd.DataFrame, lower_is_better: set[str], abso_num=False) -> str:
    df = table.copy()
    for col in df.columns:
        # Extracting numbers and uncertainties.
        nums = df[col].str.extract(r"(\d+)(?:± (\d+))?")

        nums[0] = nums[0].astype(float)
        if abso_num:
            nums[0] = nums[0].abs()
        top1, top2, top3 = (nums[0].nsmallest if col in lower_is_better else nums[0].nlargest)(
            3
        ).index

        for idx, num in df[col].items():
            if not isinstance(num, str) and isnan(num):
                formatted_number = "--"
            else:
                formatted_number = "$" + num.strip().replace("±", r"\pm") + "$"
                if idx == top1:
                    formatted_number = (
                        r"\underline{\underline{\underline{" + formatted_number + "}}}"
                    )
                elif idx == top2:
                    formatted_number = r"\underline{\underline{" + formatted_number + "}}"
                elif idx == top3:
                    formatted_number = r"\underline{" + formatted_number + "}"
            df.at[idx, col] = formatted_number
    df.columns = df.columns.str.replace("#", r"\#")
    df.columns = df.columns.str.replace(WIN_EMOJI, "")
    return df.style.to_latex()
