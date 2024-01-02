from typing import Optional

import pandas as pd

from danoliterate.evaluation.results import MetricResult


class Analyser:
    concat_key = "Concatenated"

    def __init__(self, results: dict[str, dict[str, MetricResult]]):
        self.results = results

        self.df = self._get_df()
        self.options_dict = self._compute_options()

    def _get_df(self):
        data = []
        for scenario, models in self.results.items():
            for model, metric_result in models.items():
                for example_id, result in metric_result.example_results.items():
                    data.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "metric": metric_result.short_name,
                            "score": result
                            if isinstance(result, float)
                            else float(result[0] == result[1]),
                            "example_id": example_id,
                        }
                    )
        return pd.DataFrame(data)

    def _compute_options(self):
        return {option: sorted(self.df[option].unique()) for option in ("model", "scenario")}

    def get_subset(
        self,
        model: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        if sum(kwarg is None for kwarg in (model, scenario)) > 1:
            raise ValueError
        conditions = []
        multi_queries = None
        for given_val, col in zip((model, scenario), ("model", "scenario")):
            if given_val is None:
                multi_queries = {val: f"{col} == '{val}'" for val in self.options_dict[col]}
            elif given_val != self.concat_key:
                conditions.append(f"{col} == '{given_val}'")
        if multi_queries is None:
            return {"Chosen combination": self.df.query(" & ".join(conditions))}
        all_dfs = {
            val: self.df.query(" & ".join([*conditions, extra_condition]))
            for val, extra_condition in multi_queries.items()
        }
        return {val: df for val, df in all_dfs.items() if not df.empty}
