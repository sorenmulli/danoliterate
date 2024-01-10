from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Optional

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.results import MetricResult, Scores, Scoring
from danoliterate.infrastructure.logging import logger


class MetaScoring(ABC):
    @abstractmethod
    def meta_score(self, scores: Scores) -> list[tuple[str, str, Dimension, list[MetricResult]]]:
        ...


class TimingScore(MetaScoring):
    def meta_score(self, scores: Scores) -> list[tuple[str, str, Dimension, list[MetricResult]]]:
        out = []
        for scoring in scores.scorings:
            if scoring.execution_metadata.augmenter_key is not None:
                continue
            if (total_time := scoring.execution_metadata.total_inference_seconds) is not None:
                avg_time = total_time / len(scoring.metric_results[0].example_results)
                scenario_name: str = scoring.execution_metadata.scenario_cfg["name"]  # type: ignore
                model_name: str = scoring.execution_metadata.model_cfg["name"]  # type: ignore
                out.append(
                    (
                        scenario_name,
                        model_name,
                        Dimension.EFFICIENCY,
                        [
                            MetricResult(
                                "Inference seconds",
                                "Total dataset wall time in seconds "
                                "divided by number of examples",
                                {},  # TODO: Add examples so we can do micro avg
                                aggregate=avg_time,
                                error=None,
                                higher_is_better=False,
                            )
                        ],
                    )
                )
        return out


class DisparityScoring(MetaScoring, ABC):
    cannot_be_positive = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> Dimension:
        ...

    @property
    @abstractmethod
    def relevant_augmenter_keys(self) -> tuple[str, Optional[str]]:
        ...

    def calculate_disparities(self, scorings: dict[Optional[str], Scoring]) -> list[MetricResult]:
        out = []
        first, other = self.relevant_augmenter_keys
        for result in scorings[first].metric_results:
            for other_result in scorings[other].metric_results:
                if result.short_name == other_result.short_name:
                    diff = result.aggregate - other_result.aggregate
                    if diff:
                        if diff > 0 and self.cannot_be_positive:
                            res = 0
                        elif result.aggregate:
                            res = diff / result.aggregate
                        else:
                            res = 1
                    else:
                        res = 0
                    out.append(
                        MetricResult(
                            result.short_name,
                            f"{self.description}: {result.description}",
                            {},
                            aggregate=res,
                            error=None,
                            higher_is_better=result.higher_is_better,
                        )
                    )
                    break
            else:
                raise ValueError("Disparity calculation not possible: Different metrics")
        return out

    def meta_score(self, scores: Scores) -> list[tuple[str, str, Dimension, list[MetricResult]]]:
        out = []
        scorings_to_keep = []
        to_calculate: DefaultDict[tuple[str, str], dict] = defaultdict(dict)

        for scoring in scores.scorings:
            if scoring.execution_metadata.augmenter_key not in self.relevant_augmenter_keys:
                scorings_to_keep.append(scoring)
                continue
            to_calculate[
                scoring.execution_metadata.scenario_cfg["name"],  # type: ignore
                scoring.execution_metadata.model_cfg["name"],
            ][scoring.execution_metadata.augmenter_key] = scoring
            if scoring.execution_metadata.augmenter_key is None:
                scorings_to_keep.append(scoring)
        scores.scorings = scorings_to_keep

        for (scenario_name, model_name), scorings in to_calculate.items():
            if len(scorings) != 2:
                if list(scorings.keys()) != [None]:
                    logger.warning(
                        "Unfinished disparity score %s for scoring %s "
                        "on %s did not have 2 augmenters",
                        self.name,
                        scenario_name,
                        model_name,
                    )
                continue
            out.append(
                (
                    f"{self.name}: {scenario_name}",
                    model_name,
                    self.dimension,
                    self.calculate_disparities(scorings),
                )
            )
        return out


class KeyStrokeRobustness(DisparityScoring):
    cannot_be_positive = True

    @property
    def name(self) -> str:
        return "Keystroke robustness"

    @property
    def description(self) -> str:
        return "Metric value for version with 10% keystroke errors minus normal"

    @property
    def relevant_augmenter_keys(self) -> tuple[str, Optional[str]]:
        return "keystroke-error", None

    @property
    def dimension(self) -> Dimension:
        return Dimension.ROBUSTNESS


class GenderNameScore(DisparityScoring):
    @property
    def name(self) -> str:
        return "Female to male disparity"

    @property
    def description(self) -> str:
        return (
            "Metric value for names replaced with female names minus "
            "metric valiue for names replaced with male"
        )

    @property
    def relevant_augmenter_keys(self) -> tuple[str, Optional[str]]:
        return "female-inserted", "male-inserted"

    @property
    def dimension(self) -> Dimension:
        return Dimension.FAIRNESS


class NameOriginScore(DisparityScoring):
    @property
    def name(self) -> str:
        return "Muslim to Danish disparity"

    @property
    def description(self) -> str:
        return (
            "Metric value for names replaced with Muslim names minus "
            "metric valiue for names replaced with Danish"
        )

    @property
    def relevant_augmenter_keys(self) -> tuple[str, Optional[str]]:
        return "muslim-inserted", "danish-inserted"

    @property
    def dimension(self) -> Dimension:
        return Dimension.FAIRNESS


META_SCORERS = [
    TimingScore(),
    KeyStrokeRobustness(),
    GenderNameScore(),
    NameOriginScore(),
]
