import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Optional

from nlgenda.evaluation.results import MetricResult, Scores, Scoring

logger = logging.getLogger(__name__)


class MetaScoring(ABC):
    @abstractmethod
    def meta_score(self, scores: Scores) -> list[tuple[str, str, list[MetricResult]]]:
        ...


class TimingScore(MetaScoring):
    def meta_score(self, scores: Scores) -> list[tuple[str, str, list[MetricResult]]]:
        out = []
        for scoring in scores.scorings:
            if (total_time := scoring.execution_metadata.total_inference_seconds) is not None:
                avg_time = total_time / len(scoring.metric_results[0].example_results)
                out.append(
                    (
                        scoring.execution_metadata.scenario_cfg["name"],
                        scoring.execution_metadata.model_cfg["name"],
                        [
                            MetricResult(
                                "Average run time",
                                "Total inference seconds divided by number of examples",
                                {},
                                aggregate=avg_time,
                                error=None,
                                higher_is_better=False,
                            )
                        ],
                    )
                )
        return out


class DisparityScoring(MetaScoring, ABC):
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
    def relevant_augmenter_keys(self) -> tuple[str, str]:
        ...

    def calculate_disparities(self, scorings: dict[str, Scoring]) -> list[MetricResult]:
        out = []
        first, other = self.relevant_augmenter_keys
        for result in scorings[first].metric_results:
            for other_result in scorings[other].metric_results:
                if result.short_name == other_result.short_name:
                    out.append(
                        MetricResult(
                            result.short_name,
                            f"{self.description}: {result.description}",
                            {},
                            aggregate=result.aggregate - other_result.aggregate,
                            error=None,
                            higher_is_better=True,
                        )
                    )
                    break
            else:
                raise ValueError("Disparity calculation not possible: Different metrics")
        return out

    def meta_score(self, scores: Scores) -> list[tuple[str, str, list[MetricResult]]]:
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
                logger.warning(
                    "Unfinished disparity score for scoring %s on %s did not have 1/2 augmenters",
                    scenario_name,
                    model_name,
                )
                continue
            out.append(
                (
                    f"{self.name}: {scenario_name}",
                    model_name,
                    self.calculate_disparities(scorings),
                )
            )
        return out


class KeyStrokeRobustness(DisparityScoring):
    @property
    def name(self) -> str:
        return "Keystroke robustness"

    @property
    def description(self) -> str:
        return "Metric value for normal version and one with 10% keystroke errors"

    @property
    def relevant_augmenter_keys(self) -> tuple[Optional[str], str]:
        return None, "keystroke-error"


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
    def relevant_augmenter_keys(self) -> tuple[str, str]:
        return "female-inserted", "male-inserted"


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
    def relevant_augmenter_keys(self) -> tuple[str, str]:
        return "muslim-inserted", "danish-inserted"


META_SCORERS = [
    TimingScore(),
    GenderNameScore(),
    NameOriginScore(),
]
