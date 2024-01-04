import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from danoliterate.evaluation.serialization import (
    OutDictType,
    apply_backcomp_fixes_execution_example,
    apply_backcomp_fixes_execution_result_metadata,
    apply_backcomp_reordering_metric_results,
    fix_args_for_dataclass,
)
from danoliterate.infrastructure.constants import SCORES_ARTIFACT_TYPE
from danoliterate.infrastructure.logging import commit_hash, get_compute_unit_string, logger
from danoliterate.infrastructure.timing import get_now_stamp


# pylint: disable=too-many-instance-attributes
@dataclass
class ExecutionExample:
    prompt: str
    id_: str

    target_answer: Optional[str] = None
    index_label: Optional[int] = None
    options: Optional[list[str]] = None
    generated_text: Optional[str] = None
    options_model_likelihoods: Optional[list[float]] = None
    few_shot_example_ids: Optional[list[str]] = None
    generated_score: Optional[float] = None

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        apply_backcomp_fixes_execution_example(self_dict)
        # Dict is mutable; does not guarantee type safety
        return cls(**self_dict)  # type: ignore


@dataclass
class ExecutionResultMetadata:
    timestamp: str
    id_: str
    commit: str
    compute_unit: Optional[str]

    scenario_cfg: OutDictType
    model_cfg: OutDictType
    evaluation_cfg: OutDictType

    augmenter_key: Optional[str] = None
    total_inference_seconds: Optional[float] = None
    sent_to_wandb: bool = False

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        apply_backcomp_fixes_execution_result_metadata(self_dict)
        return cls(**self_dict)  # type: ignore


@dataclass
class ExecutionResult:
    name: str
    local_path: str | os.PathLike
    metadata: ExecutionResultMetadata

    examples: list[ExecutionExample] = field(default_factory=list)

    def save_locally(self, path: Optional[str | os.PathLike] = None) -> str | os.PathLike:
        path = path or self.local_path
        with open(path, "w", encoding="utf-8") as file:
            self.local_path = str(self.local_path)
            json.dump(self.to_dict(), file)
        return path

    def to_dict(self) -> OutDictType:
        self_dict = asdict(self)
        self_dict["metadata"] = self.metadata.to_dict()
        self_dict["examples"] = [example.to_dict() for example in self.examples]
        return self_dict

    @classmethod
    def from_config(cls, cfg: DictConfig, scenario_cfg: DictConfig, augmenter):
        metadata_fields = get_reproducability_metadata_fields()
        name_parts = [cfg.model.name, scenario_cfg.name, str(metadata_fields["timestamp"])]
        if augmenter is not None:
            name_parts.append(augmenter.key)
            metadata_fields["augmenter_key"] = augmenter.key
        autoname = ".".join(
            name_part.replace(".", "").replace(" ", "-").lower() for name_part in name_parts
        )

        results_path = Path(cfg.evaluation.local_results)
        if not results_path.is_dir():
            logger.warning("Creating new local directory for results: %s", results_path)
            os.makedirs(results_path)
        out_path = results_path / f"{autoname}.json"

        return cls(
            name=autoname,
            local_path=out_path,
            metadata=ExecutionResultMetadata(
                **metadata_fields,  # type: ignore
                scenario_cfg=conf_to_dict(scenario_cfg),
                model_cfg=conf_to_dict(cfg.model),
                evaluation_cfg=conf_to_dict(cfg.evaluation),
            ),
        )

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        metadata_dict: OutDictType = self_dict.pop("metadata")  # type: ignore
        metadata = ExecutionResultMetadata.from_dict(metadata_dict)

        assert isinstance(self_dict["examples"], list)
        if self_dict["examples"] and isinstance(self_dict["examples"][0], list):
            logger.warning("Fixing erroneuous extra list in examples field!")
            self_dict["examples"] = self_dict["examples"][0]

        example_dicts: list[OutDictType] = self_dict.pop("examples")  # type: ignore
        examples = [ExecutionExample.from_dict(example_dict) for example_dict in example_dicts]
        return cls(metadata=metadata, examples=examples, **self_dict)  # type: ignore


@dataclass
class MetricResult:
    short_name: str
    description: str

    # Maps single example ID to either a float describing a score in [0, 1]
    # or two floats where the first float is the index of the correct option
    # and the second is the index of the option that the model guessed. Thus,
    # it can be reduced to accuracy by doing float(res[0] == res[1])
    example_results: dict[str, float | tuple[float, float]]
    aggregate: float
    error: Optional[float]

    higher_is_better: bool

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        return cls(**self_dict)  # type: ignore


@dataclass
class Scoring:
    timestamp: str
    id_: str
    commit: Optional[str]
    compute_unit: Optional[str]

    execution_metadata: ExecutionResultMetadata

    metric_results: list[MetricResult]

    @classmethod
    def from_execution_metadata(cls, metadata: ExecutionResultMetadata):
        return cls(
            **get_reproducability_metadata_fields(),  # type: ignore
            execution_metadata=metadata,
            metric_results=[],
        )

    def to_dict(self) -> OutDictType:
        self_dict = asdict(self)
        self_dict["execution_metadata"] = self.execution_metadata.to_dict()
        self_dict["metric_results"] = [result.to_dict() for result in self.metric_results]
        return self_dict

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        metadata_dict: OutDictType = self_dict.pop("execution_metadata")  # type: ignore
        metadata = ExecutionResultMetadata.from_dict(metadata_dict)

        result_dicts: list[OutDictType] = apply_backcomp_reordering_metric_results(
            self_dict.pop("metric_results")  # type: ignore
        )

        results = [MetricResult.from_dict(result_dict) for result_dict in result_dicts]
        return cls(execution_metadata=metadata, metric_results=results, **self_dict)  # type: ignore


@dataclass
class Scores:
    scorings: list[Scoring]
    local_path: str | os.PathLike

    debug: bool

    sent_to_wandb = False
    name = SCORES_ARTIFACT_TYPE

    @classmethod
    def from_config(cls, cfg: DictConfig):
        autoname = f"{cls.name}-{get_now_stamp()}"
        results_path = Path(cfg.evaluation.local_results)
        if not results_path.is_dir():
            logger.warning("Creating new local directory for results: %s", results_path)
            results_path.mkdir()
            os.makedirs(results_path)
        out_path = results_path / f"{autoname}.json"
        return cls(
            local_path=out_path,
            scorings=[],
            debug=cfg.evaluation.debug,
        )

    def to_dict(self) -> OutDictType:
        self_dict = asdict(self)
        self_dict["scorings"] = [scoring.to_dict() for scoring in self.scorings]
        return self_dict

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        scoring_dicts: list[OutDictType] = self_dict.pop("scorings")  # type: ignore
        scorings = [Scoring.from_dict(scoring_dict) for scoring_dict in scoring_dicts]
        return cls(scorings=scorings, **self_dict)  # type: ignore

    def save_locally(self, given_path: Optional[str | os.PathLike] = None) -> str | os.PathLike:
        path = given_path or self.local_path
        with open(path, "w", encoding="utf-8") as file:
            self.local_path = str(self.local_path)
            json.dump(self.to_dict(), file)
        return path


def conf_to_dict(cfg: DictConfig) -> OutDictType:
    return OmegaConf.to_container(cfg)  # type: ignore


def get_reproducability_metadata_fields() -> dict[str, Optional[str]]:
    return {
        "timestamp": get_now_stamp(),
        # Hash time + machine
        "id_": str(uuid.uuid1()),
        # If we are in the repo, we document current commit
        "commit": commit_hash(),
        # Save the compute used
        "compute_unit": get_compute_unit_string(),
    }
