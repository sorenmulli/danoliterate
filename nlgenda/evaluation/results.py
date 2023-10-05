import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from nlgenda.evaluation.serialization import OutDictType, fix_args_for_dataclass
from nlgenda.infrastructure.constants import SCORES_ARTIFACT_TYPE
from nlgenda.infrastructure.logging import commit_hash
from nlgenda.infrastructure.timing import get_now_stamp

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        # Dict is mutable; does not guarantee type safety
        return cls(**self_dict)  # type: ignore


@dataclass
class ExecutionResultMetadata:
    timestamp: str
    id_: Optional[str]
    commit: Optional[str]

    scenario_cfg: OutDictType
    model_cfg: OutDictType
    evaluation_cfg: OutDictType

    sent_to_wandb: bool = False

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        fix_args_for_dataclass(cls, self_dict)
        return cls(**self_dict)  # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionResultMetadata):
            return NotImplemented
        return (self.id_ or self.timestamp) == (other.id_ or other.timestamp)


@dataclass
class ExecutionResult:
    name: str
    local_path: str
    metadata: ExecutionResultMetadata

    examples: list[ExecutionExample] = field(default_factory=list)

    def save_locally(self, path: Optional[str] = None) -> str:
        path = path or self.local_path
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file)
        return path

    def to_dict(self) -> OutDictType:
        self_dict = asdict(self)
        self_dict["metadata"] = self.metadata.to_dict()
        self_dict["examples"] = [example.to_dict() for example in self.examples]
        return self_dict

    @classmethod
    def from_config(cls, cfg: DictConfig):
        metadata_fields = get_reproducability_metadata_fields()
        autoname = ".".join(
            name_part.replace(".", "").replace(" ", "-").lower()
            for name_part in (cfg.model.name, cfg.scenario.name, str(metadata_fields["timestamp"]))
        )

        results_path = cfg.evaluation.local_results
        if not os.path.isdir(results_path):
            logger.warning("Creating new local directory for results: %s", results_path)
            os.makedirs(results_path)
        out_path = os.path.join(results_path, f"{autoname}.json")

        return cls(
            name=autoname,
            local_path=out_path,
            metadata=ExecutionResultMetadata(
                **metadata_fields,  # type: ignore
                scenario_cfg=conf_to_dict(cfg.scenario),
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

    example_results: dict[str, float | tuple[float, ...]]
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
    id_: Optional[str]
    commit: Optional[str]

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
        metadata_dict: OutDictType = self_dict.pop("execution_metadata")  # type: ignore
        metadata = ExecutionResultMetadata.from_dict(metadata_dict)

        result_dicts: list[OutDictType] = self_dict.pop("metric_results")  # type: ignore
        results = [MetricResult.from_dict(result_dict) for result_dict in result_dicts]
        return cls(execution_metadata=metadata, metric_results=results, **self_dict)  # type: ignore


@dataclass
class Scores:
    scorings: list[Scoring]
    local_path: str

    debug: bool

    sent_to_wandb = False
    name = SCORES_ARTIFACT_TYPE

    @classmethod
    def from_config(cls, cfg: DictConfig):
        autoname = f"{cls.name}-{get_now_stamp()}"
        results_path = cfg.evaluation.local_results
        if not os.path.isdir(results_path):
            logger.warning("Creating new local directory for results: %s", results_path)
            os.makedirs(results_path)
        out_path = os.path.join(results_path, f"{autoname}.json")
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

    def save_locally(self, path: Optional[str] = None) -> str:
        path = path or self.local_path
        with open(path, "w", encoding="utf-8") as file:
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
    }
