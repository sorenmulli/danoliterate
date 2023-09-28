# TODO: Rename this file
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Union

from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)

OutDictType = dict[str, Union[str, int, float, bool, None, "OutDictType", list["OutDictType"]]]


# pylint: disable=too-many-instance-attributes
@dataclass
class EvaluationExample:
    prompt: str
    id_: str

    target_answer: Optional[str] = None
    target_answer_model_score: Optional[float] = None

    index_label: Optional[int] = None
    options: Optional[list[str]] = None

    generated_text: Optional[str] = None
    index_prediction: Optional[int] = None

    options_model_scores: Optional[list[float]] = None

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: Mapping):
        return cls(**self_dict)


@dataclass
class EvaluationResultMetadata:
    timestamp: str

    scenario_cfg: OutDictType
    model_cfg: OutDictType
    evaluation_cfg: OutDictType

    sent_to_wandb: bool = False

    def to_dict(self) -> OutDictType:
        return asdict(self)

    @classmethod
    def from_dict(cls, self_dict: Mapping):
        return cls(**self_dict)


@dataclass
class EvaluationResult:
    name: str
    local_path: str
    metadata: EvaluationResultMetadata

    examples: list[EvaluationExample] = field(default_factory=list)

    # TODO: Refactor this, move it to analysis and have a structured scorer
    def get_score(self) -> float:
        example_scores = []
        for example in self.examples:
            if (score := example.target_answer_model_score) is not None:
                example_scores.append(score)
            elif example.index_label is not None and example.index_prediction is not None:
                example_scores.append(example.index_label == example.index_prediction)
            else:
                raise ValueError
        return mean(example_scores)

    def send_to_wandb(self, run) -> bool:
        self.metadata.sent_to_wandb = True

        artifact = wandb.Artifact(
            name=self.name, type="evaluation_result", metadata=self.metadata.to_dict()
        )
        with TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "result.json")
            self.save_locally(temp_path)
            artifact.add_file(local_path=temp_path, name="result.json")
        run.log_artifact(artifact)
        return self.metadata.sent_to_wandb

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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        autoname = ".".join(
            name_part.replace(".", "").replace(" ", "-").lower()
            for name_part in (cfg.model.name, cfg.scenario.name, timestamp)
        )

        results_path = cfg.evaluation.local_results
        if not os.path.isdir(results_path):
            logger.warning("Creating new local directory for results: %s", results_path)
            os.makedirs(results_path)
        out_path = os.path.join(results_path, f"{autoname}.json")

        return cls(
            name=autoname,
            local_path=out_path,
            metadata=EvaluationResultMetadata(
                timestamp=timestamp,
                scenario_cfg=conf_to_dict(cfg.scenario),
                model_cfg=conf_to_dict(cfg.model),
                evaluation_cfg=conf_to_dict(cfg.evaluation),
            ),
        )

    @classmethod
    def from_dict(cls, self_dict: OutDictType):
        metadata = EvaluationResultMetadata.from_dict(self_dict.pop("metadata"))  # type: ignore

        assert isinstance(self_dict["examples"], list)
        if self_dict["examples"] and isinstance(self_dict["examples"][0], list):
            logger.warning("Fixing erroneuous extra list in examples field!")
            self_dict["examples"] = self_dict["examples"][0]

        examples = [
            EvaluationExample.from_dict(example_dict)  # type: ignore
            for example_dict in self_dict.pop("examples")  # type: ignore
        ]
        return cls(metadata=metadata, examples=examples, **self_dict)  # type: ignore

    @classmethod
    def from_wandb(cls, artifact):
        with TemporaryDirectory() as temp_dir:
            json_path = artifact.file(temp_dir)
            with open(json_path, "r", encoding="utf-8") as file:
                self_dict = json.load(file)
        return cls.from_dict(self_dict)


def conf_to_dict(cfg: DictConfig) -> OutDictType:
    return OmegaConf.to_container(cfg)  # type: ignore
