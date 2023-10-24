import json
import os
import re
from tempfile import TemporaryDirectory
from typing import Generator, Optional

import wandb
from omegaconf import DictConfig
from tqdm import tqdm
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from nlgenda.evaluation.results import ExecutionResult, Scores
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.infrastructure.constants import EXECUTION_RESULT_ARTIFACT_TYPE, SCORES_ARTIFACT_TYPE


def dict_from_artifact(artifact) -> OutDictType:
    with TemporaryDirectory() as temp_dir:
        json_path = artifact.file(temp_dir)
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)


def setup_short_run(name: str, job_type: str, wandb_cfg: DictConfig) -> Optional[Run | RunDisabled]:
    return (
        wandb.init(
            name=name,
            job_type=job_type,
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
        )
        if wandb_cfg.enabled
        else None
    )


def _clean_artifact_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_\.]", "", name)


def send_result_wandb(result: ExecutionResult, run: Run | RunDisabled):
    result.metadata.sent_to_wandb = True
    artifact = wandb.Artifact(
        name=_clean_artifact_name(result.name),
        type=EXECUTION_RESULT_ARTIFACT_TYPE,
        metadata=result.metadata.to_dict(),
    )
    with TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "result.json")
        result.save_locally(temp_path)
        artifact.add_file(local_path=temp_path, name="result.json")
    run.log_artifact(artifact)


def send_scores_wandb(scores: Scores, run):
    scores.sent_to_wandb = True
    artifact = wandb.Artifact(name=SCORES_ARTIFACT_TYPE, type=SCORES_ARTIFACT_TYPE)
    with TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "scores.json")
        scores.save_locally(temp_path)
        artifact.add_file(local_path=temp_path, name="scores.json")
    run.log_artifact(artifact)


def yield_wandb_artifacts(
    wandb_project: str, wandb_entity: str, include_debug=False
) -> Generator[wandb.Artifact, None, None]:
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    for collection in tqdm(
        api.artifact_type(
            type_name=EXECUTION_RESULT_ARTIFACT_TYPE, project=wandb_project
        ).collections(),
        desc="Fetching artifact collections",
    ):
        artifacts = list(collection.versions())
        # TODO: Also delete collection when deleting artifacts
        if not artifacts:
            # Artifact was deleted
            continue
        assert len(artifacts) == 1
        artifact = artifacts[0]
        if not include_debug and artifact.metadata["evaluation_cfg"].get("debug"):
            continue
        yield artifact


def get_results_wandb(
    wandb_project: str, wandb_entity: str, include_debug=False
) -> list[ExecutionResult]:
    results = []
    for artifact in yield_wandb_artifacts(wandb_project, wandb_entity, include_debug):
        result_dict = dict_from_artifact(artifact)
        results.append(ExecutionResult.from_dict(result_dict))
    return results


def get_scores_wandb(wandb_project: str, wandb_entity: str, include_debug=False) -> Scores:
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    collections = api.artifact_type(
        type_name=SCORES_ARTIFACT_TYPE, project=wandb_project
    ).collections()
    assert len(collections) == 1
    collection = collections[0]
    # TODO: Make sure that we iterate right order
    for artifact in collection.versions():
        scores_dict = dict_from_artifact(artifact)
        scores = Scores.from_dict(scores_dict)
        if not include_debug and scores.debug:
            continue
        return scores
    raise ValueError("No scores artifacts found")
