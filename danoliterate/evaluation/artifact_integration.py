import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import wandb
from omegaconf import DictConfig
from tqdm import tqdm
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from danoliterate.evaluation.results import ExecutionResult, Scores
from danoliterate.evaluation.serialization import OutDictType
from danoliterate.infrastructure.constants import (
    EXECUTION_RESULT_ARTIFACT_TYPE,
    SCORES_ARTIFACT_TYPE,
)
from danoliterate.infrastructure.logging import logger

CACHE_DURATION = timedelta(days=7)


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
        temp_path = Path(temp_dir) / "result.json"
        result.save_locally(temp_path)
        artifact.add_file(local_path=str(temp_path), name="result.json")
    run.log_artifact(artifact)


def send_scores_wandb(scores: Scores, run):
    scores.sent_to_wandb = True
    artifact = wandb.Artifact(name=SCORES_ARTIFACT_TYPE, type=SCORES_ARTIFACT_TYPE)
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "scores.json"
        scores.save_locally(temp_path)
        artifact.add_file(local_path=str(temp_path), name="scores.json")
    run.log_artifact(artifact)


def fetch_artifact_data(collection):
    artifacts = list(collection.versions())
    if not artifacts:
        # Artifact was deleted
        return []
    if len(artifacts) > 1:
        logger.warning("More than 1 artifact for collection %s", collection)
    return artifacts


def yield_wandb_artifacts(wandb_project: str, wandb_entity: str, include_debug=False):
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    collections = api.artifact_type(
        type_name=EXECUTION_RESULT_ARTIFACT_TYPE, project=wandb_project
    ).collections()

    for collection in tqdm(collections):
        for artifact in fetch_artifact_data(collection):
            if (artifact is not None) and not (
                artifact.metadata["evaluation_cfg"].get("debug") and not include_debug
            ):
                yield artifact


def is_cache_valid(cache_file) -> bool:
    if not os.path.exists(cache_file):
        return False
    last_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
    return datetime.now() - last_modified < CACHE_DURATION


def get_cached_results(cache_file: str) -> list[ExecutionResult]:
    with open(cache_file, "r", encoding="utf-8") as file:
        return [ExecutionResult.from_dict(result) for result in json.load(file)]


def cache_results(results: list[ExecutionResult], cache_file: str):
    Path(cache_file).parent.mkdir(exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump([result.to_dict() for result in results], file)


def get_results_wandb(
    wandb_project: str,
    wandb_entity: str,
    cache_file: str,
    cache_update=False,
    include_debug=False,
) -> list[ExecutionResult]:
    if is_cache_valid(cache_file) and not cache_update:
        return get_cached_results(cache_file)
    results = []
    for artifact in yield_wandb_artifacts(wandb_project, wandb_entity, include_debug):
        result_dict = dict_from_artifact(artifact)
        results.append(ExecutionResult.from_dict(result_dict))
    cache_results(results, cache_file)
    return results


def get_scores_wandb(wandb_project: str, wandb_entity: str, include_debug=False) -> Scores:
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    collections = api.artifact_type(
        type_name=SCORES_ARTIFACT_TYPE, project=wandb_project
    ).collections()
    assert len(collections) == 1
    collection = collections[0]
    for artifact in collection.versions():
        scores_dict = dict_from_artifact(artifact)
        scores = Scores.from_dict(scores_dict)
        if not include_debug and scores.debug:
            continue
        return scores
    raise ValueError("No scores artifacts found")
