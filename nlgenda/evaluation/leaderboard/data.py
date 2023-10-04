import wandb

from nlgenda.evaluation.results import ExecutionResult
from nlgenda.evaluation.serialization import EXECUTION_RESULT_ARTIFACT_TYPE


def get_results_wandb(wandb_project: str, wandb_entity: str, include_debug=False):
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    results = []
    for collection in api.artifact_type(
        type_name=EXECUTION_RESULT_ARTIFACT_TYPE, project=wandb_project
    ).collections():
        artifacts = list(collection.versions())
        assert len(artifacts) == 1
        artifact = artifacts[0]
        if not include_debug and artifact.metadata["evaluation_cfg"].get("debug"):
            continue
        results.append(ExecutionResult.from_wandb(artifact))
    return results
