import wandb
from nlgenda.evaluation.example import EvaluationResult


def get_results_wandb(wandb_project: str, wandb_entity: str, include_debug=False):
    wandb.login()
    api = wandb.Api(overrides={"entity": wandb_entity})
    results = []
    # TODO: Have type name as constant
    for collection in api.artifact_type(
        type_name="evaluation_result", project=wandb_project
    ).collections():
        artifacts = list(collection.versions())
        assert len(artifacts) == 1
        artifact = artifacts[0]
        if not include_debug and artifact.metadata["evaluation_cfg"].get("debug"):
            continue
        results.append(EvaluationResult.from_wandb(artifact))
    return results
