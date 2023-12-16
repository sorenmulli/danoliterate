from pathlib import Path

# TODO: Handle more robustly
CONFIG_DIR = "../configs"

_THIS_FILE_PATH = Path(__file__).resolve()

REPO_PATH = _THIS_FILE_PATH.parent.parent.parent
EXECUTION_RESULT_ARTIFACT_TYPE = "evaluation_result"
SCORES_ARTIFACT_TYPE = "scores"
