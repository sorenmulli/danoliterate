from pathlib import Path

_THIS_FILE_PATH = Path(__file__).resolve()

CONFIG_DIR = str((_THIS_FILE_PATH.parent.parent / "configs").resolve())

REPO_PATH = _THIS_FILE_PATH.parent.parent.parent
EXECUTION_RESULT_NAME = "evaluation_result"
SCORES_NAME = "scores"
