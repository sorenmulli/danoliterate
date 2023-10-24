import logging
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import DefaultDict

import hydra
from omegaconf import DictConfig
from simple_term_menu import TerminalMenu
from tqdm import tqdm
from wandb import Artifact

from nlgenda.evaluation.artifact_integration import dict_from_artifact, yield_wandb_artifacts
from nlgenda.evaluation.results import ExecutionResult
from nlgenda.infrastructure.timing import get_now_stamp

logger = logging.getLogger(__name__)

LOGO = """
░█▀█░█▀▄░▀█▀░▀█▀░█▀▀░█▀█░█▀▀░▀█▀░█▀█
░█▀█░█▀▄░░█░░░█░░█▀▀░█▀█░█░░░░█░░█░█
░▀░▀░▀░▀░░▀░░▀▀▀░▀░░░▀░▀░▀▀▀░░▀░░▀▀▀
"""


class CleanDebugOptions(Enum):
    DISPLAY_DEBUG = 0
    DELETE_ALL = 1
    CANCEL = 2


def clean_debug(cfg: DictConfig):
    logger.info("Fetching all results marked as debug ...")
    debug_artifacts = [
        result
        for result in yield_wandb_artifacts(cfg.wandb.project, cfg.wandb.entity, include_debug=True)
        if result.metadata["evaluation_cfg"].get("debug")
    ]

    logger.info("Got %i debug artifacts", len(debug_artifacts))

    while True:
        debug_index = TerminalMenu(
            ["Display debug artifacts", "Delete all debug artifacts", "Cancel"],
            title=LOGO + "Handle debug artifacts.",
            status_bar="Choose operation.",
        ).show()

        match CleanDebugOptions(debug_index):
            case CleanDebugOptions.DISPLAY_DEBUG:
                logger.info(
                    "Debug Artifacts:\n%s",
                    "\n".join(_format_result(res) for res in debug_artifacts),
                )
            case CleanDebugOptions.DELETE_ALL:
                backup_path = Path(cfg.evaluation.local_results) / f"backup-{get_now_stamp()}"
                backup_path.mkdir(parents=True, exist_ok=True)
                logger.info("Backupping debug artifacts to %s before deletion.", backup_path)
                for artifact in debug_artifacts:
                    result = ExecutionResult.from_dict(dict_from_artifact(artifact))
                    result.save_locally(backup_path / f"{result.name}.json")
                for artifact in tqdm(debug_artifacts, desc="Deleting debug artifacts"):
                    artifact.delete(delete_aliases=True)
                logger.info("Deleted %i debug artifacts.", len(debug_artifacts))
                return main_menu(cfg)
            case CleanDebugOptions.CANCEL:
                return main_menu(cfg)


class DupeMarkerOptions(Enum):
    PRINT_PAIRS = 0
    MARK_ALL = 1
    CANCEL = 2


def _format_result(result: Artifact) -> str:
    meta = result.metadata
    return (
        f"{meta['model_cfg']['name']} on {meta['scenario_cfg']['name']} "
        f"[{meta['timestamp']}] ({meta.get('id_', 'NO ID')})]"
    )


def mark_dupes(cfg: DictConfig):
    logger.info("Fetching all results ...")
    results = list(yield_wandb_artifacts(cfg.wandb.project, cfg.wandb.entity))
    logger.info("Got %i results", len(results))
    # Data structures to hold the newest and old duplicates
    newest_artifacts: dict[tuple[str, str], Artifact] = {}
    old_duplicates: DefaultDict[tuple[str, str], list[Artifact]] = defaultdict(list)

    # Dictionary to hold the most recent timestamp per unique pair
    latest_timestamps: dict[tuple[str, str], tuple[int, ...]] = {}

    for result in results:
        # Forming the key as a tuple of scenario_name and model_name
        key = (result.metadata["scenario_cfg"]["name"], result.metadata["model_cfg"]["name"])

        # Converting timestamp string to a comparable format (as a tuple of integers)
        timestamp_tuple = tuple(map(int, result.metadata["timestamp"].split("-")))

        # If this key is seen for the first time or the timestamp is newer than the previous
        if key not in latest_timestamps or timestamp_tuple > latest_timestamps[key]:
            # If the key was seen before, move the previous newest entry to old_duplicates
            if key in latest_timestamps:
                old_duplicates[key].append(newest_artifacts[key])

            # Update the newest entry and the latest timestamp for this key
            newest_artifacts[key] = result
            latest_timestamps[key] = timestamp_tuple
        else:
            # If the timestamp is not newer, add this entry to old_duplicates
            old_duplicates[key].append(result)

    newest_duplicates = {
        key: artifact for key, artifact in newest_artifacts.items() if key in old_duplicates
    }

    logger.info(
        "%i scenario/model pairs had duplicates. " + "%i old duplicates",
        len(newest_duplicates),
        sum(len(v) for v in old_duplicates.values()),
    )
    while True:
        dupe_index = TerminalMenu(
            ["Display pairs", "Mark all old as duplicates", "Cancel"],
            title=LOGO + "Handle duplicates.",
            status_bar="Choose operation.",
        ).show()
        match DupeMarkerOptions(dupe_index):
            case DupeMarkerOptions.PRINT_PAIRS:
                logger.info(
                    "Pairs:\n%s",
                    "\n".join(
                        f"Newest: {_format_result(newest)}\tOlder: "
                        + "\t".join(_format_result(res) for res in old_duplicates[key])
                        for key, newest in newest_duplicates.items()
                    ),
                )
            case DupeMarkerOptions.MARK_ALL:
                all_to_mark = [
                    artifact
                    for to_mark_artifacts in old_duplicates.values()
                    for artifact in to_mark_artifacts
                ]
                for artifact in tqdm(all_to_mark, desc="Marking old duplicates as debug"):
                    artifact.metadata["evaluation_cfg"]["debug"] = True
                    artifact.save(cfg.wandb.project)
                logger.info("Marked %i duplicates as debug, persisted with W&B.", len(all_to_mark))
                return main_menu(cfg)
            case DupeMarkerOptions.CANCEL:
                return main_menu(cfg)


class MainMenuOptions(Enum):
    MARK_DUPES = 0
    CLEAN_DEBUG = 1
    EXIT = 2


def main_menu(cfg: DictConfig):
    main_index = TerminalMenu(
        ["Handle duplicate results", "Delete results marked as debug", "Quit tool"],
        title=LOGO + "Welcome to the Artifact Database tool.",
        status_bar="Choose operation.",
    ).show()

    match MainMenuOptions(main_index):
        case MainMenuOptions.MARK_DUPES:
            mark_dupes(cfg)
        case MainMenuOptions.CLEAN_DEBUG:
            clean_debug(cfg)
        case MainMenuOptions.EXIT:
            sys.exit()


@hydra.main(config_path="../../../configs", config_name="master", version_base=None)
def hydra_entry(cfg: DictConfig):
    main_menu(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    hydra_entry()
