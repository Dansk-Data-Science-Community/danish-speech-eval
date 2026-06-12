"""Functions for updating the leaderboard with new evaluation results."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__package__)


def load_leaderboard(leaderboard_path: Path) -> list[dict]:
    """Load leaderboard entries from a JSON file.

    Args:
        leaderboard_path: Path to the leaderboard JSON file.

    Returns:
        List of leaderboard entry dicts, or an empty list if the file does not exist.
    """
    if not leaderboard_path.exists():
        return []
    with leaderboard_path.open(encoding="utf-8") as f:
        return json.load(f)


def save_leaderboard(leaderboard_path: Path, entries: list[dict]) -> None:
    """Persist leaderboard entries to a JSON file.

    Args:
        leaderboard_path: Path to write the JSON file.
        entries: List of leaderboard entry dicts to save.
    """
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with leaderboard_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    logger.info("Leaderboard saved to %s", leaderboard_path)


def update_leaderboard(
    leaderboard_path: Path,
    model_name: str,
    task: str,
    metric: str,
    score: float,
    dataset: str,
) -> None:
    """Add or update a result entry in the leaderboard JSON file.

    If an entry with the same model_name and dataset already exists it is
    overwritten; otherwise a new entry is appended.

    Args:
        leaderboard_path: Path to the leaderboard JSON file.
        model_name: HuggingFace model ID (e.g. ``"openai/whisper-large-v3"``).
        task: Evaluation task label (e.g. ``"ASR"``).
        metric: Metric name (e.g. ``"WER"``).
        score: Metric value as a percentage (e.g. ``12.34`` for 12.34 % WER).
        dataset: Human-readable dataset name (e.g. ``"CoRal"``).
    """
    entries = load_leaderboard(leaderboard_path)

    existing_idx = next(
        (
            i
            for i, e in enumerate(entries)
            if e["model_name"] == model_name and e["dataset"] == dataset
        ),
        None,
    )

    new_entry: dict = {
        "model_name": model_name,
        "task": task,
        "metric": metric,
        "score": score,
        "dataset": dataset,
        "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
    }

    if existing_idx is not None:
        old_score = entries[existing_idx]["score"]
        logger.info(
            "Updating %r on %r: %.2f%% → %.2f%%",
            model_name,
            dataset,
            old_score,
            score,
        )
        entries[existing_idx] = new_entry
    else:
        logger.info("Adding %r on %r: %.2f%%", model_name, dataset, score)
        entries.append(new_entry)

    save_leaderboard(leaderboard_path, entries)
