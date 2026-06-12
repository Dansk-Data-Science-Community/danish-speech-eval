"""Main entry point for the Danish speech evaluation benchmark."""

import argparse
import logging
from pathlib import Path

from .data import load_dataset_for_evaluation
from .evaluate import evaluate_asr
from .submit import update_leaderboard

logger = logging.getLogger(__package__)

DANISH_CHARACTERS = "abcdefghijklmnopqrstuvwxyzæøå"
SAMPLING_RATE = 16_000

# Default datasets evaluated when no --dataset flag is provided.
EVAL_DATASETS: list[dict] = [
    {
        "dataset_id": "alexandrainst/coral",
        "dataset_name": "CoRal",
        "subset": None,
        "split": "test",
        "audio_column": "audio",
        "text_column": "text",
    },
    {
        "dataset_id": "mozilla-foundation/common_voice_17_0",
        "dataset_name": "Common Voice 17",
        "subset": "da",
        "split": "test",
        "audio_column": "audio",
        "text_column": "sentence",
    },
]


def run_evaluation(
    model_id: str,
    dataset_id: str,
    dataset_name: str,
    audio_column: str,
    text_column: str,
    split: str = "test",
    subset: str | None = None,
    batch_size: int = 8,
    no_lm: bool = False,
    trust_remote_code: bool = False,
    backend: str = "huggingface",
    api_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | None = None,
) -> float:
    """Load a dataset, run ASR evaluation, and return WER as a percentage.

    Args:
        model_id:
            HuggingFace model ID **or** model name for the OpenAI-compatible API.
        dataset_id:
            HuggingFace dataset ID (e.g. ``"alexandrainst/coral"``).
        dataset_name:
            Human-readable name shown in the leaderboard.
        audio_column:
            Name of the audio column in the dataset.
        text_column:
            Name of the transcription column in the dataset.
        split:
            Dataset split to evaluate on. Defaults to ``"test"``.
        subset:
            Dataset subset/config name, or ``None``. Defaults to None.
        batch_size:
            Inference batch size (HuggingFace backend). Defaults to 8.
        no_lm:
            Disable language model decoding (Wav2Vec2 models). Defaults to False.
        trust_remote_code:
            Pass ``trust_remote_code=True`` to the HuggingFace pipeline.
            Required for some community models (e.g. Cohere). Defaults to False.
        backend:
            Evaluation backend: ``"huggingface"`` or ``"openai"``.
            Defaults to ``"huggingface"``.
        api_url:
            Base URL for the OpenAI-compatible API. Defaults to None.
        api_key:
            API key for the OpenAI-compatible API. Defaults to None.
        cache_dir:
            Directory for caching downloaded datasets. Defaults to None.

    Returns:
        WER score as a percentage (e.g. ``12.34`` for 12.34 %).
    """
    logger.info("Loading dataset %r (split: %r)...", dataset_id, split)
    dataset = load_dataset_for_evaluation(
        dataset_id=dataset_id,
        eval_split_name=split,
        audio_column=audio_column,
        text_column=text_column,
        min_seconds_per_example=0.5,
        max_seconds_per_example=30,
        sampling_rate=SAMPLING_RATE,
        subset=subset,
        lower_case=True,
        characters_to_keep=list(DANISH_CHARACTERS),
        convert_numerals=True,
        cache_dir=cache_dir,
    )

    logger.info("Evaluating %r on %r...", model_id, dataset_name)
    scores = evaluate_asr(
        model_id=model_id,
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        characters_to_keep=DANISH_CHARACTERS,
        batch_size=batch_size,
        no_lm=no_lm,
        trust_remote_code=trust_remote_code,
        backend=backend,  # type: ignore[arg-type]
        api_url=api_url,
        api_key=api_key,
    )

    wer_pct = round(scores["wer"] * 100, 2)
    logger.info("WER on %r: %.2f%%", dataset_name, wer_pct)
    return wer_pct


def main() -> None:
    """Run the Danish speech evaluation benchmark CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a Danish ASR model and update the leaderboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID to evaluate"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset ID (omit to run all configured datasets)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Human-readable dataset name for the leaderboard",
    )
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument(
        "--split", default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--audio-column", default="audio", help="Name of the audio column"
    )
    parser.add_argument(
        "--text-column", default="text", help="Name of the transcription column"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Disable language model decoding (for Wav2Vec2 models)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the HuggingFace pipeline (required for Cohere models)",
    )
    parser.add_argument(
        "--backend",
        default="huggingface",
        choices=["huggingface", "openai"],
        help="Evaluation backend",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "Base URL for an OpenAI-compatible transcription API "
            "(e.g. https://api.openai.com/v1). "
            "Can also be set via OPENAI_BASE_URL. Only used with --backend openai."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the OpenAI-compatible endpoint. "
            "Can also be set via OPENAI_API_KEY. Only used with --backend openai."
        ),
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Directory for caching datasets"
    )
    parser.add_argument(
        "--leaderboard",
        default=None,
        help="Path to the leaderboard JSON file",
    )
    args = parser.parse_args()

    leaderboard_path = (
        Path(args.leaderboard)
        if args.leaderboard
        else Path(__file__).parent / "leaderboards" / "leaderboard.json"
    )

    datasets_to_eval = EVAL_DATASETS
    if args.dataset:
        datasets_to_eval = [
            {
                "dataset_id": args.dataset,
                "dataset_name": args.dataset_name or args.dataset.split("/")[-1],
                "subset": args.subset,
                "split": args.split,
                "audio_column": args.audio_column,
                "text_column": args.text_column,
            }
        ]

    for ds in datasets_to_eval:
        wer_score = run_evaluation(
            model_id=args.model,
            dataset_id=ds["dataset_id"],
            dataset_name=ds["dataset_name"],
            audio_column=ds["audio_column"],
            text_column=ds["text_column"],
            split=ds["split"],
            subset=ds["subset"],
            batch_size=args.batch_size,
            no_lm=args.no_lm,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
            api_url=args.api_url,
            api_key=args.api_key,
            cache_dir=args.cache_dir,
        )
        update_leaderboard(
            leaderboard_path=leaderboard_path,
            model_name=args.model,
            task="ASR",
            metric="WER",
            score=wer_score,
            dataset=ds["dataset_name"],
        )

    logger.info("Done. Leaderboard updated at %s", leaderboard_path)


if __name__ == "__main__":
    main()
