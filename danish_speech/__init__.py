"""danish-speech-eval: Benchmark for evaluating Danish speech models."""

from .evaluate import evaluate_asr, load_asr_pipeline
from .run import main, run_evaluation
from .submit import update_leaderboard

__all__ = [
    "evaluate_asr",
    "load_asr_pipeline",
    "main",
    "run_evaluation",
    "update_leaderboard",
]
