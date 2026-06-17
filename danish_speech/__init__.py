"""danish-speech-eval: Benchmark for evaluating Danish speech models."""

from .evaluate import evaluate_asr, load_asr_pipeline
from .run import main, run_evaluation
from .submit import update_leaderboard
from .tts import generate_speech_elevenlabs

__all__ = [
    "evaluate_asr",
    "generate_speech_elevenlabs",
    "load_asr_pipeline",
    "main",
    "run_evaluation",
    "update_leaderboard",
]
