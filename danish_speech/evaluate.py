"""Evaluation of ASR models."""

import io
import logging
import os
import tempfile
from typing import Literal

import soundfile as sf
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.pipelines import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

from .data import DEFAULT_CONVERSION_DICT, process_example
from .metrics import cer, wer
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)

Backend = Literal["huggingface", "openai"]


def evaluate_asr(
    model_id: str,
    dataset: Dataset,
    audio_column: str,
    text_column: str,
    characters_to_keep: str,
    batch_size: int = 8,
    no_lm: bool = False,
    trust_remote_code: bool = False,
    backend: Backend = "huggingface",
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, float]:
    """Evaluate an ASR model on a pre-loaded dataset.

    Supports three evaluation backends selected via ``backend``:

    * ``"huggingface"`` — any model loadable via the ``transformers`` pipeline,
      including Whisper, Wav2Vec2, MMS, and Cohere transcription models.
    * ``"openai"`` — any service that implements the OpenAI
      ``POST /audio/transcriptions`` endpoint (OpenAI, Azure OpenAI, local
      whisper.cpp / faster-whisper servers, etc.).

    Args:
        model_id:
            HuggingFace model ID **or** the model name sent to an OpenAI-
            compatible API (e.g. ``"whisper-1"``).
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column in the dataset.
        text_column:
            Name of the transcription column in the dataset.
        characters_to_keep:
            String of characters to retain when normalising transcriptions.
        batch_size:
            Inference batch size (HuggingFace backend only). Defaults to 8.
        no_lm:
            Disable language model decoding (Wav2Vec2 models only).
            Defaults to False.
        trust_remote_code:
            Pass ``trust_remote_code=True`` to the HuggingFace pipeline.
            Required for some community models such as Cohere transcription
            models. Defaults to False.
        backend:
            Which evaluation backend to use. ``"huggingface"`` (default) or
            ``"openai"``.
        api_url:
            Base URL for the OpenAI-compatible API
            (e.g. ``"https://api.openai.com/v1"``). Required when
            ``backend="openai"``. Falls back to the ``OPENAI_BASE_URL``
            environment variable if not provided.
        api_key:
            API key for the OpenAI-compatible API. Falls back to the
            ``OPENAI_API_KEY`` environment variable if not provided.

    Returns:
        Dict with ``"wer"`` and ``"cer"`` scores in the range ``[0, 1]``.
    """
    if backend == "openai":
        predictions = _transcribe_openai(
            model_id=model_id,
            dataset=dataset,
            audio_column=audio_column,
            api_url=api_url,
            api_key=api_key,
        )
    else:
        logger.info("Loading ASR model %r...", model_id)
        transcriber = load_asr_pipeline(
            model_id=model_id,
            no_lm=no_lm,
            trust_remote_code=trust_remote_code,
        )
        predictions = _transcribe_hf(
            transcriber=transcriber,
            dataset=dataset,
            audio_column=audio_column,
            batch_size=batch_size,
            no_lm=no_lm,
        )

    predictions = [
        _normalise_text(p, characters_to_keep=characters_to_keep)
        for p in predictions
    ]
    labels = [
        _normalise_text(sample[text_column], characters_to_keep=characters_to_keep)
        for sample in dataset
    ]

    return {
        "wer": wer(predictions=predictions, labels=labels),
        "cer": cer(predictions=predictions, labels=labels),
    }


# ── HuggingFace backend ────────────────────────────────────────────────────────

def _transcribe_hf(
    transcriber: AutomaticSpeechRecognitionPipeline,
    dataset: Dataset,
    audio_column: str,
    batch_size: int,
    no_lm: bool,
) -> list[str]:
    """Transcribe a dataset using a HuggingFace ASR pipeline.

    Args:
        transcriber:
            Loaded ASR pipeline.
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column.
        batch_size:
            Inference batch size.
        no_lm:
            Whether LM decoding is disabled (controls generate_kwargs).

    Returns:
        List of raw transcription strings.
    """
    gen_kwargs: dict = (
        {} if no_lm else {"language": "danish", "task": "transcribe"}
    )
    predictions: list[str] = []
    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
        transformers_output_ignored(),
    ):
        for out in transcriber(
            KeyDataset(dataset=dataset, key=audio_column),  # type: ignore[arg-type]
            batch_size=batch_size,
            generate_kwargs=gen_kwargs,
        ):
            predictions.append(out["text"])
            pbar.update()
    return predictions


def load_asr_pipeline(
    model_id: str,
    no_lm: bool,
    trust_remote_code: bool = False,
) -> AutomaticSpeechRecognitionPipeline:
    """Load an ASR pipeline from a HuggingFace model ID.

    Args:
        model_id:
            The HuggingFace model ID to load.
        no_lm:
            When ``True``, loads the model as ``Wav2Vec2ForCTC`` without a
            language model. Only applicable to Wav2Vec 2.0 models.
        trust_remote_code:
            Pass ``trust_remote_code=True`` when loading. Required for some
            community models (e.g. Cohere transcription models).
            Defaults to False.

    Returns:
        The loaded ASR pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with transformers_output_ignored():
        if no_lm:
            model = Wav2Vec2ForCTC.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            processor = Wav2Vec2Processor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
        else:
            transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                device=device,
                trust_remote_code=trust_remote_code,
            )

    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber


# ── OpenAI-compatible API backend ─────────────────────────────────────────────

def _transcribe_openai(
    model_id: str,
    dataset: Dataset,
    audio_column: str,
    api_url: str | None,
    api_key: str | None,
) -> list[str]:
    """Transcribe a dataset using an OpenAI-compatible ``/audio/transcriptions`` endpoint.

    Writes each audio sample to a temporary WAV file and submits it to the
    API one sample at a time (the endpoint accepts a single file per request).

    Args:
        model_id:
            Model name to pass in the API request (e.g. ``"whisper-1"``).
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column.
        api_url:
            Base URL of the OpenAI-compatible service. Falls back to the
            ``OPENAI_BASE_URL`` environment variable, then the official
            OpenAI endpoint.
        api_key:
            API key. Falls back to the ``OPENAI_API_KEY`` environment variable.

    Returns:
        List of raw transcription strings.

    Raises:
        ImportError:
            If the ``openai`` package is not installed.
        ValueError:
            If no API key can be resolved.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for the OpenAI backend. "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An API key is required for the OpenAI backend. "
            "Pass --api-key or set the OPENAI_API_KEY environment variable."
        )

    resolved_url = api_url or os.getenv("OPENAI_BASE_URL")

    client = OpenAI(
        api_key=resolved_key,
        **({"base_url": resolved_url} if resolved_url else {}),
    )

    predictions: list[str] = []
    for sample in tqdm(dataset, desc="Transcribing"):
        audio = sample[audio_column]
        buf = io.BytesIO()
        sf.write(buf, audio["array"], audio["sampling_rate"], format="WAV")
        buf.seek(0)
        buf.name = "audio.wav"

        response = client.audio.transcriptions.create(
            model=model_id,
            file=buf,
            language="da",
        )
        predictions.append(response.text)

    return predictions


# ── shared helpers ─────────────────────────────────────────────────────────────

def _normalise_text(text: str, characters_to_keep: str) -> str:
    """Normalise a transcription string for metric computation.

    Args:
        text:
            Raw transcription text.
        characters_to_keep:
            String of allowed characters.

    Returns:
        Normalised text.
    """
    return process_example(
        example={"text": text},
        characters_to_keep=characters_to_keep,
        conversion_dict=DEFAULT_CONVERSION_DICT,
        text_column="text",
        audio_column=None,
        lower_case=True,
        convert_numerals=True,
        processor=None,
        normalise_audio=False,
        augment_audio=False,
    )["text"]
