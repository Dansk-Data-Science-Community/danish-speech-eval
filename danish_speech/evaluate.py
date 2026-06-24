"""Evaluation of ASR models."""

import csv
import io
import logging
import os
import tempfile
from pathlib import Path
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

Backend = Literal[
    "huggingface",
    "openai",
    "azure_openai",
    "elevenlabs",
    "qwen_asr",
]
Device = Literal["auto", "cpu", "cuda"]
TRANSCRIPTION_FAILED_LOG = "Transcription failed for sample %d; skipping."


def evaluate_asr(  # NOSONAR
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
    api_version: str | None = None,
    debug_csv_path: str | Path | None = None,
    enforce_da: bool = False,
    device: Device = "auto",
) -> dict[str, float | int]:
    """Evaluate an ASR model on a pre-loaded dataset.

    Supports five evaluation backends selected via ``backend``:

    * ``"huggingface"`` — any model loadable via the ``transformers`` pipeline,
      including Whisper, Wav2Vec2, MMS, and Cohere transcription models.
    * ``"openai"`` — any service that implements the OpenAI
      ``POST /audio/transcriptions`` endpoint (OpenAI, Azure OpenAI, local
            whisper.cpp / faster-whisper servers, etc.).
        * ``"azure_openai"`` — Azure OpenAI transcription deployments.
        * ``"elevenlabs"`` — ElevenLabs speech-to-text models
            (e.g. ``"scribe_v2"``).
        * ``"qwen_asr"`` — Qwen3-ASR models via the ``qwen-asr`` package.

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
            Which evaluation backend to use. ``"huggingface"`` (default),
            ``"openai"``, ``"azure_openai"``, ``"elevenlabs"``, or
            ``"qwen_asr"``.
        api_url:
            Base URL for the OpenAI-compatible API
            (e.g. ``"https://api.openai.com/v1"``). For ``"azure_openai"``
            this is the Azure endpoint URL
            (e.g. ``"https://<resource>.openai.azure.com"``).
            For ``"elevenlabs"``, this can be used as a custom base URL.
            Falls back to the ``OPENAI_BASE_URL`` / ``AZURE_OPENAI_ENDPOINT`` /
            ``ELEVENLABS_API_URL`` environment variables if not provided.
        api_key:
            API key for the API. Falls back to the ``OPENAI_API_KEY`` /
            ``AZURE_OPENAI_API_KEY`` / ``ELEVENLABS_API_KEY`` environment
            variable if not provided.
        api_version:
            Azure OpenAI API version (e.g. ``"2024-02-01"``).
            Required when ``backend="azure_openai"``. Falls back to the
            ``AZURE_OPENAI_API_VERSION`` environment variable if not provided.
            Ignored for other backends.
        debug_csv_path:
            Optional output path for a per-sample debug CSV containing input
            text, prediction text, and any per-sample transcription error.
            Defaults to ``Path.cwd() / "debug_predictions.csv"`` when not provided.
        enforce_da:
            When ``True``, explicitly pass the Danish language code to every
            backend that supports it:

            * HuggingFace: ``generate_kwargs={"language": "danish", "task": "transcribe"}``
              (Whisper-family only; logged as a warning for Wav2Vec2/CTC models).
            * OpenAI / Azure OpenAI: ``language="da"`` in the transcriptions request.
            * ElevenLabs: ``language_code="da"`` in the STT request.

            When ``False`` (default), no language hint is sent and the model
            auto-detects the language.
        device:
            Device selection for local model inference backends
            (``"huggingface"`` and ``"qwen_asr"``).
            ``"auto"`` selects CUDA when available, otherwise CPU.
            ``"cpu"`` forces CPU.
            ``"cuda"`` forces CUDA and raises an error if CUDA is unavailable.

    Returns:
        Dict with ``"wer"`` and ``"cer"`` scores in the range ``[0, 1]``,
        and ``"n"`` as the number of successfully evaluated samples.
    """
    if backend == "elevenlabs":
        predictions, successful_indices, failed_errors = _transcribe_elevenlabs(
            model_id=model_id,
            dataset=dataset,
            audio_column=audio_column,
            api_url=api_url,
            api_key=api_key,
            enforce_da=enforce_da,
        )
    elif backend == "azure_openai":
        predictions, successful_indices, failed_errors = _transcribe_azure_openai(
            model_id=model_id,
            dataset=dataset,
            audio_column=audio_column,
            api_url=api_url,
            api_key=api_key,
            api_version=api_version,
            enforce_da=enforce_da,
        )
    elif backend == "openai":
        predictions, successful_indices, failed_errors = _transcribe_openai(
            model_id=model_id,
            dataset=dataset,
            audio_column=audio_column,
            api_url=api_url,
            api_key=api_key,
            enforce_da=enforce_da,
        )
    elif backend == "qwen_asr":
        predictions, successful_indices, failed_errors = _transcribe_qwen_asr(
            model_id=model_id,
            dataset=dataset,
            audio_column=audio_column,
            enforce_da=enforce_da,
            device=device,
        )
    else:
        logger.info("Loading ASR model %r...", model_id)
        try:
            transcriber = load_asr_pipeline(
                model_id=model_id,
                no_lm=no_lm,
                trust_remote_code=trust_remote_code,
                device=device,
            )
            predictions, successful_indices, failed_errors = _transcribe_hf(
                transcriber=transcriber,
                dataset=dataset,
                audio_column=audio_column,
                batch_size=batch_size,
                no_lm=no_lm,
                enforce_da=enforce_da,
            )
        except ValueError as exc:
            # Qwen3-ASR uses a custom model type not supported by the generic
            # transformers ASR pipeline path used by this backend.
            if "qwen3_asr" not in str(exc).lower():
                raise

            logger.warning(
                "Detected Qwen3-ASR checkpoint with --backend huggingface; "
                "falling back to --backend qwen_asr."
            )
            predictions, successful_indices, failed_errors = _transcribe_qwen_asr(
                model_id=model_id,
                dataset=dataset,
                audio_column=audio_column,
                enforce_da=enforce_da,
                device=device,
            )

    if not successful_indices:
        raise ValueError(
            "No successful transcriptions were produced, so WER/CER cannot be computed."
        )

    raw_predictions = predictions
    predictions = [
        _normalise_text(p, characters_to_keep=characters_to_keep)
        for p in raw_predictions
    ]
    raw_labels = [str(dataset[idx][text_column]) for idx in successful_indices]
    labels = [
        _normalise_text(
            str(dataset[idx][text_column]),
            characters_to_keep=characters_to_keep,
        )
        for idx in successful_indices
    ]

    if debug_csv_path is None:
        debug_csv_path = Path.cwd() / "debug_predictions.csv"

    _write_debug_csv(
        dataset=dataset,
        text_column=text_column,
        successful_indices=successful_indices,
        raw_labels=raw_labels,
        labels=labels,
        raw_predictions=raw_predictions,
        predictions=predictions,
        failed_errors=failed_errors,
        debug_csv_path=debug_csv_path,
    )

    failed = len(dataset) - len(successful_indices)
    if failed:
        logger.warning(
            "Skipped %d samples due to transcription failures; metrics computed on n=%d.",
            failed,
            len(successful_indices),
        )

    return {
        "wer": wer(predictions=predictions, labels=labels),
        "cer": cer(predictions=predictions, labels=labels),
        "n": len(successful_indices),
    }


# ── HuggingFace backend ────────────────────────────────────────────────────────

def _transcribe_hf(
    transcriber: AutomaticSpeechRecognitionPipeline,
    dataset: Dataset,
    audio_column: str,
    batch_size: int,
    no_lm: bool,
    enforce_da: bool = False,
) -> tuple[list[str], list[int], dict[int, str]]:
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
        enforce_da:
            When ``True``, force ``language="danish"`` in generate_kwargs.
            Only effective for Whisper-type encoder-decoder models.

    Returns:
        Tuple of raw transcription strings, their dataset indices, and errors.
    """
    if enforce_da and no_lm:
        logger.warning(
            "enforce_da has no effect with no_lm=True: language forcing is only "
            "supported by Whisper-family models, not CTC/Wav2Vec2 models."
        )
    if enforce_da and not no_lm:
        gen_kwargs: dict = {"language": "danish", "task": "transcribe"}
    else:
        gen_kwargs = {}
    predictions: list[str] = []
    successful_indices: list[int] = []
    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
        transformers_output_ignored(),
    ):
        for idx, out in enumerate(
            transcriber(
            KeyDataset(dataset=dataset, key=audio_column),  # type: ignore[arg-type]
            batch_size=batch_size,
            generate_kwargs=gen_kwargs,
            )
        ):
            predictions.append(out["text"])
            successful_indices.append(idx)
            pbar.update()
    return predictions, successful_indices, {}


def load_asr_pipeline(
    model_id: str,
    no_lm: bool,
    trust_remote_code: bool = False,
    device: Device = "auto",
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
        device:
            Device selection: ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.

    Returns:
        The loaded ASR pipeline.
    """
    resolved_device = _resolve_local_device(device)
    pipeline_device = 0 if resolved_device == "cuda" else -1

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
                device=pipeline_device,
            )
        else:
            transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                device=pipeline_device,
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
    enforce_da: bool = False,
) -> tuple[list[str], list[int], dict[int, str]]:
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
        enforce_da:
            When ``True``, pass ``language="da"`` to the transcriptions endpoint.

    Returns:
        Tuple of raw transcription strings, their dataset indices, and errors.

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
    successful_indices: list[int] = []
    failed_errors: dict[int, str] = {}
    for idx, sample in enumerate(tqdm(dataset, desc="Transcribing")):
        try:
            audio = sample[audio_column]
            buf = io.BytesIO()
            sf.write(buf, audio["array"], audio["sampling_rate"], format="WAV")
            buf.seek(0)
            buf.name = "audio.wav"

            response = client.audio.transcriptions.create(
                model=model_id,
                file=buf,
                **({
                    "language": "da"
                } if enforce_da else {}),
            )
            predictions.append(response.text)
            successful_indices.append(idx)
        except Exception as exc:
            logger.exception(TRANSCRIPTION_FAILED_LOG, idx)
            failed_errors[idx] = str(exc)

    return predictions, successful_indices, failed_errors


def _transcribe_qwen_asr(
    model_id: str,
    dataset: Dataset,
    audio_column: str,
    enforce_da: bool = False,
    device: Device = "auto",
) -> tuple[list[str], list[int], dict[int, str]]:
    """Transcribe a dataset using the qwen-asr package.

    Args:
        model_id:
            Qwen3-ASR model ID to load.
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column.
        enforce_da:
            When ``True``, pass ``language="Danish"`` to the transcriber.
        device:
            Device selection: ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.

    Returns:
        Tuple of raw transcription strings, their dataset indices, and errors.

    Raises:
        ImportError:
            If the ``qwen-asr`` package is not installed.
    """
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError(
            "The 'qwen-asr' package is required for the qwen_asr backend. "
            "Install it with: pip install qwen-asr"
        ) from exc

    resolved_device = _resolve_local_device(device)
    device_map = "cuda:0" if resolved_device == "cuda" else "cpu"
    dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32

    logger.info("Loading Qwen ASR model %r...", model_id)
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    predictions: list[str] = []
    successful_indices: list[int] = []
    failed_errors: dict[int, str] = {}
    language = "Danish" if enforce_da else None

    for idx, sample in enumerate(tqdm(dataset, desc="Transcribing")):
        try:
            audio = sample[audio_column]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio["array"], audio["sampling_rate"])
                results = model.transcribe(audio=tmp.name, language=language)

            result = results[0] if isinstance(results, list) else results
            predictions.append(result.text)
            successful_indices.append(idx)
        except Exception as exc:
            logger.exception(TRANSCRIPTION_FAILED_LOG, idx)
            failed_errors[idx] = str(exc)

    return predictions, successful_indices, failed_errors


def _resolve_local_device(device: Device) -> Literal["cpu", "cuda"]:
    """Resolve local device choice for local model inference backends."""
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested, but CUDA is not available. "
                "Use --device cpu or --device auto."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _transcribe_azure_openai(
    model_id: str,
    dataset: Dataset,
    audio_column: str,
    api_url: str | None,
    api_key: str | None,
    api_version: str | None,
    enforce_da: bool = False,
) -> tuple[list[str], list[int], dict[int, str]]:
    """Transcribe a dataset using an Azure OpenAI ``/audio/transcriptions`` endpoint.

    Args:
        model_id:
            Azure OpenAI deployment name (e.g. ``"whisper"``), not the model
            name — it must match the deployment created in your Azure resource.
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column.
        api_url:
            Azure OpenAI endpoint URL
            (e.g. ``"https://<resource>.openai.azure.com"``).
            Falls back to the ``AZURE_OPENAI_ENDPOINT`` environment variable.
        api_key:
            Azure OpenAI API key. Falls back to the ``AZURE_OPENAI_API_KEY``
            environment variable.
        api_version:
            Azure OpenAI API version (e.g. ``"2024-02-01"``).
            Falls back to the ``AZURE_OPENAI_API_VERSION`` environment variable.
        enforce_da:
            When ``True``, pass ``language="da"`` to the transcriptions endpoint.

    Returns:
        Tuple of raw transcription strings, their dataset indices, and errors.

    Raises:
        ImportError:
            If the ``openai`` package is not installed.
        ValueError:
            If the endpoint URL, API key, or API version cannot be resolved.
    """
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for the Azure OpenAI backend. "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An API key is required for the Azure OpenAI backend. "
            "Pass --api-key or set the AZURE_OPENAI_API_KEY environment variable."
        )

    resolved_url = api_url or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not resolved_url:
        raise ValueError(
            "An endpoint URL is required for the Azure OpenAI backend. "
            "Pass --api-url or set the AZURE_OPENAI_ENDPOINT environment variable."
        )

    resolved_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
    if not resolved_version:
        raise ValueError(
            "An API version is required for the Azure OpenAI backend. "
            "Pass --api-version or set the AZURE_OPENAI_API_VERSION environment variable."
        )

    client = AzureOpenAI(
        api_key=resolved_key,
        azure_endpoint=resolved_url,
        api_version=resolved_version,
    )

    predictions: list[str] = []
    successful_indices: list[int] = []
    failed_errors: dict[int, str] = {}
    for idx, sample in enumerate(tqdm(dataset, desc="Transcribing")):
        try:
            audio = sample[audio_column]
            buf = io.BytesIO()
            sf.write(buf, audio["array"], audio["sampling_rate"], format="WAV")
            buf.seek(0)
            buf.name = "audio.wav"

            response = client.audio.transcriptions.create(
                model=model_id,
                file=buf,
                **({
                    "language": "da"
                } if enforce_da else {}),
            )
            predictions.append(response.text)
            successful_indices.append(idx)
        except Exception as exc:
            logger.exception(TRANSCRIPTION_FAILED_LOG, idx)
            failed_errors[idx] = str(exc)

    return predictions, successful_indices, failed_errors


def _transcribe_elevenlabs(
    model_id: str,
    dataset: Dataset,
    audio_column: str,
    api_url: str | None,
    api_key: str | None,
    enforce_da: bool = False,
) -> tuple[list[str], list[int], dict[int, str]]:
    """Transcribe a dataset using ElevenLabs speech-to-text.

    Args:
        model_id:
            ElevenLabs speech-to-text model ID (e.g. ``"scribe_v2"``).
        dataset:
            Pre-loaded evaluation dataset.
        audio_column:
            Name of the audio column.
        api_url:
            Optional custom ElevenLabs base URL.
            Falls back to ``ELEVENLABS_API_URL``.
        api_key:
            ElevenLabs API key.
            Falls back to ``ELEVENLABS_API_KEY``.
        enforce_da:
            When ``True``, pass ``language_code="da"`` to the STT endpoint.

    Returns:
        Tuple of raw transcription strings, their dataset indices, and errors.

    Raises:
        ImportError:
            If the ``elevenlabs`` package is not installed.
        ValueError:
            If no API key can be resolved.
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as exc:
        raise ImportError(
            "The 'elevenlabs' package is required for the ElevenLabs backend. "
            "Install it with: pip install elevenlabs"
        ) from exc

    resolved_key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An API key is required for the ElevenLabs backend. "
            "Pass --api-key or set the ELEVENLABS_API_KEY environment variable."
        )

    client_kwargs: dict[str, str] = {"api_key": resolved_key}
    resolved_url = api_url or os.getenv("ELEVENLABS_API_URL")
    if resolved_url:
        client_kwargs["base_url"] = resolved_url

    client = ElevenLabs(**client_kwargs)

    predictions: list[str] = []
    successful_indices: list[int] = []
    failed_errors: dict[int, str] = {}
    for idx, sample in enumerate(tqdm(dataset, desc="Transcribing")):
        try:
            audio = sample[audio_column]
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, audio["array"], audio["sampling_rate"], format="WAV")
                with open(tmp.name, "rb") as audio_file:
                    response = client.speech_to_text.convert(
                        file=audio_file,
                        model_id=model_id,
                        **({
                            "language_code": "da"
                        } if enforce_da else {}),
                    )

            text = getattr(response, "text", None)
            if not text and isinstance(response, dict):
                text = response.get("text")
            if not text:
                raise ValueError(
                    "ElevenLabs response did not include transcription text."
                )

            predictions.append(text)
            successful_indices.append(idx)
        except Exception as exc:
            logger.exception(TRANSCRIPTION_FAILED_LOG, idx)
            failed_errors[idx] = str(exc)

    return predictions, successful_indices, failed_errors


def _write_debug_csv(
    dataset: Dataset,
    text_column: str,
    successful_indices: list[int],
    raw_labels: list[str],
    labels: list[str],
    raw_predictions: list[str],
    predictions: list[str],
    failed_errors: dict[int, str],
    debug_csv_path: str | Path,
) -> None:
    """Write per-sample debug rows with labels, predictions, and errors."""
    output_path = Path(debug_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_position = {idx: pos for pos, idx in enumerate(successful_indices)}

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample_index",
                "status",
                "error",
                "input_text_raw",
                "input_text_normalized",
                "prediction_text_raw",
                "prediction_text_normalized",
            ],
        )
        writer.writeheader()

        for idx, sample in enumerate(dataset):
            pos = success_position.get(idx)
            if pos is not None:
                writer.writerow(
                    {
                        "sample_index": idx,
                        "status": "success",
                        "error": "",
                        "input_text_raw": raw_labels[pos],
                        "input_text_normalized": labels[pos],
                        "prediction_text_raw": raw_predictions[pos],
                        "prediction_text_normalized": predictions[pos],
                    }
                )
            else:
                writer.writerow(
                    {
                        "sample_index": idx,
                        "status": "failed",
                        "error": failed_errors.get(idx, "unknown_error"),
                        "input_text_raw": str(sample[text_column]),
                        "input_text_normalized": "",
                        "prediction_text_raw": "",
                        "prediction_text_normalized": "",
                    }
                )

    logger.info("Saved debug CSV to %s", output_path)


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
