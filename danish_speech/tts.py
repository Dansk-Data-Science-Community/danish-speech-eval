"""Text-to-Speech generation utilities."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__package__)


def generate_speech_elevenlabs(
    text: str,
    output_path: str | Path,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    api_key: str | None = None,
) -> Path:
    """Generate speech from text using the ElevenLabs API.

    Args:
        text:
            The text to convert to speech.
        output_path:
            Path where the audio file will be saved.
        voice_id:
            ElevenLabs voice ID to use. Defaults to the ``Rachel`` voice
            (``"21m00Tcm4TlvDq8ikWAM"``).
        api_key:
            ElevenLabs API key. Falls back to the ``ELEVENLABS_API_KEY``
            environment variable if not provided.

    Returns:
        Path to the saved audio file.

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
            "The 'elevenlabs' package is required for TTS generation. "
            "Install it with: pip install elevenlabs"
        ) from exc

    resolved_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An API key is required for ElevenLabs TTS. "
            "Pass api_key or set the ELEVENLABS_API_KEY environment variable."
        )

    client = ElevenLabs(api_key=resolved_key)

    logger.info("Generating speech for %d characters with voice %r...", len(text), voice_id)
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in response:
            f.write(chunk)

    logger.info("Saved TTS audio to %s", output_path)
    return output_path
