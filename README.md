# danish-speech-eval

A benchmark for evaluating speech to text models on Danish datasets and domains.

## Table of Contents

- [Evaluation metrics](#evaluation-metrics)
- [Package structure](#package-structure)
- [Setup](#setup)
- [Supported models](#supported-models)
  - [HuggingFace pipeline backend (default)](#huggingface-pipeline-backend-default)
  - [OpenAI-compatible API backend (`--backend openai`)](#openai-compatible-api-backend---backend-openai)
  - [Azure OpenAI backend (`--backend azure_openai`)](#azure-openai-backend---backend-azure_openai)
  - [ElevenLabs Scribe v2](#elevenlabs-scribe-v2)
  - [Qwen3-ASR backend (`--backend qwen_asr`)](#qwen3-asr-backend---backend-qwen_asr)
  - [Evaluate on a specific dataset](#evaluate-on-a-specific-dataset)
  - [All CLI flags](#all-cli-flags)
- [Get your model on the leaderboard](#-get-your-model-on-the-leaderboard)
- [Leaderboard (HuggingFace Space)](#leaderboard-huggingface-space)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [CoRal](#coral)

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate — primary leaderboard metric, lower is better |
| **Normalized WER** | WER after text normalisation (numerals → words, lower-case, etc.) |

## Package structure

```
danish_speech/
├── __init__.py        # public API
├── data.py            # dataset loading & preprocessing
├── evaluate.py        # ASR evaluation logic
├── metrics.py         # WER / CER computation
├── run.py             # CLI entry point
├── submit.py          # leaderboard JSON management
├── types.py           # shared type aliases
└── leaderboards/
    ├── index.html     # HuggingFace Space leaderboard (deploy this)
    └── leaderboard.json  # evaluation results
```

## Setup

```bash
pip install .
```

If you want backend-specific support, install the matching extra:

```bash
# Azure OpenAI backend
pip install ".[azureopenai]"

# Qwen3-ASR backend
pip install ".[qwen_asr]"
```

Remember to set your HF token, using hf auth login or HF_TOKEN env var.

## Supported models

### HuggingFace pipeline backend (default)

Any model loadable via the `transformers` `automatic-speech-recognition` pipeline
works out of the box.

| Model family | Example IDs | Notes |
|---|---|---|
| **Whisper** | `openai/whisper-large-v3`, `openai/whisper-medium` | Recommended for best Danish WER |
| **Wav2Vec2** | `chcaa/xls-r-300m-danish`, `facebook/wav2vec2-large-xlsr-53-danish` | Add `--no-lm` to skip LM decoding |
| **MMS** | `facebook/mms-1b-all` | Multilingual; covers Danish |
| **Cohere transcription models** | `CohereForAI/c4ai-aya-expanse-8b` | Add `--trust-remote-code` |
| **Any community ASR model** | any HF model with `automatic-speech-recognition` tag | Add `--trust-remote-code` if prompted |

```bash
# Whisper (default, no extra flags)
danish-speech-eval --model openai/whisper-large-v3

# Wav2Vec2 without language model
danish-speech-eval --model Alvenir/wav2vec2-base-da --no-lm

# Cohere / models requiring remote code
danish-speech-eval --model syvai/hviske-v5.3 --trust-remote-code
```

### OpenAI-compatible API backend (`--backend openai`)

Any service that implements the `POST /audio/transcriptions` endpoint can be
evaluated. This includes OpenAI and self-hosted servers (whisper.cpp,
faster-whisper, etc.).

| Service | `--api-url` | `--model` |
|---|---|---|
| **OpenAI** | *(omit — uses default)* | `whisper-1` |
| **Local whisper.cpp / faster-whisper** | `http://localhost:8080/v1` | `whisper-1` |
| **Any OpenAI-compatible endpoint** | your endpoint | model name |

```bash
# OpenAI whisper-1
danish-speech-eval \
  --model whisper-1 \
  --backend openai \
  --api-key $OPENAI_API_KEY

# Local server (no key needed)
danish-speech-eval \
  --model whisper-1 \
  --backend openai \
  --api-url http://localhost:8080/v1 \
  --api-key none
```

The API key can also be set via the `OPENAI_API_KEY` environment variable and
the base URL via `OPENAI_BASE_URL`.

> Note: `openai` is only required when you use the OpenAI or Azure OpenAI
> backends. It is installed via the `azureopenai` extra.

### Azure OpenAI backend (`--backend azure_openai`)

Use `--backend azure_openai` for any model deployed in an Azure OpenAI resource,
including `gpt-4o-transcribe`.

| Setting | Value |
|---|---|
| `--api-url` | `https://<resource>.openai.azure.com` |
| `--model` | your deployment name (e.g. `gpt-4o-transcribe`) |
| `--api-version` | Azure API version (e.g. `2025-01-01-preview`) |

```bash
# Azure OpenAI — gpt-4o-transcribe deployment
danish-speech-eval \
  --model gpt-4o-transcribe \
  --backend azure_openai \
  --api-url https://my-resource.openai.azure.com \
  --api-key $AZURE_OPENAI_API_KEY \
  --api-version 2025-01-01-preview
```

You can also set these via environment variables to avoid passing them on every run:

```bash
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_API_VERSION=2025-01-01-preview

danish-speech-eval --model gpt-4o-transcribe --backend azure_openai
```

### ElevenLabs Scribe v2

[Scribe v2](https://elevenlabs.io/docs/api-reference/speech-to-text) is ElevenLabs'
speech-to-text model.

You can now run it directly from the CLI:

```bash
danish-speech-eval \
  --model scribe_v2 \
  --backend elevenlabs \
  --api-url https://api.elevenlabs.io \
  --api-key $ELEVENLABS_API_KEY
```

For `--backend elevenlabs`, use an ElevenLabs key (`$ELEVENLABS_API_KEY`).
`--api-version` is ignored for this backend.

### Qwen3-ASR backend (`--backend qwen_asr`)

The `qwen-asr` package can run Qwen3-ASR models directly without going through
the OpenAI-compatible API.

| Setting | Value |
|---|---|
| `--model` | Qwen model ID, for example `Qwen/Qwen3-ASR-1.7B` |
| `--backend` | `qwen_asr` |
| `--api-url` | Ignored |
| `--api-key` | Ignored |
| `--api-version` | Ignored |

```bash
# Install the optional dependency first
pip install ".[qwen_asr]"

# Run Qwen3-ASR directly
danish-speech-eval \
  --model Qwen/Qwen3-ASR-1.7B \
  --backend qwen_asr \
  --enforce-da
```

Qwen3-ASR supports local audio files, URLs, and batch inference. When
`--enforce-da` is enabled, the evaluator passes Danish language hints to the
model.


### Evaluate on all sets

Evaluate a model on all configured datasets (CoRal + Common Voice 17 Danish):

```bash
python -m danish_speech.run --model openai/whisper-large-v3
```

Or use the installed CLI entry point:

```bash
danish-speech-eval --model openai/whisper-large-v3
```

### Evaluate on a specific dataset

```bash
danish-speech-eval \
  --model openai/whisper-large-v3 \
  --dataset CoRal-project/coral-v3 \
  --subset conversation \
  --split test \
  --text-column text
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID or OpenAI-compatible model name |
| `--dataset` | all | HuggingFace dataset ID |
| `--dataset-name` | derived | Name shown in leaderboard |
| `--subset` | `None` | Dataset subset/config name |
| `--split` | `test` | Dataset split |
| `--audio-column` | `audio` | Name of the audio column |
| `--text-column` | `text` | Name of the transcription column |
| `--batch-size` | `8` | Inference batch size (HuggingFace backend) |
| `--no-lm` | `False` | Disable LM decoding (Wav2Vec2 models) |
| `--trust-remote-code` | `False` | Required for Cohere and some community models |
| `--backend` | `huggingface` | `huggingface`, `openai`, `azure_openai`, `elevenlabs`, or `qwen_asr` |
| `--api-url` | `None` | Base URL for OpenAI-compatible API (`OPENAI_BASE_URL`) / Azure endpoint (`AZURE_OPENAI_ENDPOINT`) / optional ElevenLabs base URL (`ELEVENLABS_API_URL`) |
| `--api-key` | `None` | API key (`OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY` / `ELEVENLABS_API_KEY`) |
| `--api-version` | `None` | Azure OpenAI API version, e.g. `2025-01-01-preview` (`AZURE_OPENAI_API_VERSION`); ignored for non-Azure backends |
| `--enforce-da` | `False` | Explicitly pass `language=da` to the backend. For Whisper (HuggingFace) also sets `task=transcribe`. No effect on CTC/Wav2Vec2 models. |
| `--enforce-hf-link` | `False` | For `--backend huggingface`, enforce Hugging Face linking behavior. Non-huggingface backends (`openai`, `azure_openai`, `elevenlabs`) are always marked `closed_source` and not linked. |
| `--n-indices` | `None` | Evaluate only the first N samples — useful for quick smoke-tests |
| `--cache-dir` | `None` | Directory for caching datasets |
| `--leaderboard` | `danish_speech/leaderboards/leaderboard.json` | Path to leaderboard JSON |

Results are written to `leaderboard.json` automatically.

A per-sample debug CSV is always saved after each run. The default path is:

```
mlruns/<model>-<timestamp>/<dataset>.csv
```

When called programmatically without a path, it falls back to `./debug_predictions.csv` in the working directory.

The CSV columns are: `sample_index`, `status`, `error`, `input_text_raw`, `input_text_normalized`, `prediction_text_raw`, `prediction_text_normalized`.

## Leaderboard data

The leaderboard JSON can be extended with optional per-model sample details
for a richer web viewer. A future leaderboard page can use that data to show
hover details, comparison views, and per-sample prediction examples without
changing the summary score table.

## 🚀 Get your model on the leaderboard

We evaluate models on request. To add your model:

1. **[Open an issue](https://github.com/Dansk-Data-Science-Community/danish-speech-eval/issues/new?template=model-eval-request.yml)** using the *Model evaluation request* template.
2. Fill in your HuggingFace model ID and any relevant details.
3. A maintainer will run the evaluation and push the result to the leaderboard.

## Leaderboard (HuggingFace Space)

The leaderboard is a self-contained HTML page that reads from `leaderboard.json`.

To deploy to a **HuggingFace static Space**:

```bash
# Copy both files to the root of your Space repository
cp danish_speech/leaderboards/index.html .
cp danish_speech/leaderboards/leaderboard.json .
```

The page supports:
- Sortable columns (click any header)
- Filter by dataset
- Model search
- WER colour coding (green < 15 %, amber < 30 %, red ≥ 30 %)

## Roadmap

- [ ] Text-to-speech (TTS) benchmark
- [ ] Automated CI evaluation on new issue submissions
- [ ] Per-speaker demographic breakdown (age, dialect, gender) for CoRal

## Acknowledgements

### CoRal

This benchmark is based on knowledge from the development of Danish ASR and TTS datasets
and models, as part of the [CoRal project](https://alexandra.dk/coral/), funded by the
[Innovation Fund](https://innovationsfonden.dk/).

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/coral/coral.html)
[![License](https://img.shields.io/github/license/CoRal-project/coral)](https://github.com/CoRal-project/coral/blob/main/LICENSE)

