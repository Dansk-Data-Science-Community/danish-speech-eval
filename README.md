# danish-speech-eval

A benchmark for evaluating speech to text models on Danish datasets and domains.

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

### OpenAI-compatible API backend

Any service that implements the `POST /audio/transcriptions` endpoint can be
evaluated. This includes OpenAI, Azure OpenAI, and self-hosted servers
(whisper.cpp, faster-whisper, etc.).

| Service | `--api-url` | `--model` |
|---|---|---|
| **OpenAI** | *(omit — uses default)* | `whisper-1` |
| **Azure OpenAI** | `https://<resource>.openai.azure.com/openai` | your deployment name |
| **Local whisper.cpp / faster-whisper** | `http://localhost:8080/v1` | `whisper-1` |
| **Any OpenAI-compatible endpoint** | your endpoint | model name |

```bash
# OpenAI whisper-1
danish-speech-eval \
  --model whisper-1 \
  --backend openai \
  --api-key $OPENAI_API_KEY

# Azure OpenAI
danish-speech-eval \
  --model my-whisper-deployment \
  --backend openai \
  --api-url https://my-resource.openai.azure.com/openai \
  --api-key $AZURE_OPENAI_API_KEY

# Local server (no key needed)
danish-speech-eval \
  --model whisper-1 \
  --backend openai \
  --api-url http://localhost:8080/v1 \
  --api-key none
```

The API key can also be set via the `OPENAI_API_KEY` environment variable, and
the base URL via `OPENAI_BASE_URL`, so you don't need to pass them on every run.



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
  --subset da \
  --split test \
  --text-column sentence
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
| `--backend` | `huggingface` | `huggingface` or `openai` |
| `--api-url` | `None` | Base URL for OpenAI-compatible API (or set `OPENAI_BASE_URL`) |
| `--api-key` | `None` | API key for OpenAI-compatible API (or set `OPENAI_API_KEY`) |
| `--cache-dir` | `None` | Directory for caching datasets |
| `--leaderboard` | `danish_speech/leaderboards/leaderboard.json` | Path to leaderboard JSON |

Results are written to `leaderboard.json` automatically.

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

