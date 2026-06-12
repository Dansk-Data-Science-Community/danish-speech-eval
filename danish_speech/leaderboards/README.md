---
title: Danish Speech Eval
emoji: 🏆
colorFrom: green
colorTo: red
sdk: static
pinned: false
license: mit
short_description: A leaderboard for speech models on Danish (TtS and StT)
---

# Deploying the leaderboard to a HuggingFace Space

The leaderboard is a self-contained static site: `index.html` fetches
`leaderboard.json` from the same directory and renders the results table.

## 1. Create a HuggingFace Space

1. Go to <https://huggingface.co/new-space>.
2. Fill in a name (e.g. `danish-speech-eval`).
3. Select **Static** as the Space SDK.
4. Choose a licence and set visibility (public recommended).
5. Click **Create Space**.

## 2. Clone the Space repository

```bash
git clone https://huggingface.co/spaces/<your-org>/<your-space-name>
cd <your-space-name>
```

## 3. Copy the leaderboard files

From the root of **this** repository:

```bash
cp danish_speech/leaderboards/index.html      <your-space-name>/index.html
cp danish_speech/leaderboards/leaderboard.json <your-space-name>/leaderboard.json
cp -r danish_speech/leaderboards/assets/       <your-space-name>/assets/
```

The Space must have both files at the **root** so that the `fetch("leaderboard.json")`
call in `index.html` resolves correctly.

## 4. Push to HuggingFace

```bash
cd <your-space-name>
git add index.html leaderboard.json
git commit -m "Deploy Danish speech leaderboard"
git push
```

The Space goes live at:
`https://huggingface.co/spaces/<your-org>/<your-space-name>`

---

## Keeping the leaderboard up to date

After running a new evaluation the updated `leaderboard.json` is written
automatically by `danish-speech-eval`. Push the file to the Space to publish
the new result:

```bash
# Run evaluation (updates leaderboard.json in place)
danish-speech-eval --model openai/whisper-large-v3

# Copy updated file to the Space repo and push
cp danish_speech/leaderboards/leaderboard.json <your-space-name>/leaderboard.json
cd <your-space-name>
git add leaderboard.json
git commit -m "Add openai/whisper-large-v3 results"
git push
```

### Tip — use the HuggingFace Hub Python client

If you want to update the leaderboard without maintaining a local clone of the
Space you can upload the file directly:

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="danish_speech/leaderboards/leaderboard.json",
    path_in_repo="leaderboard.json",
    repo_id="<your-org>/<your-space-name>",
    repo_type="space",
    commit_message="Update leaderboard results",
)
```

---

## File reference

| File | Role |
|------|------|
| `index.html` | Static leaderboard page — deploy once, update rarely |
| `leaderboard.json` | Results data — update after every new evaluation |
