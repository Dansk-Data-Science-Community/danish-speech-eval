# GitHub Copilot Instructions

This file configures GitHub Copilot for the **DSB SAP Classifier Assistant** repository.

---

## Python Coding Conventions

All Python code in this repository follows the DSB AI/ML standards defined in:

> 📄 [`instructions/python.instructions.md`](instructions/python.instructions.md)

Key rules at a glance:

- Follow **PEP 8**; max line length **79 chars** for code, **72** for docstrings.
- **Always type-hint** function signatures and return types (PEP 484).
- **Google-style docstrings** on all public functions, classes, and modules.
- `snake_case` for variables/functions, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for constants.
- **Never commit secrets** — use `os.environ.get("<SECRET_KEY>")`.
- Lint with **ruff** via pre-commit (`ruff-check --fix` + `ruff-format`).
- Test with **pytest** using the Arrange–Act–Assert pattern.

---

## Dependency Management

This project uses **uv** as the Python dependency manager. 

Key rules at a glance:

- Define dependencies in `pyproject.toml` with a minimum version and
  major-version upper bound: `fastapi>=0.116.1,<1`.
- Always commit `uv.lock` alongside `pyproject.toml`.
- Use `uv sync --locked` in CI/CD to fail fast on lockfile drift.
- Add `.venv` to both `.gitignore` and `.dockerignore`.
- In Docker: copy `uv` binary from the official image, pin the version,
  copy `pyproject.toml` + `uv.lock` before source for layer caching.

```bash
uv sync           # install / sync dependencies
uv sync -U        # update all dependencies
uv run main.py    # run without activating the venv manually
uv sync --locked  # CI — fail if uv.lock is out of sync
```
