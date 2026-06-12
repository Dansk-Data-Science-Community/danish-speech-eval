---
applyTo: "**/*.py,**/pyproject.toml,**/.flake8,**/.pre-commit-config.yaml,**/requirements*.txt"
---

# Python Coding Conventions

General Python conventions, patterns, and best practices for DSB AI/ML projects.
Derived from the [DSB AI/ML Code Bible](https://github.com/DanskeStatsbaner/aiml-codebible).

## Guiding Principles

- **Follow PEP 8** as the baseline standard. See [PEP 8](https://peps.python.org/pep-0008/).
- **Always type-hint** function signatures and return types (PEP 484).
- **Always write Google-style docstrings** on all public functions, classes, and modules.
- **Use snake_case** for variables, functions, methods, and module names.
- **Use PascalCase** for class names.
- **Names must be meaningful** — avoid single-letter names (`x`, `i`, `l`) except in short mathematical contexts.
- **Never add secrets** - remove login passwords, creds, other secrets in clear text and instead use/guess `os.env("<SECRET_KEY>", None)`
---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables & functions | `snake_case` | `frame_count`, `process_image()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRY_COUNT` |
| Classes | `PascalCase` | `CameraService`, `LogLevel` |
| Modules | lowercase, short, underscores OK | `camera_service`, `data_utils` |
| Packages | lowercase, alphanumeric, no underscores | `cameraservice`, `datautils` |
| Private attributes | `_single_underscore` prefix | `_client`, `_app_name` |

Note: Use meaningfull names when possible, e.g. never name a module `src`, try to avoid generic names and use domain and framework domain names and if deploying an application use component names for the modules, e.g. if fastapi use `router` module for routers and make a seperate module or import for the actual application code.  

---

## Formatting

### Line Length

- **Code**: max 79 characters.
- **Docstrings and comments**: max 72 characters.
- Does NOT apply to Markdown files.

### Indentation

- 4 spaces — never tabs. VS Code translates tabs to spaces automatically.
- You cannot mix spaces and tabs.

### Blank Lines

- **Top-level functions and class definitions**: surrounded by **2 blank lines**.
- **Methods inside a class**: surrounded by **1 blank line**.

### String Quotes

- Single `'` or double `"` quotes are both acceptable — pick one and stay consistent within a file.
- Docstrings always use triple double quotes: `"""docstring"""`.

### Whitespace in Expressions

```python
# Correct ✅
def function(arg1, arg2):
    result = arg1 + arg2

# Wrong ❌
def function( arg1,arg2 ):
    result=arg1+arg2
```

### Hanging Indentation

```python
# Dict
config = {
    "base_url": "https://example.com",
    "timeout": 30,
}

# List
cameras = [
    "cam_01",
    "cam_02",
]

# Function definition
def capture_frames(
    video_stream_class: CameraService,
    camera_id: str,
    logger: LoggerService,
) -> None:
```

---

## Docstrings

Use **Google Style** docstrings ([reference](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)) on all public functions, methods, classes, and modules. See [PEP 257](https://peps.python.org/pep-0257/).

### Rules

- The summary line must start with an **infinitive verb** (e.g., "Capture frames", "Return the processed result").
- Always use triple double quotes `"""`.
- Document all arguments, return values, and raised exceptions.

```python
def capture_frames(
    video_stream_class: CameraService,
    camera_id: str,
    logger: LoggerService,
) -> None:
    """Capture frames from a camera and upload them to the datalake.

    Args:
        video_stream_class (CameraService): Camera service instance.
        camera_id (str): ID of the camera to connect to.
        logger (LoggerService): Logger service instance.

    Raises:
        ConnectionError: If the camera cannot be reached.
    """
```

---

## Type Hints (PEP 484)

Always annotate function arguments and return types. For NumPy arrays, use `numpy.typing`:

```python
import numpy as np
from numpy.typing import NDArray


def sum_array(arr: NDArray[np.float64]) -> float:
    """Return the sum of a float64 array."""
    return float(np.sum(arr))
```

> **Note**: Type hints are advisory — Python does not enforce them at runtime. Use a linter (mypy / flake8) to catch violations statically.

---

## Function and Method Arguments

- Always use `self` as the first argument for instance methods.
- Always use `cls` as the first argument for class methods.

```python
class MyService:
    def process(self, data: str) -> str:
        """Process input data and return the trimmed result."""
        return data.strip()

    @classmethod
    def from_config(cls, config: dict) -> "MyService":
        """Create a MyService instance from a configuration dict."""
        return cls()
```

---

## Linting Toolchain

All Python projects must be linted using **ruff**, enforced via **pre-commit**.

```yaml
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
        rev: cef0300fd0fc4d2a87a85fa2093c6b283ea36f4b # v5.0.0
        hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
    - repo: https://github.com/astral-sh/ruff-pre-commit
        # Ruff version.
        rev: v0.15.15
        hooks:
            # Run the linter.
            - id: ruff-check
            types_or: [ python, pyi, jupyter, pyproject ]
            args: [ --fix ]
            # Run the formatter.
            - id: ruff-format
            types_or: [ python, pyi, jupyter ]
```

### Setup

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## Dependency Management

Use **uv** with exact version pins and a lockfile.

### Rules

- **Pin exact versions** in `pyproject.toml` — do not use `^`, `~`, or `*`:

  ```toml
  # ✅ Correct — no surprise updates
  pandas = "2.2.2"

  # ❌ Wrong — risks pulling breaking changes on re-lock
  pandas = "^2.2.2"
  ```

## uv (Preferred for New Projects)

**uv** is the preferred dependency manager for new Python projects at DSB. For full setup instructions, Dockerfile integration, and lockfile workflow, use the **uv skill**:

> 📦 **Skill**: [`uv-py-dependency-management`](../skills/uv-py-dependcy-management/SKILL.md)
> _Use it to scaffold `pyproject.toml`, set up the lockfile workflow, and configure uv in Docker._

Quick reference:

```bash
uv sync          # install / sync dependencies (creates uv.lock on first run)
uv sync -U       # update all dependencies
uv run main.py   # run without activating the venv manually
uv sync --locked # CI — fail if uv.lock is out of sync
```

---

## Testing

Use **pytest** as the primary testing framework.

### Test Types

| Type | What it tests |
|------|--------------|
| Unit | Single function/class in isolation with mocked dependencies |
| Functional | Application behaviour against business requirements |
| Integration | Interaction with external systems (APIs, databases, storage) |

### File and Folder Structure

```
tests/
├── unit/
│   └── test_<module>.py
├── functional/
│   └── test_<feature>.py
└── integration/
    └── test_<service>.py
```

### Naming

- Test files: `test_<module>.py`
- Test functions: `test_<method>_<expected>_<condition>`

### Pattern: Arrange–Act–Assert

```python
import unittest


class StringUtilsTests(unittest.TestCase):
    def test_concatenate_strings_returns_joined_string(self):
        # Arrange
        string1 = "Hello"
        string2 = "World"
        expected = "Hello World"

        # Act
        result = StringUtils.concatenate_strings(string1, string2)

        # Assert
        self.assertEqual(expected, result)
```

### Best Practices

- Mock external dependencies — never test third-party library internals.
- Unit tests must be **fast** and have **minimal third-party dependencies**.
- A failing unit test must **stop deployment**.
