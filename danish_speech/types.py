"""Shared type aliases for the danish-speech-eval package."""

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

Data = Dataset | DatasetDict | IterableDataset | IterableDatasetDict
