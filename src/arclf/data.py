import ast
import json
import pathlib
from typing import Any, Literal, TypedDict, Optional

import torch
import transformers
from torch.utils.data import Dataset

from arclf.taxonomy import ArxivTaxonomy

_ArxivDatasetSchema = TypedDict(
    "_ArxivDatasetSchema",
    {
        "labels": list[str],
        "input": str,
    },
)
_ArxivDatasetBatchEntrySchema = TypedDict(
    "_ArxivDatasetBatchEntrySchema",
    {
        "labels": torch.Tensor,
        "input": str,
    },
)
_ArxivDatasetBatchSchema = TypedDict(
    "_ArxivDatasetBatchSchema",
    {"labels": torch.Tensor, "attention_mask": torch.Tensor, "input_ids": torch.Tensor},
)


class ArxivDataset(torch.utils.data.Dataset):
    _data: list[_ArxivDatasetSchema]

    def __init__(
        self,
        tokenizer: transformers.BertTokenizer,
        taxonomy: ArxivTaxonomy,
        *,
        path: Optional[pathlib.Path] = None,
        mode: Literal["title", "summary"] = "summary"
    ):
        self._taxonomy = taxonomy
        self._tokenizer = tokenizer
        self._training = True

        if path is not None:
            with open(path) as fd:
                raw_data = json.load(fd)

            self._data = self._format_schema(raw_data, taxonomy, mode)

    @classmethod
    def from_prompt(
        cls,
        tokenizer: transformers.BertTokenizer,
        taxonomy: ArxivTaxonomy,
        prompt: str,
    ) -> "ArxivDataset":
        obj = cls(tokenizer, taxonomy)
        obj._data = [{"input": prompt}]
        obj._training = False

        return obj

    @staticmethod
    def _format_schema(
        dataset: list[dict[str, Any]],
        taxonomy: ArxivTaxonomy,
        mode: Literal["title", "summary"] = "summary",
    ) -> list[_ArxivDatasetSchema]:
        formatted_dataset = []

        for record in dataset:
            labels = [tag["term"] for tag in ast.literal_eval(record["tag"])]
            labels = [code for code in labels if code in taxonomy.codes]

            if not labels:
                continue

            formatted_dataset.append(
                {
                    "labels": labels,
                    "input": record[mode],
                }
            )

        return formatted_dataset

    def __getitem__(self, idx: int) -> _ArxivDatasetBatchEntrySchema:
        if not self._training:
            return {  # type: ignore
                "input": self._data[idx]["input"],
            }

        labels_tensor = torch.zeros(len(self._taxonomy.codes))
        labels = self._data[idx]["labels"]

        for code in labels:
            labels_tensor[self._taxonomy.convert_code_to_code_id(code)] = 1.0

        return {
            "input": self._data[idx]["input"],
            "labels": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self._data)

    def collate_fn(self, batch: list[_ArxivDatasetBatchEntrySchema]):
        inputs = [item["input"] for item in batch]
        tokenized_inputs = self._tokenizer(
            inputs, return_tensors="pt", truncation=True, padding=True
        )

        if self._training:
            labels = torch.stack([item["labels"] for item in batch])
            return tokenized_inputs | {"labels": labels}

        return tokenized_inputs
