import itertools
import pathlib
from typing import Literal, Optional

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer

from arclf.data import ArxivDataset
from arclf.model import ArxivClassificationModel
from arclf.taxonomy import ArxivTaxonomy
from arclf.data import _ArxivDatasetBatchSchema


class TrainingEngine:
    def __init__(
        self,
        data_path: pathlib.Path,
        mode: Literal["title", "summary"] = "summary",
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.mode = mode
        self.tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilbert-base-cased"
        )
        self.taxonomy = ArxivTaxonomy()

        self.dataset = ArxivDataset(
            self.tokenizer, self.taxonomy, path=data_path, mode=mode
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )

        self.device = device
        self.model = ArxivClassificationModel(len(self.taxonomy.codes)).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def train(self, num_epochs: int = 30) -> None:
        for _ in tqdm(range(num_epochs), desc="Epoch"):  # type: ignore
            self.train_single_epoch()
            torch.save(self.model.state_dict(), f"model-{self.mode}.pth")

    def train_single_epoch(self) -> None:
        self.model.train()
        pbar = tqdm(self.dataloader, leave=False)

        for batch in pbar:  # type: _ArxivDatasetBatchSchema
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.3f}")


class InferenceEngine:
    def __init__(self, model: ArxivClassificationModel, device: str = "cpu"):
        self.tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilbert-base-cased"
        )
        self.taxonomy = ArxivTaxonomy()
        self.device = torch.device(device)
        self.model = ArxivClassificationModel(len(self.taxonomy.codes)).to(self.device)
        self.model.eval()

    @classmethod
    def from_model(cls, model: ArxivClassificationModel) -> "InferenceEngine":
        obj = cls()

    @torch.inference_mode()
    def run_inference_for_prompt(
        self, prompt: str, threshold: float = 0.6
    ) -> list[str]:
        dataset = ArxivDataset.from_prompt(self.tokenizer, self.taxonomy, prompt=prompt)
        batch = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=1,
                    collate_fn=dataset.collate_fn,
                )
            )
        )

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        probs = torch.sigmoid(self.model(input_ids, attention_mask)).detach().cpu()[0]
        indices = list(itertools.chain(*(probs >= threshold).nonzero().tolist()))
        return sorted([self.taxonomy.convert_code_id_to_code(idx) for idx in indices])


if __name__ == "__main__":
    print(InferenceEngine().run_inference_for_prompt("foo bar"))
