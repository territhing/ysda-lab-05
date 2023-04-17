import transformers
from torch import nn
import torch


class ArxivClassificationModel(nn.Module):
    def __init__(self, n_classes: int, *, hidden_size: int = 768):
        super().__init__()

        self.backbone = transformers.AutoModel.from_pretrained("distilbert-base-cased")
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_ids, attention_mask)
        return self.fc(embeddings)

    @torch.no_grad()
    def get_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.backbone.eval()

        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_state = outputs["last_hidden_state"]

        return hidden_state[:, 0, :]
