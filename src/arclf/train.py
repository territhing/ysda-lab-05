import pathlib

import fire
import torch

from arclf.engine import TrainingEngine


def main(train_data_path: pathlib.Path, num_epochs: int = 10, device: str = "cpu"):
    device = torch.device(device)
    engine = TrainingEngine(train_data_path, device=device)
    engine.train(num_epochs=num_epochs)


if __name__ == "__main__":
    fire.Fire(main)
