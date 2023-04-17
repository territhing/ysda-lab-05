import os.path

import torch

from arclf.engine import InferenceEngine
import streamlit as st

from arclf.model import ArxivClassificationModel
from arclf.taxonomy import ArxivTaxonomy


@st.cache_resource
def arxiv_model() -> ArxivClassificationModel:
    model = ArxivClassificationModel(len(ArxivTaxonomy().codes))
    # model.load_state_dict(torch.load("model-summary.pth", map_location="cpu"))
    return model.to("cpu")


def display_results(title: str, summary: str) -> None:
    if not title and not summary:
        st.error("You should specify either title or summary")
        return

    engine = InferenceEngine(model=arxiv_model())
    threshold: float = 0.65
    labels: set[str] = set()

    if title:
        labels.update(engine.run_inference_for_prompt(title, threshold=threshold))

    if summary:
        labels.update(engine.run_inference_for_prompt(summary, threshold=threshold))

    taxonomy = ArxivTaxonomy()
    table: list[dict[str, str]] = []

    for code in sorted(labels):
        table.append(
            {
                "Taxonomy Code": code,
                "Humane Name": taxonomy.get_human_name_for_code(code),
                "Description": taxonomy.get_description_for_code(code),
            }
        )

    st.table(table)


def main():
    st.title("Arxiv Abstract Classification Service")

    title = st.text_input(label="Title")
    summary = st.text_area(label="Abstract")

    if st.button(label="Submit"):
        display_results(title, summary)


main()
