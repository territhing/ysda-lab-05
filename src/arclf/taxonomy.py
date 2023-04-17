import functools
import json
import pathlib
from functools import cached_property
from typing import TypedDict

_DEFAULT_TAXONOMY_PATH = pathlib.Path(__file__).parent / "taxonomy.json"
_TaxonomyEntranceSchema = TypedDict(
    "_TaxonomyEntranceSchema",
    {
        "code": str,
        "human name": str,
        "description": str,
    },
)


class ArxivTaxonomy:
    def __init__(self, *, file_path: pathlib.Path = _DEFAULT_TAXONOMY_PATH):
        with open(file_path) as fd:
            self._data: list[_TaxonomyEntranceSchema] = json.load(fd)

        pairs = list(enumerate(sorted(set(record["code"] for record in self._data))))

        self._code2id: dict[str, int] = {code: code_id for code_id, code in pairs}
        self._id2code: dict[int, str] = dict(pairs)

    @classmethod
    def from_file_path(cls, file_path: pathlib.Path) -> "ArxivTaxonomy":
        return cls(file_path=file_path)

    @cached_property
    def codes(self) -> list[str]:
        return [record["code"] for record in self._data]

    @functools.cache
    def get_human_name_for_code(self, code: str) -> str:
        for record in self._data:
            if record["code"] == code:
                return record["human name"]

        raise RuntimeError(f"could not find human name for code: {code}")

    @functools.cache
    def get_description_for_code(self, code: str) -> str:
        for record in self._data:
            if record["code"] == code:
                return record["description"]

        raise RuntimeError(f"could not find description for code: {code}")

    def convert_code_to_code_id(self, code: str) -> int:
        return self._code2id[code]

    def convert_code_id_to_code(self, code_id: int) -> str:
        return self._id2code[code_id]
