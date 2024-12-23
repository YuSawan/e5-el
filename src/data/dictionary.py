import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class Entity:
    id: str
    name: str
    description: str
    label_id: int
    encoding: Optional[BatchEncoding] = None
    meta: Optional[Any] = None


@dataclass
class EntityDictionary:
    entity_dict: Dataset
    entity_ids: list[str]
    nil_id: Optional[str] = None

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            dictionary_path: str|os.PathLike,
            cache_dir: Optional[str] = None,
            training_arguments: Optional[TrainingArguments] = None,
            nil: Optional[dict[str, str]] = None,
        ) -> None:
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path
        self.entity_dict = self.read_dictionary(dictionary_path, cache_dir, training_arguments, nil)
        self.entity_ids = list(self.entity_dict['id'])
        if not nil:
            self.nil_id = None

    def read_dictionary(
            self,
            dictionary_path: str|os.PathLike,
            cache_dir: Optional[str] = None,
            training_arguments: Optional[TrainingArguments] = None,
            nil: Optional[dict[str, str]] = None
        ) -> Dataset:
        def preprocess(documents: Dataset) -> Any:
            outputs: dict[str, list[Any]] = {"id": [], "name": [], "description": [], "encodings": []}
            for idx, name, description in zip(documents["id"], documents["name"], documents["description"]):
                try:
                    description, encodings = self.get_encoding(name=name, description=description)
                except KeyError:
                    description, encodings = self.get_encoding(name=name)
                outputs["id"].append(idx)
                outputs["name"].append(name)
                outputs["description"].append(description)
                outputs["encodings"].append(encodings)
            return outputs

        raw_dictionary = load_dataset("json", data_files={"dictionary": dictionary_path}, cache_dir=cache_dir)
        if nil:
            nil_id = nil["id"] if "id" in list(nil.keys()) else "-1"
            nil_name = nil["name"] if "name" in list(nil.keys()) else "[NIL]"
            nil_description = nil["description"] if "description" in list(nil.keys()) else '[NIL] is an entity that does not exist in the dictionary.'
            raw_dictionary['dictionary'] = raw_dictionary["dictionary"].add_item({"id": nil_id, "name": nil_name, "description": nil_description})
            self.nil_id = nil_id
        if training_arguments:
            with training_arguments.main_process_first(desc="dataset map pre-processing"):
                column_names = next(iter(raw_dictionary.values())).column_names
                dictionary = raw_dictionary.map(preprocess, batched=True, remove_columns=column_names)["dictionary"]
        else:
            column_names = next(iter(raw_dictionary.values())).column_names
            dictionary = raw_dictionary.map(preprocess, batched=True, remove_columns=column_names)["dictionary"]

        return dictionary

    def __call__(self, key: str) -> Entity:
        index = self.entity_dict["id"].index(key)
        value = self.entity_dict[index]
        return Entity(id=value["id"], name=value["name"], description=value["description"], label_id=index, encoding=value["encodings"])

    def __len__(self) -> int:
        return len(self.entity_dict)

    def __iter__(self) -> Iterator[Entity]:
        for index, value in enumerate(self.entity_dict):
            yield Entity(id=value["id"], name=value["name"], description=value["description"], label_id=index, encoding=value["encodings"])

    def __getitem__(self, idx: int) -> Entity:
        value = self.entity_dict[idx]
        return Entity(id=value["id"], name=value["name"], description=value["description"], label_id=idx, encoding=value["encodings"])

    def get_encoding(self, name: str, description: Optional[str] = None) -> tuple[str, BatchEncoding]:
        if "luke" in self.model_name:
            description = description if description else f"{name} is an entity in this dictionary."
            encoding  = self.tokenizer(
                description,
                entity_spans=[(0, len(name))],
                padding=True,
                truncation=True
            )
        elif "e5" in self.model_name:
            description = description if description else f"{name} is an entity in this dictionary."
            if "instruct" not in self.model_name:
                description = 'passages: ' + description
            encoding = self.tokenizer(description, padding=True, truncation=True)
        else:
            text = name + f" {self.tokenizer.sep_token} "
            text += description if description else f"{name} is an entity in this dictionary."
            description = text
            encoding = self.tokenizer(
                description,
                padding=True,
                truncation=True
            )
        return description, encoding
