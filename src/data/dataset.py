from collections.abc import Iterable
from typing import Any, Optional, TypedDict

from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding


class Mention(TypedDict):
    start: int
    end: int
    label: list[str]


class Example(TypedDict):
    id: str
    text: str
    entities: list[Mention]


def read_dataset(
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        cache_dir: Optional[str] = None
        ) -> DatasetDict:
    """
    DatasetReader is for processing
    Input:
        train_file: dataset path for training
        validation_file: dataset path for validation
        test_file: dataset path for test
    Output: DatasetDict
    """
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
    cache_dir = cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    return raw_datasets


class Preprocessor:
    """
    Base processor for Prepare of models.
    The preprocessing is differ by Models such as BERT, LUKE
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            labels: list[str],
            ent_start_token: str = '[START_ENT]',
            ent_end_token: str = '[END_ENT]',
            task_description: str = "Given a document, retrieve entity descriptions that are relevant to the mention located between special tokens '[START_ENT]' and '[END_ENT]'",
            remove_nil: bool = False
            ) -> None:
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.ent_start_token = ent_start_token
        self.ent_end_token = ent_end_token
        self.task_description = task_description
        self.remove_nil = remove_nil
        if "e5" in self.model_name:
            if 'instruct' in self.model_name:
                self.instruct = """Instruct: {task_description}\nQuery: {query}"""
            else:
                self.instruct = """query: {query}"""
        else:
            self.task_description = ""

    def __call__(self, document: list[dict[str, Any]]) -> Iterable[BatchEncoding]:
        """
        Input: list[dict[str, Any]]
        Output: Iterable[Dict[str, Any]]
        """
        if "luke" in self.model_name:
            for example in document:
                for ent in example["entities"]:
                    encodings  = self.tokenizer(
                        example["text"],
                        entity_spans=[(ent["start"], ent["end"])],
                        padding=True,
                        truncation=True
                    )
                    encodings["text"] = example["text"]
                    encodings["entity_span"] = (ent["start"], ent["end"])
                    encodings["id"] = example['paragraph-id']
                    try:
                        encodings["labels"] = [self.label2id[label] for label in ent["label"]]
                    except KeyError:
                        if self.remove_nil:
                            continue
                        else:
                            raise KeyError
                    yield encodings
        elif "e5" in self.model_name:
            for example in document:
                for ent in example["entities"]:
                    if 'instruct' in self.model_name:
                        text = self.instruct.format(task_description=self.task_description, query=example['text'][:ent["start"]])
                    else:
                        text = self.instruct.format(query=example['text'][:ent["start"]])
                    encodings  = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True
                    )
                    head = self.tokenizer.encode(text + self.ent_start_token, add_special_tokens=False)
                    mention = self.tokenizer.encode(example["text"][ent["start"]:ent["end"]], add_special_tokens=False)
                    tail = self.tokenizer.encode(self.ent_end_token + example["text"][ent["end"]:], add_special_tokens=False)

                    input_ids = head + mention + tail
                    text = text + self.ent_start_token + example["text"][ent["start"]:ent["end"]] + self.ent_end_token + example["text"][ent["end"]:]
                    encodings = self.tokenizer.prepare_for_model(input_ids, truncation=True, add_special_tokens=False)
                    encodings["text"] = text
                    encodings["entity_span"] = (len(head), len(head)+len(mention))
                    encodings["id"] = example['paragraph-id']
                    try:
                        encodings["labels"] = [self.label2id[label] for label in ent["label"]]
                    except KeyError:
                        if self.remove_nil:
                            continue
                        else:
                            raise KeyError
                    yield encodings
        else:
            for example in document:
                for ent in example["entities"]:
                    text = example["text"][:ent["start"]] + self.ent_start_token + example["text"][ent["start"]:ent["end"]] + self.ent_end_token + example["text"][ent["end"]:]
                    encodings  = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True
                    )
                    encodings["text"] = text
                    encodings["entity_span"] = (ent["start"], ent["end"])
                    encodings["id"] = example['paragraph-id']
                    encodings["labels"] = [self.label2id[label] for label in ent["label"]]

                    yield encodings


def get_splits(
        raw_datasets: DatasetDict,
        preprocessor: Preprocessor,
        training_arguments: Optional[TrainingArguments]=None
        ) -> dict[str, Dataset]:
    def preprocess(documents: Dataset) -> Any:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = next(iter(raw_datasets.values())).column_names
            splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)
    else:
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    return splits


