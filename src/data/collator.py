from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

from .dictionary import EntityDictionary


@dataclass
class Collator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]|tuple[dict[str, Any], list[list[int]]]:
        features = [f.copy() for f in features]
        if "labels" in list(features[0].keys()):
            labels = []
            for k, f in enumerate(features):
                _ = f.pop('text')
                _ = f.pop('entity_span')
                _ = f.pop("id")
                labels.append(f.pop("labels"))
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            return batch, labels
        else:
            encodings = [f.pop("encodings") for f in features]
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                encodings,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            return batch


@dataclass
class CollatorForEntityLinking:
    tokenizer: PreTrainedTokenizerBase
    dictionary: EntityDictionary
    negative_sample: str = "inbatch"

    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str | None = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [f.copy() for f in features]
        inbatch_encodings, negative_encodings = [], []
        for f in features:
            _ = f.pop('text')
            _ = f.pop('entity_span')
            _ = f.pop("id")
            labels = f.pop("labels")
            inbatch_encodings.append(self.dictionary[labels[0]].encoding)
            if self.negative_sample != "inbatch":
                negatives = f.pop("negatives")
                negative_encodings.extend([self.dictionary(n).encoding for n in negatives])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        inbatch_candidates = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            inbatch_encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update({f'candidates_{k}': v for k, v in inbatch_candidates.items()})

        if self.negative_sample != 'inbatch':
            negative_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                negative_encodings,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch.update({f'hard_negatives_{k}': v for k, v in negative_batch.items()})

        batch["labels"] = torch.arange(len(features)) if self.return_tensors == "pt" else [_ for _ in range(len(features))]

        return batch
