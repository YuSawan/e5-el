# import json
import json
import os
from logging import INFO, getLogger
from typing import Optional

import faiss
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments

from .data import Collator, EntityDictionary
from .pooling import average_pool

logger = getLogger(__name__)
logger.setLevel(INFO)


class DenseRetriever:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            dictionary: EntityDictionary,
            measure: str,
            batch_size: int,
            top_k: int,
            vector_size: int,
            device: torch.device,
            training_args: TrainingArguments,
        ) -> None:
            self.tokenizer = tokenizer
            self.dictionary = dictionary
            self.measure = measure
            self.batch_size = batch_size
            self.top_k = top_k
            self.device = device
            self.vector_size = vector_size
            if self.measure not in ["cos", "ip", "l2"]:
                raise NotImplementedError(f"{measure} is not supported")
            if self.top_k <= 0:
                raise RuntimeError("K is zero or under zero.")
            if self.top_k >= len(self.dictionary):
                raise RuntimeError("K is same or over the size of dictionary")
            if self.measure == 'l2':
                self.index = faiss.IndexFlatL2(vector_size)
            else:
                self.index = faiss.IndexFlatIP(vector_size)
            self.meta_ids_to_keys: dict[int, str] = {}
            self.args = training_args

    def reset_index(self) -> None:
        self.index.reset()
        self.meta_ids_to_keys = {}

    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        collator = Collator(self.tokenizer)
        dataloader_params = {
            "batch_size": self.batch_size,
            "collate_fn": collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        dataloader_params["sampler"] = SequentialSampler(dataset)
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(dataset, **dataloader_params)

    @torch.no_grad()
    def build_index(self, model: PreTrainedModel) -> None:
        self.reset_index()
        dataloader = self.get_dataloader(self.dictionary.entity_dict)
        self.meta_ids_to_keys = {i: idx for i, idx in enumerate(self.dictionary.entity_ids)}

        model.eval()
        model.to(self.device)
        pbar = tqdm(total=len(dataloader), desc='Hard Negative(Load Entity Emb)')
        for batch in dataloader:
            pbar.update()
            batch = batch.to(self.device)
            outputs = model(**batch)
            entity_embedding = average_pool(outputs.last_hidden_state, batch.attention_mask).to('cpu').detach().numpy().copy()
            if self.measure == 'cos':
                faiss.normalize_L2(entity_embedding)
            self.index.add(entity_embedding)
        pbar.close()

    def search_knn(self, query_vectors: np.ndarray, top_k: Optional[int] = None) -> tuple[np.ndarray, list[list[str]]]:
        K = self.top_k if not top_k else top_k
        if K <= 0:
            raise RuntimeError("K is zero or under zero.")
        if K > len(self.dictionary):
            K = len(self.dictionary)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.dictionary)}")
        if self.measure == "cos":
            faiss.normalize_L2(query_vectors)
        distances, indices = self.index.search(query_vectors, K)
        indices_keys = []
        for indice in indices:
            indices_keys.append([self.meta_ids_to_keys[ind] for ind in indice])

        return distances, indices_keys

    @torch.no_grad()
    def get_hard_negatives(self, model: PreTrainedModel, dataset: Dataset, reset_index: bool = True) -> list[list[str]]:
        model.to(self.device)
        if reset_index:
            self.build_index(model)
        dataloader = self.get_dataloader(dataset)

        model.eval()
        pbar = tqdm(total=(len(dataloader)), desc='Hard Negative(Search)')
        candidate_ids = []
        for batch, labels in dataloader:
            pbar.update()
            batch = batch.to(self.device)
            outputs = model(**batch)
            outputs = average_pool(outputs.last_hidden_state, batch.attention_mask).to('cpu').detach().numpy().copy()
            _, batch_indices = self.search_knn(outputs, top_k=self.top_k+1)
            for idxs, indices in zip(labels, batch_indices):
                if self.dictionary[idxs[0]].id in indices:
                    indices.remove(self.dictionary[idxs[0]].id)
                    candidate_ids.append(indices)
                else:
                    candidate_ids.append(indices[:self.top_k])
        pbar.close()
        return candidate_ids

    def dump(self, model: PreTrainedModel, index_path: str, ensure_ascii: bool = False) -> None:
        self.build_index(model)
        self.serialize(index_path, ensure_ascii)

    def serialize(self, index_path: str, ensure_ascii: bool = False) -> None:
        logger.info("Serializing index to %s", index_path)
        if os.path.isdir(index_path):
            index_file = os.path.join(index_path, "index.dpr")
            meta_file = os.path.join(index_path, "meta.json")
        else:
            index_file = index_path + ".index.dpr"
            meta_file = index_path + ".meta.json"
        faiss.write_index(self.index, index_file)
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def deserialize_from(self, index_path: str) -> None:
        if os.path.isdir(index_path):
            index_file = os.path.join(index_path, "index.dpr")
            meta_file = os.path.join(index_path, "meta.json")
        else:
            index_file = index_path + ".index.dpr"
            meta_file = index_path + ".meta.json"

        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        self.meta_ids_to_keys = {int(k): v for k, v in json.load(open(meta_file)).items()}
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

    def index_exists(self, path: str) -> bool:
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "meta.json")
        else:
            index_file = path + ".index.dpr"
            meta_file = path + ".meta.json"
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def __len__(self) -> int:
        return self.index.ntotal
