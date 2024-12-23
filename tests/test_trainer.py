from importlib.resources import files

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
)

import tests.test_data as test_data
from src.data import (
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from src.pooling import average_pool
from src.retriever import DenseRetriever

model_name = "intfloat/multilingual-e5-small"
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(['[START_ENT]', '[END_ENT]'])
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
training_args = TrainingArguments(output_dir=".tmp/")
dictionary = EntityDictionary(
    tokenizer=tokenizer,
    dictionary_path=dictionary_path,
    nil={"description": "該当するエンティティが存在しません。"}
)
preprocessor = Preprocessor(
    tokenizer,
    dictionary.entity_ids,
)

@pytest.mark.parametrize('sampling', ['inbatch', 'dense'])
@pytest.mark.parametrize('measure', ['ip', 'cos', 'l2'])
def test_compute_loss(sampling: str, measure: str) -> None:
    raw_datasets = read_dataset(train_file=dataset_path)
    splits = get_splits(raw_datasets, preprocessor, training_args)
    if sampling == 'dense':
        retriever = DenseRetriever(
            tokenizer,
            dictionary,
            measure=measure,
            batch_size=2,
            top_k=1,
            vector_size=model.config.hidden_size,
            device=torch.device('cpu'),
            training_args=training_args
        )
        train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
        splits['train'] = splits['train'].add_column("negatives", train_candidate_ids)

    collator = CollatorForEntityLinking(tokenizer, dictionary, negative_sample=sampling)
    dataloader_params = {
        "batch_size": 2,
        "collate_fn": collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
    }
    dataloader_params["sampler"] = SequentialSampler(splits['train'])
    dataloader_params["drop_last"] = training_args.dataloader_drop_last
    dataloader_params["prefetch_factor"] = training_args.dataloader_prefetch_factor
    dataloader = DataLoader(splits['train'], **dataloader_params)

    for inputs in dataloader:
        inputs_ids = inputs.pop('input_ids')
        attention_mask = inputs.pop('attention_mask')
        candidates_input_ids = inputs.pop('candidates_input_ids')
        candidates_attention_mask = inputs.pop('candidates_attention_mask')
        queries = model(input_ids=inputs_ids, attention_mask=attention_mask)
        queries = average_pool(queries.last_hidden_state, attention_mask)
        candidates = model(input_ids=candidates_input_ids, attention_mask=candidates_attention_mask)
        candidates = average_pool(candidates.last_hidden_state, candidates_attention_mask)
        bs, hs = candidates.size(0), candidates.size(-1)
        candidates = candidates.unsqueeze(0).repeat(bs, 1, 1)
        labels = inputs.pop('labels')
        assert isinstance(labels, Tensor)
        assert labels.size(0) == 2
        assert queries.size() == (2, 384)
        assert candidates.size() == (2, 2, 384)

        hard_negatives_input_ids = inputs.get('hard_negatives_input_ids', None)
        hard_negatives_attention_mask = inputs.get('hard_negatives_attention_mask', None)
        if sampling == 'inbatch':
            assert hard_negatives_input_ids is None
            assert hard_negatives_attention_mask is None
        else:
            assert hard_negatives_input_ids.size(0) == 2
            assert hard_negatives_attention_mask.size(0) == 2

        if hard_negatives_input_ids is not None and hard_negatives_attention_mask is not None:
            hard_negatives = model(input_ids=hard_negatives_input_ids, attention_mask=hard_negatives_attention_mask)
            hard_negatives = average_pool(hard_negatives.last_hidden_state, hard_negatives_attention_mask)
            hard_negatives = hard_negatives.reshape([bs, -1, hs])
            candidates = torch.concat([candidates, hard_negatives], dim=1)
            assert candidates.size() == (2, 3, 384)

        if measure == 'ip':
            scores = torch.bmm(queries.unsqueeze(1), candidates.transpose(1, -1)).squeeze(1)
        elif measure == 'cos':
            queries_norm = queries.unsqueeze(1) / torch.norm(queries.unsqueeze(1), dim=2, keepdim=True)
            candidates_norm = candidates / torch.norm(candidates, dim=2, keepdim=True)
            scores = torch.bmm(queries_norm, candidates_norm.transpose(1, -1)).squeeze(1)
        else:
            scores = torch.cdist(queries.unsqueeze(1), candidates).squeeze(1)

        loss = nn.functional.cross_entropy(scores, labels, reduction="mean")
        assert isinstance(loss, Tensor)
        assert isinstance(scores, Tensor)
