from importlib.resources import files

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
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
from src.retriever import DenseRetriever

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))


class TestPreprocessor:
    @pytest.mark.parametrize("model_name", ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large-instruct"])
    def test___init__(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(['[START_ENT]', '[END_ENT]'])
        dictionary = EntityDictionary(tokenizer, dictionary_path, nil={"name": "[NIL]"})

        preprocessor = Preprocessor(
            tokenizer,
            dictionary.entity_ids,
            max_entity_length=32,
            max_mention_length=30,
        )
        assert preprocessor.tokenizer == tokenizer
        assert preprocessor.max_entity_length == 32
        assert preprocessor.max_mention_length == 30
        assert preprocessor.labels == dictionary.entity_ids
        assert len(preprocessor.label2id.keys()) == 6
        assert len(preprocessor.id2label.keys()) == 6


    @pytest.mark.parametrize("model_name", ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large-instruct"])
    @pytest.mark.parametrize("remove_nil", [True, False])
    def test___call__(self, model_name: str, remove_nil: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(['[START_ENT]', '[END_ENT]'])

        dictionary = EntityDictionary(tokenizer, dictionary_path, nil = None if remove_nil else {"id": "-1"})
        preprocessor = Preprocessor(
            tokenizer,
            dictionary.entity_ids,
            max_entity_length=32,
            max_mention_length=30,
            remove_nil=remove_nil
        )

        raw_dataset = read_dataset(test_file=dataset_path)
        for document in raw_dataset['test']["examples"]:
            for example in document:
                for ent in example['entities']:
                    if preprocessor.model_name.endswith('-instruct'):
                        text = preprocessor.instruct.format(task_description=preprocessor.task_description, query=example['text'][:ent["start"]])
                        assert text.startswith("Instruct: Given a document, retrieve entity descriptions that are relevant to the mention located between special tokens '[START_ENT]' and '[END_ENT]'\nQuery: ")
                    else:
                        text = preprocessor.instruct.format(query=example['text'][:ent["start"]])
                        assert text.startswith('query: ')
                    head = tokenizer.encode(text + preprocessor.ent_start_token, add_special_tokens=False)
                    mention = tokenizer.encode(example["text"][ent["start"]:ent["end"]], add_special_tokens=False)
                    tail = tokenizer.encode(preprocessor.ent_end_token + example["text"][ent["end"]:], add_special_tokens=False)
                    input_ids = head + mention + tail
                    encodings = tokenizer.prepare_for_model(input_ids, truncation=True, add_special_tokens=False)

                    assert tokenizer.convert_ids_to_tokens(input_ids[len(head): len(head)+len(mention)]) == tokenizer.tokenize(example["text"][ent["start"]:ent["end"]])
                    assert isinstance(encodings, BatchEncoding)

        features: list[BatchEncoding] = []
        for document in raw_dataset['test']["examples"]:
            features.extend(preprocessor(document))

        assert isinstance(features, list)
        if remove_nil:
            assert len(features) == 5
        else:
            assert len(features) == 8
        assert isinstance(features[0], BatchEncoding)

        outputs: dict = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]

        assert "input_ids" in outputs.keys()
        assert "labels" in outputs.keys()
        assert "id" in outputs.keys()
        assert "entity_span" in outputs.keys()


@pytest.mark.parametrize("sampling", ["inbatch", "dense"])
def test_CollatorForEntityLinking(sampling: str) -> None:
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['[NIL]'])
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    training_args = TrainingArguments(output_dir=".tmp/")

    dictionary = EntityDictionary(
        tokenizer=tokenizer,
        dictionary_path=dictionary_path,
        nil={"description": "該当するエンティティが存在しません。"}
    )
    raw_datasets = read_dataset(train_file=dataset_path)
    preprocessor = Preprocessor(
        tokenizer,
        dictionary.entity_ids,
        max_entity_length=32,
        max_mention_length=30,
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)

    if sampling == 'dense':
        retriever = DenseRetriever(
            tokenizer,
            dictionary,
            measure="cos",
            batch_size=1,
            top_k=3,
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

    for batch in dataloader:
        assert isinstance(batch, BatchEncoding)
        keys = list(batch.keys())
        assert "input_ids" in keys and isinstance(batch['input_ids'], torch.Tensor)
        assert "attention_mask" in keys and isinstance(batch['attention_mask'], torch.Tensor)
        assert "candidates_input_ids" in keys and isinstance(batch['candidates_input_ids'], torch.Tensor)
        assert "candidates_attention_mask" in keys and isinstance(batch['candidates_attention_mask'], torch.Tensor)
        assert "labels" in keys and isinstance(batch['labels'], torch.Tensor)
        assert batch['input_ids'].size(0) == batch['attention_mask'].size(0) == 2
        assert batch['candidates_input_ids'].size(0) == batch['candidates_attention_mask'].size(0) == 2
        assert batch['labels'].size(0) == 2

        hard_negatives_input_ids = batch.get("hard_negatives_input_ids", None)
        hard_negatives_attention_mask = batch.get("hard_negatives_attention_mask", None)
        if sampling != "inbatch":
            assert hard_negatives_input_ids is not None
            assert hard_negatives_attention_mask is not None
            assert isinstance(hard_negatives_input_ids, torch.Tensor)
            assert isinstance(hard_negatives_attention_mask, torch.Tensor)
            assert hard_negatives_input_ids.size(0) == hard_negatives_attention_mask.size(0) == 6
        else:
            assert hard_negatives_input_ids is None
            assert hard_negatives_attention_mask is None
