import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from .data import (
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from .eval import evaluate, submit_wandb_eval
from .predict import predict, submit_wandb_predict
from .retriever import DenseRetriever
from .training import EntityLinkingTrainer, LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    dictionary_file: str
    train_file : str
    validation_file : str
    test_file: str
    model_name: str = "intfloat/multilingual-e5-small"
    cache_dir: Optional[str] = None
    max_context_length: int = 512
    max_entity_length: int = 32
    max_mention_length: int = 30
    measure: str = 'cos'
    negative: str = 'inbatch'
    add_nil: bool = False
    top_k: int = 10


def main(args: Arguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(args.model_name)
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(['<company>', '</company>'])
    model = AutoModel.from_pretrained(args.model_name, config=config)
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    dictionary = EntityDictionary(
        tokenizer=tokenizer,
        dictionary_path=args.dictionary_file,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"description": "該当するエンティティが存在しません。"} if args.add_nil else None
    )

    raw_datasets = read_dataset(
        args.train_file,
        args.validation_file,
        args.test_file,
        cache_dir
    )
    preprocessor = Preprocessor(
        tokenizer,
        dictionary.entity_ids,
        args.max_entity_length,
        args.max_mention_length,
        ent_start_token='<company>',
        ent_end_token='</company>',
        task_description="Given a Japanese news article, retrieve entity descriptions that are relevant to the mention located between special tokens '<company>' and '</company>'",
        remove_nil=False if args.add_nil else True
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)

    if training_args.do_train:
        if args.negative == 'dense':
            retriever = DenseRetriever(
                tokenizer,
                dictionary,
                measure=args.measure,
                batch_size=training_args.eval_batch_size*2,
                top_k=args.top_k,
                vector_size=model.encoder.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
        elif args.negative == 'bm25':
            raise NotImplementedError
        else:
            assert args.negative == 'inbatch'
            retriever = None

        if retriever is not None:
            train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
            splits['train'] = splits['train'].add_column("negatives", train_candidate_ids)
            dev_candidate_ids = retriever.get_hard_negatives(model, splits['validation'], reset_index=False)
            splits['validation'] = splits['validation'].add_column("negatives", dev_candidate_ids)

        trainer = EntityLinkingTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = CollatorForEntityLinking(tokenizer, dictionary, negative_sample=args.negative)
        )
        trainer.add_callback(LoggerCallback(logger))

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        retriever = DenseRetriever(
            tokenizer,
            dictionary,
            measure=args.measure,
            batch_size=training_args.eval_batch_size*2,
            top_k=args.top_k,
            vector_size=model.encoder.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        results = evaluate(model=model, dataset=splits['test'], retriever=retriever)
        if training_args.do_train:
            submit_wandb_eval(results)
        logger.info(f"R@1: {round(results['tp_1']/results['true'], 5)} ({results['tp_1']}/{results['true']})")
        logger.info(f"R@10: {round(results['tp_10']/results['true'], 5)} ({results['tp_10']}/{results['true']})")
        logger.info(f"R@20: {round(results['tp_20']/results['true'], 5)} ({results['tp_20']}/{results['true']})")
        logger.info(f"R@50: {round(results['tp_50']/results['true'], 5)} ({results['tp_50']}/{results['true']})")
        logger.info(f"R@100: {round(results['tp_100']/results['true'], 5)} ({results['tp_100']}/{results['true']})")

    if training_args.do_predict:
        if not training_args.do_eval:
            retriever = DenseRetriever(
                tokenizer,
                dictionary,
                measure=args.measure,
                batch_size=training_args.eval_batch_size*2,
                top_k=args.top_k,
                vector_size=model.encoder.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
        assert isinstance(retriever, DenseRetriever)
        predicts = predict(
            model=model,
            dataset=splits['validation'].remove_columns('negatives') if args.negative != 'inbatch' else splits['validation'],
            retriever=retriever,
            reset_index=False if training_args.do_eval else True
        )
        if training_args.do_train:
            submit_wandb_predict(predicts)
        with open(os.path.join(training_args.output_dir, "predicts.jsonl"), 'w') as f:
            for p in predicts:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")


if __name__ == '__main__':
    CONFIG_FILE = Path(__file__).parents[2] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    if args.validation_file is None:
        training_args.eval_strategy = "no"

    main(args, training_args)
