import json
import logging
import os

import torch
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from src.argparser import DatasetArguments, ModelArguments, parse_args
from src.data import (
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from src.eval import evaluate, submit_wandb_eval
from src.predict import predict, submit_wandb_predict
from src.retriever import DenseRetriever
from src.training import EntityLinkingTrainer, LoggerCallback, setup_logger

logger = logging.getLogger(__name__)
TOKEN = os.environ.get('TOKEN', True)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {data_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)
    if model_args.prev_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.prev_path)
        model = AutoModel.from_pretrained(model_args.prev_path)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name, token=TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, model_max_length=model_args.model_max_length, token=TOKEN)
        tokenizer.add_tokens([data_args.start_ent_token, data_args.end_ent_token, data_args.nil_label])
        model = AutoModel.from_pretrained(model_args.model_name, config=config, token=TOKEN)
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    cache_dir = model_args.cache_dir or get_temporary_cache_files_directory()
    dictionary = EntityDictionary(
        tokenizer=tokenizer,
        dictionary_path=data_args.dictionary_file,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"name": data_args.nil_label, "description": data_args.nil_description} if data_args.add_nil else None
    )

    raw_datasets = read_dataset(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        cache_dir
    )
    preprocessor = Preprocessor(
        tokenizer,
        dictionary.entity_ids,
        ent_start_token=data_args.start_ent_token,
        ent_end_token=data_args.end_ent_token,
        task_description=data_args.task_description,
        remove_nil=False if data_args.add_nil else True
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)

    if training_args.do_train:
        if model_args.negative == 'dense':
            retriever = DenseRetriever(
                tokenizer,
                dictionary,
                measure=model_args.measure,
                batch_size=training_args.eval_batch_size*2,
                top_k=model_args.top_k,
                vector_size=model.encoder.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
        elif model_args.negative == 'bm25':
            raise NotImplementedError
        else:
            assert model_args.negative == 'inbatch'
            retriever = None

        if retriever is not None:
            train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
            splits['train'] = splits['train'].add_column("negatives", train_candidate_ids)
            dev_candidate_ids = retriever.get_hard_negatives(model, splits['validation'], reset_index=False)
            splits['validation'] = splits['validation'].add_column("negatives", dev_candidate_ids)

        trainer = EntityLinkingTrainer(
            model = model,
            measure=model_args.measure,
            temperature=model_args.temperature,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = CollatorForEntityLinking(tokenizer, dictionary, negative_sample=model_args.negative)
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
            measure=model_args.measure,
            batch_size=training_args.eval_batch_size*2,
            top_k=model_args.top_k,
            vector_size=model.encoder.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        results = evaluate(model=model, dataset=splits['test'], retriever=retriever)
        if training_args.do_train:
            submit_wandb_eval(results)
        logger.info(f"R@1: {round(results['tp_1']/results['true'], 5)} ({results['tp_1']}/{results['true']})")
        logger.info(f"R@5: {round(results['tp_5']/results['true'], 5)} ({results['tp_5']}/{results['true']})")
        logger.info(f"R@10: {round(results['tp_10']/results['true'], 5)} ({results['tp_10']}/{results['true']})")
        logger.info(f"R@20: {round(results['tp_20']/results['true'], 5)} ({results['tp_20']}/{results['true']})")
        logger.info(f"R@50: {round(results['tp_50']/results['true'], 5)} ({results['tp_50']}/{results['true']})")
        logger.info(f"R@100: {round(results['tp_100']/results['true'], 5)} ({results['tp_100']}/{results['true']})")

    if training_args.do_predict:
        if not training_args.do_eval:
            retriever = DenseRetriever(
                tokenizer,
                dictionary,
                measure=model_args.measure,
                batch_size=training_args.eval_batch_size*2,
                top_k=model_args.top_k,
                vector_size=model.encoder.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
        assert isinstance(retriever, DenseRetriever)
        predicts = predict(
            model=model,
            dataset=splits['validation'].remove_columns('negatives') if model_args.negative != 'inbatch' else splits['validation'],
            retriever=retriever,
            reset_index=False if training_args.do_eval else True
        )
        if training_args.do_train:
            submit_wandb_predict(predicts)
        with open(os.path.join(training_args.output_dir, "predicts.jsonl"), 'w') as f:
            for p in predicts:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")


def cli_main() -> None:
    args, model_args, training_args = parse_args()
    if args.validation_file is None:
        training_args.eval_strategy = "no"
    main(args, model_args, training_args)


if __name__ == '__main__':
    cli_main()
