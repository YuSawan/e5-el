from typing import Any

import torch
import wandb
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .pooling import average_pool
from .retriever import DenseRetriever


@torch.no_grad()
def predict(model: PreTrainedModel, dataset: Dataset, retriever: DenseRetriever, reset_index: bool = False) -> list[dict[str, Any]]:
    if reset_index:
        retriever.build_index(model)
    dataloader = retriever.get_dataloader(dataset)
    pbar = tqdm(total=len(dataloader), desc='Predict')
    model.to(retriever.device)
    predict_distances: list[list[float]] = []
    predict_indices = []
    for batch, _ in dataloader:
        pbar.update()
        batch = batch.to(retriever.device)
        outputs = model(**batch)
        query = average_pool(outputs.last_hidden_state, batch.attention_mask).to('cpu').detach().numpy().copy()
        batch_dists, batch_indices = retriever.search_knn(query, top_k=5)
        predict_distances.extend(batch_dists)
        predict_indices.extend(batch_indices)
    pbar.close()

    dictionary = retriever.dictionary
    predicts = []

    pbar = tqdm(total=len(dataset), desc='Predict')
    assert len(dataset) == len(predict_distances) == len(predict_indices)
    for data, dists, indices in zip(dataset, predict_distances, predict_indices):
        pbar.update()
        pid = data['id']
        text = data['text']
        labels = data['labels']
        start, end = data['entity_span']
        if 'e5' in model.name_or_path:
            mention = retriever.tokenizer.decode(data['input_ids'][start: end])
        else:
            mention = text[start: end]
        results, golds = [], []
        for d, ind in zip(dists, indices):
            entry = dictionary(ind)
            results.append({
                "name": entry.name,
                "id": entry.id,
                'similarity': round(float(d), 4),
                "description": entry.description
            })
        for idx in labels:
            entry = dictionary[idx]
            golds.append(f"{entry.name}({entry.id})")
        predicts.append({"pid": pid, "text": text, "mention": mention, "gold": golds, "predict": results})
    pbar.close()

    return predicts


def submit_wandb_predict(predicts: list[dict[str, Any]]) -> None:
    columns = ["pid", "text", "mention", "gold", "rank1", "rank2","rank3", "rank4", "rank5", "rank1_desc", "rank2_desc", "rank3_desc", "rank4_desc", "rank5_desc"]
    result_table = wandb.Table(columns=columns)
    for p in predicts:
        result_table.add_data(
            p["pid"], p["text"], p["mention"], ', '.join(p['gold']),
            f"{p['predict'][0]['name']} ({p['predict'][0]['id']}; {p['predict'][0]['similarity']})",
            f"{p['predict'][1]['name']} ({p['predict'][1]['id']}; {p['predict'][1]['similarity']})",
            f"{p['predict'][2]['name']} ({p['predict'][2]['id']}; {p['predict'][2]['similarity']})",
            f"{p['predict'][3]['name']} ({p['predict'][3]['id']}; {p['predict'][3]['similarity']})",
            f"{p['predict'][4]['name']} ({p['predict'][4]['id']}; {p['predict'][4]['similarity']})",
            f"{p['predict'][0]['description']}", f"{p['predict'][1]['description']}", f"{p['predict'][2]['description']}", f"{p['predict'][3]['description']}", f"{p['predict'][4]['description']}"
        )
    wandb.log({"predictions": result_table})
