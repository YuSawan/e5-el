from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchEncoding, EvalPrediction, PreTrainedModel, Trainer
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_pt_utils import nested_detach

from ..pooling import average_pool


@dataclass
class EntityLinkingOutput(ModelOutput):
    """
    Outputs of bi-encoder entity linking models.

    Args:
        loss (`torch.Tensor` of shape `(1,)`):
            Classification loss.
        logits (`torch.Tensor` of shape `(batch_size, batch_size + config.negative_num)`):
            Classification scores (before SoftMax).
        mention_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True`):
            `torch.Tensor` (one for the output of the mention_encoder) of shape `(batch_size, hidden_size)`.
        candidate_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True`):
            `torch.Tensor` (one for the output of the entity_encoder) of shape `(batch_size, batch_size + config.negative_num, hidden_size)`.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    mention_hidden_states: Optional[torch.Tensor] = None
    candidate_hidden_states: Optional[torch.Tensor] = None


class EntityLinkingTrainer(Trainer):
    def __init__(self, measure: str = "cos", temperature: float = 0.01, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)
        self.measure = measure
        self.temperature = temperature

    def _compute_metrics(self, p: EvalPrediction) -> dict[str, Any]:
        self.last_prediction = p
        return _compute_metrics(p, self.label_names)

    def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: BatchEncoding,
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None
        ) -> Union[torch.Tensor, tuple[torch.Tensor, EntityLinkingOutput]]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = _compute_loss(model, inputs, self.measure, self.temperature)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = _compute_loss(model, inputs, self.measure, self.temperature)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        detached_logits = nested_detach(logits)
        if len(detached_logits) == 1:
            detached_logits = detached_logits[0]
        labels = inputs.get('labels')

        return (loss, detached_logits, labels)


def _compute_metrics(p: EvalPrediction, label_names: list[str]) -> dict[str, Any]:
    _, scores = p.predictions
    preds = scores.argmax(axis=1).ravel()
    labels = p.label_ids.ravel()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    num_corrects = (preds == labels).sum().item()
    num_golds = mask.sum().item()
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")

    return {"recall": recall}


def _compute_loss(model: PreTrainedModel, inputs: BatchEncoding, measure: str, temperature: float) -> EntityLinkingOutput:
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
    labels = inputs.get('labels')

    hard_negatives_input_ids = inputs.get("hard_negatives_input_ids", None)
    hard_negatives_attention_mask = inputs.get("hard_negatives_attention_mask", None)
    if hard_negatives_input_ids is not None and hard_negatives_attention_mask is not None:
        hard_negatives = model(input_ids=hard_negatives_input_ids, attention_mask=hard_negatives_attention_mask)
        hard_negatives = average_pool(hard_negatives.last_hidden_state, hard_negatives_attention_mask)
        hard_negatives = hard_negatives.reshape([bs, -1, hs])
        candidates = torch.concat([candidates, hard_negatives], dim=1)

    if measure == 'ip':
        scores = torch.bmm(queries.unsqueeze(1), candidates.transpose(1, -1)).squeeze(1)
    elif measure == 'cos':
        queries_norm = queries.unsqueeze(1) / torch.norm(queries.unsqueeze(1), dim=2, keepdim=True)
        candidates_norm = candidates / torch.norm(candidates, dim=2, keepdim=True)
        scores = torch.bmm(queries_norm, candidates_norm.transpose(1, -1)).squeeze(1)
    else:
        scores = torch.cdist(queries.unsqueeze(1), candidates).squeeze(1)

    loss = nn.functional.cross_entropy(scores / temperature, labels, reduction="mean")

    return EntityLinkingOutput(loss=loss, logits=scores)
