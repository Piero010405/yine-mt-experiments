"""
Trainer class for NSL-style training. This extends the Seq2SeqTrainer from HuggingFace Transformers 
and overrides the compute_loss method to apply different weighting to negative examples based on 
their severity. 
The alpha parameter controls how much the loss for negative examples is scaled down.
"""
import torch
from transformers import Seq2SeqTrainer

class NSLTrainer(Seq2SeqTrainer):
    """
    Efficient NSL-style:
    - If example is negative: loss *= alpha * severity
    - If positive: standard loss
    """
    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for a batch of inputs. It expects the inputs to contain:
        - labels: [bs, seq]
        - is_negative: [bs] (0 or 1)
        - severity: [bs] (float, e.g. 0.0 to 1.0)
        """
        labels = inputs.get("labels")
        is_negative = inputs.pop("is_negative", None)
        severity = inputs.pop("severity", None)

        outputs = model(**inputs)
        logits = outputs.logits  # [bs, seq, vocab]

        # per-token CE, then reduce to per-example
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        vocab = logits.size(-1)

        loss_tokens = loss_fct(logits.view(-1, vocab), labels.view(-1)).view(labels.size(0), -1)

        # mask padding tokens
        mask = (labels != -100).float()
        loss_sum = (loss_tokens * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        loss_per_example = loss_sum / denom  # [bs]

        if is_negative is not None and severity is not None:
            weights = torch.ones_like(loss_per_example)
            neg_mask = (is_negative == 1).float()
            # weights = 1 for positive, alpha*severity for negative
            weights = (
                    (1.0 - neg_mask) +
                    neg_mask *
                    (self.alpha * severity.to(loss_per_example.device))
                )
            loss = (loss_per_example * weights).mean()
        else:
            loss = loss_per_example.mean()

        return (loss, outputs) if return_outputs else loss
