"""
Custom data collator for NSL
"""
import torch
from transformers import DataCollatorForSeq2Seq

class NSLDataCollator(DataCollatorForSeq2Seq):
    """
    Extends HF DataCollatorForSeq2Seq to also collate:
    - is_negative (0/1)
    - severity (float)
    """
    def __call__(self, features, return_tensors=None):
        # extract extra fields before parent collator removes them
        is_neg = [f.get("is_negative", 0) for f in features]
        sev = [f.get("severity", 1.0) for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        batch["is_negative"] = torch.tensor(is_neg, dtype=torch.long)
        batch["severity"] = torch.tensor(sev, dtype=torch.float32)
        return batch
