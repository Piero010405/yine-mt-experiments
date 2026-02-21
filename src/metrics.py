"""
Metrics for evaluation
"""
import numpy as np
import evaluate

_bleu = evaluate.load("sacrebleu")
_chrf = evaluate.load("chrf")
_ter  = evaluate.load("ter")

def build_compute_metrics(tokenizer, custom_tgt_token: str):
    """
    Builds a compute_metrics function for HuggingFace Trainer that decodes the predictions and labels, 
    removes the custom target token if present, and computes BLEU, ChrF, ChrF++, TER, and average 
    generated length. The metrics are returned in a dictionary format suitable for logging.
    """
    def compute_metrics(eval_pred):
        """
        Computes metrics for a batch of predictions and labels.
        """
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # strip + remove custom token if appears
        def clean(s: str):
            """
            Strips whitespace and removes the custom target token from the beginning of the string if 
            it is present.
            """
            s = (s or "").strip()
            if s.startswith(custom_tgt_token):
                s = s[len(custom_tgt_token):].strip()
            return s

        decoded_preds = [clean(x) for x in decoded_preds]
        decoded_labels = [clean(x) for x in decoded_labels]

        bleu = _bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["score"]
        chrf = _chrf.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels], word_order=0)["score"]
        chrfpp = _chrf.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels], word_order=2)["score"]
        ter = _ter.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["score"]

        gen_len = float(np.mean([len(x.split()) for x in decoded_preds])) if len(decoded_preds) else 0.0
        return {
            "bleu": float(bleu),
            "chrf": float(chrf),
            "chrfpp": float(chrfpp),
            "ter": float(ter),
            "gen_len": gen_len
        }
    return compute_metrics
