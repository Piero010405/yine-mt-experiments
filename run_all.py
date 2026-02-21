"""
Runs the full training and evaluation pipeline for both baseline and NSL models. This script loads the 
data, sets up the tokenizer and model, trains the models, evaluates on the dev and test sets, saves 
metrics and plots, and exports predictions. The results are saved in the experiments/ directory with a 
timestamped subdirectory for each run. Finally, it builds a summary comparison CSV for the test 
metrics of both models.
"""
import os
import gc
from pathlib import Path

import torch
import pandas as pd
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorForSeq2Seq
)

from src.utils import set_env_defaults, ensure_dir, load_json, save_json, now_tag
from src.data_build import load_positive, load_negatives, build_baseline_datasets, build_nsl_train_dataset
from src.tokenization import setup_tokenizer, preprocess_batch
from src.collator import NSLDataCollator
from src.metrics import build_compute_metrics
from src.plots import save_training_plots
from src.trainer_nsl import NSLTrainer


def free_gpu():
    """
    Free GPU memory by calling garbage collection and emptying the CUDA cache. This can help prevent 
    out-of-memory
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def export_predictions(trainer, tokenizer, dataset, out_csv: str, custom_tgt_token: str):
    """
    Exports predictions from the trainer on the given dataset to a CSV file. The predictions and     
    references are cleaned and saved to the specified output CSV file.
    """
    preds = trainer.predict(dataset)
    pred_ids = preds.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    decoded = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # decode references
    import numpy as np
    labels = preds.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def clean(s: str):
        """
        Cleans whitespace and removes the custom target token from the beginning of the string if
        it is present.
        """
        s = (s or "").strip()
        if s.startswith(custom_tgt_token):
            s = s[len(custom_tgt_token):].strip()
        return s

    decoded = [clean(x) for x in decoded]
    refs = [clean(x) for x in refs]

    # dataset has src_text if we keep it; otherwise we skip
    src = None
    if "src_text" in dataset.column_names:
        src = [str(x) for x in dataset["src_text"]]
    else:
        src = [""] * len(decoded)

    df = pd.DataFrame({"src": src, "reference": refs, "prediction": decoded})
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def run_experiment(cfg_path: str, mode: str):
    """
    mode: 'baseline' or 'nsl'
    """
    cfg = OmegaConf.load(cfg_path)
    exp_name = cfg.experiment_name
    tag = now_tag()
    out_dir = Path("experiments") / exp_name
    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")

    # save config used
    OmegaConf.save(cfg, str(out_dir / "config_used.yaml"))

    set_seed(int(cfg.seed))

    # Load data
    pos_df = load_positive("data/positive.parquet")
    split_json = load_json("data/split_v1.json")

    if mode == "baseline":
        dsd = build_baseline_datasets(pos_df, split_json)
    else:
        neg_df = load_negatives("data/negatives.parquet")
        dsd = build_nsl_train_dataset(pos_df, neg_df, split_json)

    # Tokenizer/model
    tokenizer = setup_tokenizer(
        cfg.model_name,
        cfg.src_lang,
        cfg.custom_tgt_token
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # fp16 for RTX 3060
    use_fp16 = torch.cuda.is_available()

    # Tokenize
    def tok_fn(examples):
        """
        Tokenizes a batch of examples using the provided tokenizer and configuration.
        """
        return preprocess_batch(tokenizer, examples, int(cfg.max_length), cfg.custom_tgt_token)

    remove_cols = dsd["train"].column_names
    tokenized = dsd.map(tok_fn, batched=True, remove_columns=remove_cols, desc=f"Tokenizing ({exp_name})")

    train_ds = tokenized["train"]
    dev_ds = tokenized["dev"]
    test_ds = tokenized["test"]

    # Collator
    if mode == "baseline":
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    else:
        collator = NSLDataCollator(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        lr_scheduler_type=str(cfg.lr_scheduler_type),
        warmup_ratio=float(cfg.warmup_ratio),
        num_train_epochs=float(cfg.num_train_epochs),
        label_smoothing_factor=float(cfg.label_smoothing_factor),
        predict_with_generate=True,
        generation_max_length=int(cfg.generation_max_length),
        generation_num_beams=int(cfg.generation_num_beams),
        fp16=use_fp16,
        bf16=False,
        logging_steps=int(cfg.logging_steps),
        save_total_limit=int(cfg.save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrfpp",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=int(cfg.seed),
    )

    compute_metrics = build_compute_metrics(tokenizer, cfg.custom_tgt_token)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=int(cfg.early_stopping_patience))]

    if mode == "baseline":
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
    else:
        trainer = NSLTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            alpha=float(cfg.alpha),
        )

    # Train
    baseline_before = trainer.evaluate()
    train_result = trainer.train()
    final_dev = trainer.evaluate()

    # Test eval
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # Save metrics
    metrics = {
        "baseline_before": baseline_before,
        "final_dev": final_dev,
        "test": test_metrics,
    }
    save_json(metrics, str(out_dir / "metrics.json"))

    # Save plots
    save_training_plots(trainer.state.log_history, str(out_dir / "plots"))

    # Export predictions on TEST (for thesis tables)
    # For export we need src_text, so we export from original dsd test split (not tokenized)
    # Rebuild plain test DF
    # baseline datasets kept src_text/tgt_text in dsd
    test_plain = dsd["test"]
    preds_csv = str(out_dir / "predictions_test.csv")
    export_predictions(trainer, tokenizer, test_plain, preds_csv, cfg.custom_tgt_token)

    # Save model
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    return metrics


def main():
    """
    Main entry point for running the experiments. This function runs both the baseline and NSL 
    experiments, saves the results, and builds a summary comparison CSV for the test metrics of both 
    models.
    """
    set_env_defaults()
    ensure_dir("experiments")

    print("=== Running BASELINE ===")
    m0 = run_experiment("configs/baseline.yaml", mode="baseline")
    free_gpu()

    print("=== Running NSL ===")
    m1 = run_experiment("configs/nsl.yaml", mode="nsl")
    free_gpu()

    # Build comparison table
    def pick_test(m):
        t = m["test"]
        return {
            "bleu": t.get("test_bleu", None),
            "chrf": t.get("test_chrf", None),
            "chrfpp": t.get("test_chrfpp", None),
            "ter": t.get("test_ter", None),
            "loss": t.get("test_loss", None),
        }

    b = pick_test(m0)
    n = pick_test(m1)

    comp = []
    comp.append({"model": "baseline", **b})
    comp.append({"model": "nsl", **n})

    # deltas
    delta = {"model": "delta(nsl-baseline)"}
    for k in ["bleu","chrf","chrfpp","ter","loss"]:
        try:
            delta[k] = (n[k] - b[k]) if (n[k] is not None and b[k] is not None) else None
        except Exception:
            delta[k] = None
    comp.append(delta)

    df = pd.DataFrame(comp)
    df.to_csv("experiments/summary_comparison.csv", index=False, encoding="utf-8")
    print("✅ Saved: experiments/summary_comparison.csv")

if __name__ == "__main__":
    main()
