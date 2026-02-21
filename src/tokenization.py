"""
Tokenization utilities for the Yine MT experiments. This includes setting up the tokenizer, adding 
custom
"""
from transformers import AutoTokenizer

def setup_tokenizer(model_name: str, src_lang: str, custom_tgt_token: str):
    """
    Sets up the tokenizer for the given model name and source language. It also adds a custom target
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.src_lang = src_lang

    # add custom target token if not present
    if custom_tgt_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [custom_tgt_token]})
    return tokenizer

def maybe_init_custom_token_embeddings(
        model,
        tokenizer,
        custom_tgt_token: str,
        proxy_lang_token: str
    ):
    """
    Initializes the custom token embedding weights using a proxy token embedding if available.
    """
    custom_id = tokenizer.convert_tokens_to_ids(custom_tgt_token)
    proxy_id = tokenizer.convert_tokens_to_ids(proxy_lang_token)

    # resize embeddings if new tokens were added
    model.resize_token_embeddings(len(tokenizer))

    # try to copy proxy embeddings
    try:
        if proxy_id is not None and proxy_id != tokenizer.unk_token_id:
            with torch.no_grad():
                if (
                    hasattr(model, "model") and
                    hasattr(model.model, "encoder") and
                    hasattr(model.model.encoder, "embed_tokens")
                    ):
                    encoder_embed_tokens = model.model.encoder.embed_tokens.weight[proxy_id].clone()
                    model.model.encoder.embed_tokens.weight[custom_id] = encoder_embed_tokens
                if (
                    hasattr(model, "model") and
                    hasattr(model.model, "decoder") and
                    hasattr(model.model.decoder, "embed_tokens")
                ):
                    decoder_embed_tokens = model.model.decoder.embed_tokens.weight[proxy_id].clone()
                    model.model.decoder.embed_tokens.weight[custom_id] = decoder_embed_tokens
    except Exception:
        # if anything fails, we keep random init (still works, just a bit noisier)
        pass

def preprocess_batch(tokenizer, examples, max_length: int, custom_tgt_token: str):
    """
    Preprocesses a batch of examples by tokenizing the source and target texts. The target texts are
    prepended with a custom target token to steer the decoder. The tokenized inputs and labels are returned in a format suitable for 
    model training. Optional fields for 
    NSL (is_negative, severity) are also propagated if present in the examples.
    """
    src = [str(x) for x in examples["src_text"]]
    # prepend target token to steer decoder
    tgt = [f"{custom_tgt_token} {str(x)}" for x in examples["tgt_text"]]

    model_inputs = tokenizer(
        src,
        max_length=max_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=tgt,
        max_length=max_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    # propagate optional fields for NSL
    if "is_negative" in examples:
        model_inputs["is_negative"] = examples["is_negative"]
    if "severity" in examples:
        model_inputs["severity"] = examples["severity"]

    return model_inputs
