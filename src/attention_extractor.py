# src/attention_extractor.py

import torch

def extract_attention(text, tokenizer, model):
    """
    Executa o modelo e retorna tokens + attention weights.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return {
        "tokens": tokens,
        "attentions": attentions
    }
