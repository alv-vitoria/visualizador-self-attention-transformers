def compare_token_attention_simple(
    attentions_a,
    attentions_b,
    tokens_a,
    tokens_b,
    layer,
    head
):
    """
    Comparação exploratória entre duas frases usando:
    - atenção recebida (importância contextual)
    - atenção emitida (atividade relacional)

    OBS:
    - Não faz alinhamento de tokens
    - Assume mesma tokenização ou análise manual posterior
    """

    attn_a = attentions_a[layer][0, head].cpu().numpy()
    attn_b = attentions_b[layer][0, head].cpu().numpy()

    # Atenção recebida (importance)
    attention_in_a = attn_a.sum(axis=0)
    attention_in_b = attn_b.sum(axis=0)

    # Atenção emitida (activity)
    attention_out_a = attn_a.sum(axis=1)
    attention_out_b = attn_b.sum(axis=1)

    delta_in = attention_in_a - attention_in_b
    delta_out = attention_out_a - attention_out_b

    ranking_in_a = sorted(
        zip(tokens_a, attention_in_a),
        key=lambda x: x[1],
        reverse=True
    )

    ranking_in_b = sorted(
        zip(tokens_b, attention_in_b),
        key=lambda x: x[1],
        reverse=True
    )

    delta_ranking = sorted(
        zip(tokens_a, delta_in),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "ranking_in_a": ranking_in_a,
        "ranking_in_b": ranking_in_b,
        "delta_in": delta_ranking,
        "attention_out_a": list(zip(tokens_a, attention_out_a)),
        "attention_out_b": list(zip(tokens_b, attention_out_b)),
        "delta_out": list(zip(tokens_a, delta_out)),
    }
