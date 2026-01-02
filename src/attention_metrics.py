import torch


def compute_attention_out(
    attentions,
    aggregate_layers=True,
    aggregate_heads=True
):
    """
    Calcula atenção emitida (attention_out).

    Mede o quanto cada token distribui atenção para os outros.

    Parâmetros
    ----------
    attentions : tuple(torch.Tensor)
        Tuple de tensores de atenção por layer.
        Cada tensor tem shape: (batch, heads, T, T)

    aggregate_layers : bool
        Se True, soma todas as layers.
        Se False, retorna por layer.

    aggregate_heads : bool
        Se True, soma todas as heads.
        Se False, retorna por head.

    Retorno
    -------
    torch.Tensor
        attention_out por token.
        Shape final depende das agregações.
    """

    # Stack: (L, batch, H, T, T)
    attn = torch.stack(attentions)

    # Remove batch (assumindo batch_size = 1)
    attn = attn[:, 0]  # (L, H, T, T)

    # Soma sobre j (tokens atendidos)
    # Resultado: (L, H, T)
    attention_out = attn.sum(dim=-1)

    if aggregate_heads:
        attention_out = attention_out.sum(dim=1)  # (L, T)

    if aggregate_layers:
        attention_out = attention_out.sum(dim=0)  # (T,)

    return attention_out


def compute_attention_in(
    attentions,
    aggregate_layers=True,
    aggregate_heads=True
):
    """
    Calcula atenção recebida (attention_in).

    Mede o quanto cada token é consultado pelos outros.

    Parâmetros
    ----------
    attentions : tuple(torch.Tensor)
        Tuple de tensores de atenção por layer.
        Cada tensor tem shape: (batch, heads, T, T)

    aggregate_layers : bool
        Se True, soma todas as layers.
        Se False, retorna por layer.

    aggregate_heads : bool
        Se True, soma todas as heads.
        Se False, retorna por head.

    Retorno
    -------
    torch.Tensor
        attention_in por token.
        Shape final depende das agregações.
    """

    # Stack: (L, batch, H, T, T)
    attn = torch.stack(attentions)

    # Remove batch
    attn = attn[:, 0]  # (L, H, T, T)

    # Soma sobre i (tokens que consultam)
    # Resultado: (L, H, T)
    attention_in = attn.sum(dim=-2)

    if aggregate_heads:
        attention_in = attention_in.sum(dim=1)  # (L, T)

    if aggregate_layers:
        attention_in = attention_in.sum(dim=0)  # (T,)

    return attention_in
