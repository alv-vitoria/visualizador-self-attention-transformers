import plotly.express as px
import numpy as np


def plot_attention_interactive(
    attentions,
    tokens,
    layer,
    head,
    title_prefix="Mapa de Atenção"
):
    """
    Visualização interativa do mapa de self-attention usando Plotly.

    Parâmetros:
    - attentions: lista de tensores de atenção (uma por layer)
    - tokens: lista de tokens da frase
    - layer: índice da camada (int)
    - head: índice da head (int)

    Observações:
    - Usa apenas batch_size = 1
    - Não altera os valores de atenção, apenas visualiza
    - Eixos seguem a convenção:
        linha (y) = token consultando
        coluna (x) = token atendido
    """

    # Extrai a matriz de atenção da layer/head desejadas
    # Formato esperado: (batch, head, seq_len, seq_len)
    attn_matrix = attentions[layer][0, head]

    # Converte para numpy (Plotly trabalha melhor assim)
    attn_matrix = attn_matrix.detach().cpu().numpy()

    # Cria o heatmap interativo
    fig = px.imshow(
        attn_matrix,
        x=tokens,
        y=tokens,
        color_continuous_scale="Viridis",
        labels={
            "x": "Token atendido",
            "y": "Token consultando",
            "color": "Peso de atenção"
        },
        title=f"{title_prefix} | Layer {layer} | Head {head}"
    )

    # Ajustes visuais para leitura didática
    fig.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=700
    )

    fig.show()
