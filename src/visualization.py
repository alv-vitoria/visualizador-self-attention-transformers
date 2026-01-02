# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attentions, tokens, layer, head):
    """
    Plota o mapa de atenção de uma camada e head específica.
    """
    attn_matrix = attentions[layer][0, head].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis"
    )

    plt.title(f"Camada {layer} | Head {head}")
    plt.xlabel("Token atendido")
    plt.ylabel("Token consultando")
    plt.tight_layout()
    plt.show()
