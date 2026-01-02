"""
Visualização e análise de self-attention em BERT (português).

Este script:
- carrega um modelo Transformer real
- extrai attention weights
- visualiza uma head específica
- calcula relevância de tokens com base na atenção recebida
"""

# ======================================================
# 1. CONFIGURAÇÃO DE PATH (para acesso à pasta src)
# ======================================================

import sys
import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(ROOT_DIR)

print("Processo iniciado")

# ======================================================
# 2. IMPORTS DO PROJETO
# ======================================================

from src.model_loader import load_bert_pt
from src.attention_extractor import extract_attention
from src.visualization import plot_attention

import numpy as np

# ======================================================
# 3. TEXTO DE ENTRADA
# ======================================================

text = "Da comida o gato não gostou."

print(f"\nTexto analisado:\n{text}")

# ======================================================
# 4. CARREGAMENTO DO MODELO
# ======================================================

tokenizer, model = load_bert_pt()

# ======================================================
# 5. EXECUÇÃO DO MODELO E EXTRAÇÃO DAS ATTENTIONS
# ======================================================

data = extract_attention(
    text=text,
    tokenizer=tokenizer,
    model=model
)

tokens = data["tokens"]
attentions = data["attentions"]

# ======================================================
# 6. INSPEÇÃO DOS TOKENS
# ======================================================

print("\nTOKENS GERADOS PELO TOKENIZER:")
print(tokens)

# ======================================================
# 7. INSPEÇÃO DAS ATTENTION WEIGHTS
# ======================================================

print("\nINFORMAÇÕES DE ATTENTION:")

print(f"Numero de camadas do modelo: {len(attentions)}")

print("Shape de uma camada de attention:")
print(attentions[0].shape)
print(
    "(batch_size, num_heads, seq_len, seq_len)"
)

# ======================================================
# 8. VISUALIZAÇÃO DE UMA CAMADA E HEAD ESPECÍFICA
# ======================================================

# Camadas intermediárias tendem a capturar relações sintático-semânticas
layer = 8
head = 3

print(
    f"\nVisualizando attention | Camada {layer} | Head {head}"
)

plot_attention(
    attentions=attentions,
    tokens=tokens,
    layer=layer,
    head=head
)

# ======================================================
# 9. ANÁLISE QUANTITATIVA DE IMPORTÂNCIA DOS TOKENS
# ======================================================

# Matriz de attention da camada/head selecionada
# Shape: (seq_len, seq_len)
attn_matrix = attentions[layer][0, head].cpu().numpy()

# Soma da atenção RECEBIDA por cada token (colunas)
token_importance = attn_matrix.sum(axis=0)

# Ranking decrescente de relevância
ranking = sorted(
    zip(tokens, token_importance),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOKENS MAIS IMPORTANTES (por atenção recebida):")
for token, score in ranking:
    print(f"{token:>10} -> {score:.3f}")

print("\nProcesso finalizado")
