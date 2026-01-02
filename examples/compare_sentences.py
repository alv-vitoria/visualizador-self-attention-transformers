"""
Experimento educacional para comparar padrões de self-attention
entre duas frases semanticamente diferentes.

Objetivos:
- Separar claramente duas métricas:
    * atenção recebida (attention_in)  -> importância contextual
    * atenção emitida (attention_out) -> atividade relacional
- Ver como pequenas mudanças semânticas afetam a distribuição de atenção
- Trabalhar com uma única layer e head, de forma controlada

Este script é exploratório e NÃO afirma causalidade nem "entendimento" do modelo.
"""

import sys
import os

# ------------------------------------------------------------
# Ajuste de path para permitir imports do diretório src
# ------------------------------------------------------------
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(ROOT_DIR)

# ------------------------------------------------------------
# Imports do projeto
# ------------------------------------------------------------
from src.model_loader import load_bert_pt
from src.attention_extractor import extract_attention
from src.comparison import compare_token_attention_simple
from src.visualization import plot_attention
from src.interactive_visualization import plot_attention_interactive

print("Iniciando experimento de comparação de atenção")

# ------------------------------------------------------------
# Frases escolhidas de forma controlada
# ------------------------------------------------------------
sentences = {
    "positiva": "O aluno passou na prova.",
    "negativa": "O aluno falhou na prova."
}

# ------------------------------------------------------------
# Carregando tokenizer e modelo BERT em português
# ------------------------------------------------------------
tokenizer, model = load_bert_pt()

# ------------------------------------------------------------
# Extração de tokens e atenções
# ------------------------------------------------------------
results = {}

for label, text in sentences.items():
    print(f"Extraindo atenção da frase: {label}")
    data = extract_attention(text, tokenizer, model)
    results[label] = data

# ------------------------------------------------------------
# Layer e head escolhidos para análise
# ------------------------------------------------------------
layer = 8
head = 3

# ------------------------------------------------------------
# Comparação de métricas de atenção
# ------------------------------------------------------------
comparison = compare_token_attention_simple(
    attentions_a=results["positiva"]["attentions"],
    attentions_b=results["negativa"]["attentions"],
    tokens_a=results["positiva"]["tokens"],
    tokens_b=results["negativa"]["tokens"],
    layer=layer,
    head=head
)

# ------------------------------------------------------------
# RESULTADOS — ATENÇÃO RECEBIDA (IMPORTÂNCIA)
# ------------------------------------------------------------
print("\n🔹 Atenção recebida — frase positiva (importância)")
for tok, score in comparison["ranking_in_a"]:
    print(f"{tok:>10} -> {score:.3f}")

print("\n🔹 Atenção recebida — frase negativa (importância)")
for tok, score in comparison["ranking_in_b"]:
    print(f"{tok:>10} -> {score:.3f}")

print("\n🔹 Maior variação de atenção recebida (positiva - negativa)")
for tok, diff in comparison["delta_in"]:
    print(f"{tok:>10} -> {diff:+.3f}")

# ------------------------------------------------------------
# RESULTADOS — ATENÇÃO EMITIDA (ATIVIDADE)
# ------------------------------------------------------------
print("\n🔹 Atenção emitida — frase positiva (atividade)")
for tok, score in comparison["attention_out_a"]:
    print(f"{tok:>10} -> {score:.3f}")

print("\n🔹 Atenção emitida — frase negativa (atividade)")
for tok, score in comparison["attention_out_b"]:
    print(f"{tok:>10} -> {score:.3f}")

# ------------------------------------------------------------
# Visualizações estáticas (Matplotlib)
# ------------------------------------------------------------
print("\nPlotando mapas de atenção (Matplotlib)")

plot_attention(
    results["positiva"]["attentions"],
    results["positiva"]["tokens"],
    layer,
    head
)

plot_attention(
    results["negativa"]["attentions"],
    results["negativa"]["tokens"],
    layer,
    head
)

# ------------------------------------------------------------
# Visualizações interativas (Plotly)
# ------------------------------------------------------------
print("\nAbrindo visualizações interativas (Plotly)")

plot_attention_interactive(
    attentions=results["positiva"]["attentions"],
    tokens=results["positiva"]["tokens"],
    layer=layer,
    head=head,
    title_prefix="Atenção — Frase Positiva"
)

plot_attention_interactive(
    attentions=results["negativa"]["attentions"],
    tokens=results["negativa"]["tokens"],
    layer=layer,
    head=head,
    title_prefix="Atenção — Frase Negativa"
)
