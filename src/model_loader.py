# src/model_loader.py

from transformers import BertTokenizer, BertModel

def load_bert_pt():
    """
    Carrega um BERT treinado em português com attention ativada.
    """
    model_name = "neuralmind/bert-base-portuguese-cased"

    tokenizer = BertTokenizer.from_pretrained(model_name)

    model = BertModel.from_pretrained(
        model_name,
        output_attentions=True,  # MUITO IMPORTANTE
        output_hidden_states=False
    )

    model.eval()

    return tokenizer, model
