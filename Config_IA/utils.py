import re
import random

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"<.*?>", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto

respostas_spam = [
    "⚠️ Isso parece um golpe.",
    "Cuidado! Esse email tem características de spam.",
    "Isso não parece confiável.",
]

respostas_ham = [
    "✅ Parece um email legítimo.",
    "Tudo certo com essa mensagem.",
    "Não vejo sinais de spam.",
]

def responder(texto, pred):
    if pred == 1:
        return random.choice(respostas_spam)
    else:
        return random.choice(respostas_ham)