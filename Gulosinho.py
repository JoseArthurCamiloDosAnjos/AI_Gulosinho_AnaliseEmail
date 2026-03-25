from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter 
import os
import re

emails = []
labels = []

def limpar_email(texto):
    partes = texto.split("\n\n", 1)

    if len(partes) >1:
        return partes [1] #pega só corpo
    
    return texto
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)   #remove links
    texto = re.sub(r"<.*?>","", texto)      #remove Html
    texto = re.sub(r"[^\w\s]", "", texto)   #remove pontuação
    texto = re.sub(r"\b\w*3d\w*\b", "", texto)#remove frazes quebradas
    texto = re.sub(r"\b(helvetica|nbsp|amp|font|table)\b", "", texto)
    return texto

def carregar_pasta(caminho, label):
    for arquivo in os.listdir(caminho):
        caminho_arquivo = os.path.join(caminho, arquivo)

        #garante que é um arquivo
        if os.path.isfile(caminho_arquivo):
            with open(caminho_arquivo, "r", encoding= "latin-1") as f:
                texto = f.read().strip()
                

            if texto: #evita vazio
                texto = limpar_email(texto)

                texto = limpar_texto(texto)

                emails.append(texto)
                labels.append(label)

carregar_pasta("dados/spam", 1)
carregar_pasta("dados/ham",0)

# Vetorização
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2, stop_words="english")
X = vectorizer.fit_transform(emails)

# Divisão
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.3, stratify=labels, random_state=42)

# Modelo
model = MultinomialNB()
model.fit(X_train, y_train)

#avaliação 
pred = model.predict(X_test)
print(classification_report(y_test, pred, zero_division=0))
#teste
teste = [
    "ganhe dinheiro agora com essa oferta exclusiva clique aqui",
    "reunião confirmada amanhã",
    "clique aqui urgente",
    "segue relatório em anexo",
    "oferta exclusiva para clientes",
    "documento importante clique aqui"
    ]
for t in teste:
    print(t, "->", model.predict(vectorizer.transform([t]))[0])
print(model.predict(vectorizer.transform(teste)))

print("\nEXEMPLO LIMPO:")
print(emails[0][:500])

feature_names = vectorizer.get_feature_names_out()
probs = model.feature_log_prob_

top_spam = probs[1].argsort()[-10:]
top_ham = probs[0].argsort()[-10:]

print("SPAM:", [feature_names[i] for i in top_spam])
print("HAM:", [feature_names[i] for i in top_ham])