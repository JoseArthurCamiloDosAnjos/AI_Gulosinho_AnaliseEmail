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
    texto = re.sub(r"\b(td|tr|table|div|span|font|email|list|roman)\b", "", texto)  
    texto = re.sub(r"\b\w{1,2}\b", "", texto) #remove palavras curtas
    return texto

def sistema_de_feedback(texto, predicao_modelo):
    print(f"A frase testada foi: '{texto}'")
    print(f"O modelo classificou como: {'SPAM' if predicao_modelo == 1 else 'HAM'}")
    correto = input("A classificação está correta? (s/n): ").strip().lower()
    if correto == 'n':
        real_label = 0 if predicao_modelo == 1 else 1  
        pasta = "dados/ham" if real_label == 0 else "dados/spam"
        nome_arquivo = f"feedback_{len(os.listdir(pasta))}.txt"
        with open(os.path.join(pasta, nome_arquivo), "w") as f:
            f.write(texto)
        print("Feedback registrado. Obrigado!")
        treinar_modelo()  # Re-treina o modelo com os novos dados
        
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


def treinar_modelo():
    global emails, labels, vectorizer, model
    emails, labels = [], [] # Limpa para recarregar do zero
    
    carregar_pasta("dados/spam", 1)
    carregar_pasta("dados/ham", 0)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(emails)
    
    model = MultinomialNB()
    model.fit(X, labels)
    print("\n[Reforço] Modelo atualizado com os novos dados!")
    
treinar_modelo() # Treino inicial

frase_teste = "meeting confirmed tomorrow"
# 1. Faz a predição
predicao = model.predict(vectorizer.transform([frase_teste]))[0]

# 2. Chama o feedback passando a frase e o que o modelo achou
sistema_de_feedback(frase_teste, predicao)

# 3. Se você marcou 'n' no feedback, ele salvou o arquivo. 
# Agora você chama o treino de novo para ele aprender:
treinar_modelo()