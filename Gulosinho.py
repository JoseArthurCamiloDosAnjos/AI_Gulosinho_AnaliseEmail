from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

emails = []
labels = []

# spam
with open("Dados/spam.txt", "r", encoding="utf-8") as f:
    for linha in f:  
        emails.append(linha.strip())
        labels.append(1)

# ham
with open("Dados/ham.txt", "r", encoding="utf-8") as f:
    for linha in f:
        emails.append(linha.strip())
        labels.append(0)
# Vetorização
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Divisão
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2)

# Modelo
model = MultinomialNB()
model.fit(X_train, y_train)

#teste
pred = model.predict(X_test)

print(classification_report(y_test, pred))

teste = ["Clique aqui para ver o relatório"]
print(model.predict(vectorizer.transform(teste)))