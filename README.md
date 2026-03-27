# 🤖 Gulosinho - Classificador de Emails (Spam vs Ham)

O **Gulosinho** é um projeto simples de Machine Learning feito em Python para **identificar se um email é SPAM ou HAM (legítimo)**.

> ⚠️ Este projeto é educacional e foi desenvolvido para fins de estudo.

---

## 🧠 Como funciona

O modelo utiliza técnicas básicas de NLP:

* Vetorização de texto com `TfidfVectorizer`
* Classificação com `MultinomialNB` (Naive Bayes)
* Pré-processamento de texto (remoção de ruído)

O modelo é treinado com dados locais e salvo em arquivos `.pkl` para reutilização.

---

## ⚙️ Pré-requisitos

* Python 3.8 ou superior
* Pip instalado

---

## 📦 Ambiente virtual (recomendado)

Crie um ambiente virtual para evitar conflitos de dependências:

```bash
python -m venv venv
```

Ative:

### Windows:

```bash
venv\Scripts\activate
```

### Linux/Mac:

```bash
source venv/bin/activate
```

---

## 📥 Instalação das dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install scikit-learn joblib deep-translator
```

---

## 🏋️ Treinando o modelo

Antes de usar a IA, é necessário treinar:

```bash
python train.py
```

Isso irá:

* Ler os emails das pastas `dados/spam` e `dados/ham`
* Treinar o modelo
* Salvar os arquivos em `modelo/`

---

## 🤖 Usando a IA

Depois de treinar:

```bash
python main.py
```

Você poderá:

* Digitar um email no terminal
* Receber a classificação (SPAM ou HAM)
* Ver o nível de confiança
* Receber uma resposta simples da IA

---



## ⚠️ Observações importantes

* O modelo precisa ser treinado antes do uso
* A qualidade da IA depende diretamente dos dados
* A tradução automática pode impactar o desempenho
* Este projeto não é otimizado para produção

---

## 🚧 Limitações

* Modelo simples (Naive Bayes)
* Sem explicação detalhada das decisões
* Sem interface gráfica (apenas terminal)
* Dependência de tradução automática

---

## 💡 Melhorias futuras

* Interface web (Flask ou FastAPI)
* Explicação das previsões
* Remoção da tradução automática
* Avaliação com métricas (precision, recall, etc.)
* Deploy como API

---

## 📚 Referências

* https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
* https://docs.python.org/3/tutorial/venv.html

---

## 📌 Autor

Projeto desenvolvido para aprendizado em Machine Learning e Python.

---
