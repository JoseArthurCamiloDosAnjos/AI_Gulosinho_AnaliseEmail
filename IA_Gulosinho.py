import joblib
from Config_IA.utils import responder, limpar_texto

print("Carregando o modelo...")

#Ele carrega o modelo e o vectorizer treinados
model = joblib.load("modelo/modelo.pkl")
vectorizer = joblib.load("modelo/vectorizer.pkl")


print("Bem-vindo ao Gulosinho! O modelo está pronto para analisar seus emails.")

while True:
    email = input("\n Digite um email (ou 'sair'): ")

    if email.lower() == "sair":
        print("Até mais👋! Te vejo na proxima análise.")
        break

    if not email.strip():
        print("⚠️ Digite um email válido.")
        continue

    email_limpo = limpar_texto(email)

    X = vectorizer.transform([email_limpo])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    print(responder(email, pred))
    print(f"Classificação: {'SPAM' if pred == 1 else 'HAM'}")
    print(f"Nível de segurança: {prob:.2%}")

    resposta = input("Quer mandar outro email? (s/n): ").strip().lower()

    if resposta != "s":
        print("Agora o Gulosinho vai descansar, tenha um ótimo dia!👋")
        break


