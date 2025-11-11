from chatbot.data import training_data
from chatbot.model import buid_and_train_model, predict_answer, load_model

def main():
    model, vectorizer, unique_answer = load_model()
# si el modelo viene vacio se debe entrenar
    if model is None:
        model, vectorizer, unique_answer = buid_and_train_model(training_data)

        print("\n 🤖Chatbot listo para conversar. Escribe 'salir' para terminar 🤖")

    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("bot: ¡Hasta luego!")
            break
        response= predict_answer(model, vectorizer, unique_answer, user_input)
        print(f"Bot:" , response)

if __name__ == "__main__":  
    main()      