# main.py
from flask import Flask, render_template, request, jsonify
from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_cluster
import random 

app = Flask(__name__)

# Intentamos cargar el modelo (o entrenamos si no existe)
model, vectorizer = load_model()
if model is None:
    model, vectorizer = build_and_train_model(training_data, n_clusters=6)  # ✅ Número de grupos ajustable


#Respuestas por grupo
Respuestas ={
    0:["Soy una asistente virtual creada para ayudarte. 🤖",
       "¡Por supuesto! ¿Con que necesitas ayuda?",
       "Cuentame tu problema y buscare la mejor solución.",
       ],
    1:["Lo siento, no entiendo tu pregunta, puedes intentarlo de nuevo 🤔",
       "Parece que algo no salió bien. ¿Quieres que lo rebisemos?",
       "No siempre soy perfecto.",
       ],      
    2:[
      "¡Hola! ¿En qué puedo ayudarte hoy? 😊",
      "¡Un gusto saldarte! ",
      "¡Hola! ¿Como estas?",
      "Que tal",
      "Buenos dias",
      "hey",
       "saludos",

      "Hasta luego",
       "¡Nos vemos pronto! 👋",
      "¡Cuídate! Hasta la próxima.",
    "Adios, que estés bien.",
    "me despido",

      ],
    3:["Hasta luego",
       "¡Nos vemos pronto! 👋",
      "¡Cuídate! Hasta la próxima.",
    "Adios, que estés bien.",
      ],

    4:["¡Gracias a ti! 😊",
       "De nada, estoy aquí para ayudarte.",
       "¡Muy amable de tu parte!",
       ],
    5:["Puedo ofecerte información o resolver tus dudas.",
       "¡En qué tema necesitas ayuda?",
       "Estoy aqui para resolver tus preguntas.",
       "Lo siento, no entiendo tu pregunta, puedes intentarlo de nuevo 🤔",
       "Parece que algo no salió bien. ¿Quieres que lo rebisemos?",
       "No siempre soy perfecto.",
       ],
    
    
}
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.form.get("message", "")
    if not user_text.strip():
        return jsonify({"response": "Por favor escribe algo 😅"})

    # Predice el grupo al que pertenece el mensaje
    cluster = predict_cluster(model, vectorizer, user_text)

    # ✅ Mensaje más descriptivo
    #response = f"Tu mensaje pertenece al grupo {cluster}. Este grupo contiene frases con significados similares.
    response = f"Tu mensaje pertenece al grupo {cluster}. Este grupo contiene frases con significados similares."
    response = random.choice(Respuestas.get(cluster, [
        "No estoy seguro de entender, pero puedo intentarlo otra vez 🤔"
        ]))
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
