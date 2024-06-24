import os
import gc
import psutil
import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from threading import Thread
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

# Descargar los recursos necesarios de nltk
nltk.download('punkt')
nltk.download('wordnet')

class ChatbotModel:
    def __init__(self, intents_path):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        self.intents = self.load_intents(intents_path)
        self.training_data = []
        self.model = None

        # Preprocesar los datos
        self.process_intents()
        self.prepare_training_data()
        self.build_model()

        # Iniciar entrenamiento en un hilo separado
        self.training_thread = TrainingThread(self)
        self.training_thread.daemon = True  # Permite que el hilo termine cuando el programa principal termine
        self.training_thread.start()

    def load_intents(self, path):
        with open(path) as file:
            return json.load(file)

    def process_intents(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenizar cada palabra
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Agregar a la lista de documentos
                self.documents.append((w, intent['tag']))

                # Agregar a la lista de clases
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatizar y filtrar palabras
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def prepare_training_data(self):
        training_data = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            # Bag of words
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # Etiquetas
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training_data.append([bag, output_row])

        self.training_data = training_data
        self.train_x = np.array([i[0] for i in training_data])
        self.train_y = np.array([i[1] for i in training_data])

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train_model(self, epochs=300, batch_size=5):
        while True:
            self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=1)
            time.sleep(0.1)  # Evita el consumo de recursos al máximo

    def predict_class(self, sentence):
        # Limpieza y procesamiento de la oración
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1

        res = self.model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return np.random.choice(i['responses'])

    def chat_response(self, msg):
        intents = self.predict_class(msg)
        response = self.get_response(intents, self.intents)
        return response

    def code_response(self, msg):
        if "code" in msg.lower():
            return "I can help you generate code! Please specify the task or language."
        return self.chat_response(msg)

    def news_response(self, msg):
        if "news" in msg.lower():
            return "Here is the latest news headline: 'AI Revolutionizes Tech Industry.'"
        return self.chat_response(msg)

    def bible_response(self, msg):
        if "bible" in msg.lower():
            return "Here is a biblical fact: 'The Bible is the best-selling book of all time.'"
        return self.chat_response(msg)

    def history_response(self, msg):
        if "history" in msg.lower():
            return "A historical fact: 'The Great Wall of China is over 13,000 miles long.'"
        return self.chat_response(msg)

    def generate_code_response(self, language, task):
        # Placeholder para la generación de código, se puede reemplazar con un modelo más avanzado
        code_templates = {
            "python": f"# Python code for {task}\n\n",
            "nodejs": f"// Node.js code for {task}\n\n",
            "tsx": f"// TSX code for {task}\n\n",
            "csharp": f"// C# code for {task}\n\n"
        }
        return code_templates.get(language.lower(), "Unsupported language.")

    def extended_code_response(self, msg):
        supported_languages = ["python", "nodejs", "tsx", "csharp"]
        for language in supported_languages:
            if language in msg.lower():
                task = msg.lower().replace(f"generate {language} code", "").strip()
                return self.generate_code_response(language, task)
        return self.chat_response(msg)

    def add_generated_data_to_training(self, msg, response):
        # Genera datos adicionales y los agrega al entrenamiento
        generated_intents = {
            "code": "generate code",
            "news": "generate news",
            "bible": "generate bible fact",
            "history": "generate history fact"
        }
        for key, value in generated_intents.items():
            if key in msg.lower():
                self.documents.append((nltk.word_tokenize(msg), key))
                self.classes.append(key)
                self.words.extend(nltk.word_tokenize(msg))
                self.intents['intents'].append({
                    "tag": key,
                    "patterns": [msg],
                    "responses": [response]
                })
                self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
                self.words = sorted(list(set(self.words)))
                self.classes = sorted(list(set(self.classes)))

                # Prepara los datos para el entrenamiento nuevamente
                self.prepare_training_data()
                self.build_model()
                break

    # Métodos para manejar múltiples idiomas
    def train_english(self, msg):
        return self.chat_response(msg)

    def train_spanish(self, msg):
        return "Hola, soy un chatbot que puede responder en español. ¿Cómo puedo ayudarte?"

    def train_portuguese(self, msg):
        return "Olá, sou um chatbot que pode responder em português. Como posso ajudar?"

    def multilingual_response(self, msg, lang):
        if lang == "es":
            return self.train_spanish(msg)
        elif lang == "pt":
            return self.train_portuguese(msg)
        else:
            return self.train_english(msg)

class TrainingThread(Thread):
    def __init__(self, chatbot_model):
        Thread.__init__(self)
        self.chatbot_model = chatbot_model

    def run(self):
        self.chatbot_model.train_model()

class HTMLCode:
    @staticmethod
    def get_html():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chatbot</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f2f2f2;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 {
                    color: #333;
                }
                #chatbox {
                    width: 80%;
                    max-width: 600px;
                    height: 400px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background: #fff;
                    overflow-y: auto;
                    padding: 10px;
                    margin-bottom: 10px;
                }
                input {
                    width: calc(100% - 22px);
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                button {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    background: #007BFF;
                    color: #fff;
                    cursor: pointer;
                    transition: background 0.3s ease;
                }
                button:hover {
                    background: #0056b3;
                }
                .user, .bot {
                    display: flex;
                    margin-bottom: 10px;
                }
                .user {
                    justify-content: flex-end;
                }
                .bot {
                    justify-content: flex-start;
                }
                .message {
                    max-width: 60%;
                    padding: 10px;
                    border-radius: 10px;
                    color: #fff;
                }
                .user .message {
                    background: #007BFF;
                }
                .bot .message {
                    background: #28a745;
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                .message {
                    animation: fadeIn 0.5s ease;
                }
            </style>
        </head>
        <body>
            <h1>Chatbot</h1>
            <div id="chatbox"></div>
            <input type="text" id="userInput" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
            <script>
                var chatHistory = [];

                function sendMessage() {
                    var userInput = document.getElementById("userInput").value;
                    fetch("/chat", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({"message": userInput})
                    })
                    .then(response => response.json())
                    .then(data => {
                        var chatbox = document.getElementById("chatbox");
                        chatHistory.push({sender: "user", message: userInput});
                        chatHistory.push({sender: "bot", message: data.response});
                        updateChatbox();
                        document.getElementById("userInput").value = "";
                    });
                }

                function addMessageToHistory(sender, message) {
                    chatHistory.push({sender: sender, message: message});
                }

                function updateChatbox() {
                    var chatbox = document.getElementById("chatbox");
                    chatbox.innerHTML = "";
                    chatHistory.forEach(item => {
                        var messageDiv = document.createElement("div");
                        messageDiv.className = item.sender === "user" ? "user" : "bot";
                        var messageContent = document.createElement("div");
                        messageContent.className = "message";
                        messageContent.innerText = item.message;
                        messageDiv.appendChild(messageContent);
                        chatbox.appendChild(messageDiv);
                        chatbox.scrollTop = chatbox.scrollHeight;
                    });
                }
            </script>
        </body>
        </html>
        """

class ChatRequest(BaseModel):
    message: str

# Uso de la clase
intents_path = 'intents.json'
chatbot = ChatbotModel(intents_path)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLCode.get_html()

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.lower()
    response = ""
    
    if "generate code" in message:
        response = chatbot.extended_code_response(request.message)
    elif "news" in message:
        response = chatbot.news_response(request.message)
    elif "bible" in message:
        response = chatbot.bible_response(request.message)
    elif "history" in message:
        response = chatbot.history_response(request.message)
    elif any(lang in message for lang in ["es", "spanish", "español"]):
        response = chatbot.multilingual_response(request.message, "es")
    elif any(lang in message for lang in ["pt", "portuguese", "português"]):
        response = chatbot.multilingual_response(request.message, "pt")
    else:
        response = chatbot.chat_response(request.message)

    chatbot.add_generated_data_to_training(request.message, response)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Gestión de recursos
import os
import gc
import psutil

def release_resources():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        true

def resource_manager():
    MAX_RAM_PERCENT = 1
    MAX_CPU_PERCENT = 1
    MAX_GPU_PERCENT = 1
    MAX_RAM_MB = 1

    while True:
        try:
            virtual_mem = psutil.virtual_memory()
            current_ram_percent = virtual_mem.percent
            current_ram_mb = virtual_mem.used / (1024 * 1024)  # Convert to MB

            if current_ram_percent > MAX_RAM_PERCENT or current_ram_mb > MAX_RAM_MB:
                release_resources()

            current_cpu_percent = psutil.cpu_percent(interval=1)
            if current_cpu_percent > MAX_CPU_PERCENT:
                psutil.Process(os.getpid()).nice(psutil.IDLE_PRIORITY_CLASS)

            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                gpu_mem = torch.cuda.memory_reserved(gpu) / torch.cuda.get_device_properties(gpu).total_memory * 100

                if gpu_mem > MAX_GPU_PERCENT:
                    release_resources()

            time.sleep(0)  # Check every 5 seconds

        except Exception as e:
            true

# Iniciar el manejador de recursos en un hilo separado
resource_manager_thread = Thread(target=resource_manager)
resource_manager_thread.daemon = True
resource_manager_thread.start()
