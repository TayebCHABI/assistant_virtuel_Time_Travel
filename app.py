# -*- coding: utf-8 -*-
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import random
import pickle
from flask import Flask, request, jsonify, render_template

# Charger le modèle et les données
model = load_model('chatbot_model6.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Charger le modèle de SpaCy
nlp = spacy.load('fr_core_news_md')

# Initialiser l'application Flask
app = Flask(__name__)

# Fonction pour nettoyer les phrases
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_.lower() for token in doc if token.text not in ['?', '!', '.', ',']]
    return tokens

# Fonction pour créer un bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Fonction pour prédire la classe
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Fonction pour obtenir une réponse
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['intent'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Fonction principale pour obtenir une réponse à une entrée de l'utilisateur
def chatbot_response(text):
    ints = predict_class(text, model)
    if ints:
        res = get_response(ints, intents)
        return res
    else:
        return "Je ne comprends pas bien. Pouvez-vous reformuler?"

# Définir une route pour le chatbot en JSON
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get('message')
    response = chatbot_response(message)
    return jsonify({"response": response})

# Définir une route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Démarrer l'application Flask sur le port 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
