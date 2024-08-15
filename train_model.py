import json
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
import random
import pickle

# Charger les données d'intentions
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Charger le modèle de SpaCy
nlp = spacy.load('fr_core_news_md')

# Initialiser les listes pour les mots, les classes et les documents
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Traiter les intentions
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokeniser chaque mot dans la phrase
        doc = nlp(pattern)
        tokens = [token.lemma_.lower() for token in doc if token.text not in ignore_words]
        words.extend(tokens)
        documents.append((tokens, intent['intent']))
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

# Éliminer les doublons et trier les mots et les classes
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Créer les données d'entraînement
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if word in doc[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mélanger les données et les convertir en numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Créer les ensembles d'entraînement et de test
train_x = np.array([i[0] for i in training.tolist()])
train_y = np.array([i[1] for i in training.tolist()])

# Définir le modèle avec Keras
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')  # Changer à softmax
])

# Compiler le modèle avec un optimiseur Adam et un taux d'apprentissage fixe
adam = Adam(learning_rate=0.005)  # Réduire le taux d'apprentissage
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Définir la custom callback d'arrêt anticipé
class CustomEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        if logs.get('accuracy') > 0.9 and logs.get('val_accuracy') > 0.85:
            print("\nReached desired accuracy, stopping training.")
            self.model.stop_training = True

# Utilisation dans votre modèle
custom_early_stopping = CustomEarlyStopping()

# Entraîner le modèle avec la custom callback d'arrêt anticipé
history = model.fit(train_x, train_y, epochs=200, batch_size=10, verbose=1,
                    validation_split=0.3, callbacks=[custom_early_stopping])

# Sauvegarder le modèle au format .keras
model.save('chatbot_model6.keras')

# Sauvegarder les données nécessaires pour le traitement ultérieur
with open('words.pkl', 'wb') as words_file:
    pickle.dump(words, words_file)
with open('classes.pkl', 'wb') as classes_file:
    pickle.dump(classes, classes_file)
