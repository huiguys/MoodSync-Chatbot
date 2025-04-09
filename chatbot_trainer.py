# chatbot_trainer.py
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import os
import sys

# Ensure UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Download NLTK data if missing
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' data...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK 'wordnet' data...")
        nltk.download('wordnet', quiet=True)

ensure_nltk_data()

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
data_file_path = 'data/intents.json'
models_dir = 'models'
words_pkl_path = os.path.join(models_dir, 'words.pkl')
classes_pkl_path = os.path.join(models_dir, 'classes.pkl')
model_h5_path = os.path.join(models_dir, 'chatbot_model.h5')

if not os.path.exists(models_dir):
    print(f"Creating directory: {models_dir}")
    os.makedirs(models_dir)

print("Loading and preprocessing data...")
try:
    with open(data_file_path, 'r', encoding='utf-8') as file:
        raw_content = file.read()
        if not raw_content.strip():
            print(f"Error: '{data_file_path}' is empty.")
            exit()
        intents = json.loads(raw_content)
except FileNotFoundError:
    print(f"Error: '{data_file_path}' not found.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from '{data_file_path}'. Details: {e}")
    exit()

for intent in intents['intents']:
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
    if intent['tag'] == 'invalid':
        continue
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print("-" * 50)
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")
print("-" * 50)

print(f"Saving words to {words_pkl_path}")
pickle.dump(words, open(words_pkl_path, 'wb'))
print(f"Saving classes to {classes_pkl_path}")
pickle.dump(classes, open(classes_pkl_path, 'wb'))

print("Creating training data...")
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Training data created successfully.")
print(f"Input shape (train_x): {train_x.shape}")
print(f"Output shape (train_y): {train_y.shape}")
print("-" * 50)

print("Building the Keras Sequential model...")
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Model Summary:")
model.summary()
print("-" * 50)

print("Training the model...")
history = model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    validation_split=0.1,
    verbose=1
)
print("Model training completed.")
print("-" * 50)

print(f"Saving the trained model to {model_h5_path}")
try:
    model.save(model_h5_path, save_format='h5')
    print("Model saved successfully in .h5 format.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("-" * 50)
print("Chatbot training process finished.")