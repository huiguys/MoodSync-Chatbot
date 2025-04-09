# test_model.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
model = load_model('models/chatbot_model.h5', compile=False)
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    matches = 0
    for s_word in sentence_words:
        for i, vocab_word in enumerate(words):
            if vocab_word == s_word:
                bag[i] = 1
                matches += 1
    return np.array(bag), matches

def predict_class(sentence, model):
    p, matches = bow(sentence, words)
    if matches == 0:
        return [{"intent": "invalid", "probability": 1.0}]
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]
    if not return_list or max([r["probability"] for r in return_list]) < 0.9:  # Match app.py
        return [{"intent": "invalid", "probability": 1.0}]
    return return_list

test_inputs = ["hi", "hello", "bye", "who are you?", "what google do?", "can you able to google?", "what is wether today?", "who is elon musk?"]
for input_text in test_inputs:
    prediction = predict_class(input_text, model)
    print(f"Input: {input_text}")
    print(f"Prediction: {prediction}\n")