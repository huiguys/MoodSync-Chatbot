from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import os
import sys
from transformers import pipeline

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

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

lemmatizer = WordNetLemmatizer()

print("Initializing LLM pipeline with GPT-Neo-125M...")
llm_pipeline = pipeline(
    "text-generation", model="EleutherAI/gpt-neo-125m", max_length=200)

# Load Trained Model and Data
models_dir = 'models'
data_dir = 'data'
model_path = os.path.join(models_dir, 'chatbot_model.h5')
words_path = os.path.join(models_dir, 'words.pkl')
classes_path = os.path.join(models_dir, 'classes.pkl')
intents_path = os.path.join(data_dir, 'intents.json')

required_files = [model_path, words_path, classes_path, intents_path]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("Error: The following required files are missing:")
    for f in missing_files:
        print(f"- {f}")
    print("Please run 'chatbot_trainer.py' first.")
    exit()

try:
    print("Loading model and data files...")
    model = load_model(model_path, compile=False)
    words = pickle.load(open(words_path, 'rb'))
    classes = pickle.load(open(classes_path, 'rb'))
    with open(intents_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    print(
        f"Model loaded from {model_path} and data files loaded successfully.")
except Exception as e:
    print(f"Error loading model or data files: {e}")
    exit()

# Mood Detection Function


def detect_mood(message):
    message = message.lower()
    positive_keywords = ['happy', 'great', 'awesome', 'good', 'yay']
    negative_keywords = ['sad', 'bad', 'terrible', 'sorry', 'ugh']
    angry_keywords = ['angry', 'mad', 'annoyed', 'furious']

    if any(keyword in message for keyword in positive_keywords):
        return "positive"
    elif any(keyword in message for keyword in negative_keywords):
        return "negative"
    elif any(keyword in message for keyword in angry_keywords):
        return "angry"
    return "neutral"

# Helper Functions


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    matches = 0
    for s_word in sentence_words:
        for i, vocab_word in enumerate(words):
            if vocab_word == s_word:
                bag[i] = 1
                matches += 1
                if show_details:
                    print(f"Found in bag: {vocab_word}")
    return np.array(bag), matches


def predict_class(sentence, model):
    p, matches = bow(sentence, words, show_details=False)
    if matches == 0:
        return [{"intent": "invalid", "probability": 1.0}]
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]],
                    "probability": float(r[1])} for r in results]
    if not return_list or max([r["probability"] for r in return_list]) < 0.9:
        return [{"intent": "invalid", "probability": 1.0}]
    return return_list


def get_response(ints, intents_json, user_message):
    tag = ints[0]['intent']
    probability = ints[0]['probability']
    mood = detect_mood(user_message)

    if tag == "invalid" or probability < 0.9:
        print(
            f"Falling back to LLM for: '{user_message}' (Intent: {tag}, Prob: {probability}, Mood: {mood})")
        prompt = f"Provide a concise answer to this question: '{user_message}'"
        llm_response = llm_pipeline(
            prompt,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.7
        )[0]['generated_text']
        print(f"Raw LLM output: '{llm_response}'")
        response = llm_response.replace(prompt, "").strip()
        sentences = response.split('. ')
        clean_response = sentences[0].strip() + '.' if sentences else response
        print(f"Cleaned LLM response: '{clean_response}'")
        if not clean_response or len(clean_response) < 5:
            if "weather" in user_message.lower() or "wether" in user_message.lower():
                base_response = "I can’t check the weather, but I hope it’s nice today!"
            elif "google" in user_message.lower():
                base_response = "Google is a tech company that provides search and other services."
            elif "elon" in user_message.lower():
                base_response = "Did you mean Elon Musk? He’s an entrepreneur known for Tesla and SpaceX."
            elif "trump" in user_message.lower():
                base_response = "Donald Trump is a businessman and former U.S. President."
            else:
                base_response = "Sorry, I didn’t get that. Could you rephrase it?"
        else:
            base_response = clean_response[:150]
    else:
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                base_response = random.choice(i['responses'])
                break
        else:
            base_response = "Sorry, I don't have a specific response for that right now."

    # Adjust response based on mood
    if mood == "positive":
        return f"Yay, glad you're in a good mood! {base_response}"
    elif mood == "negative":
        return f"Aw, sorry you’re feeling down. {base_response}"
    elif mood == "angry":
        return f"Whoa, take it easy! {base_response}"
    return base_response

# Flask Routes


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        user_message = request.json['message']
        print(f"Received message: {user_message}")
        ints = predict_class(user_message, model)
        print(f"Predicted intents: {ints}")
        response = get_response(ints, intents, user_message)
        print(f"Sending response: {response}")
        return jsonify({"response": response})
    except KeyError:
        print("Error: 'message' key not found in request JSON.")
        return jsonify({"response": "Error: Invalid request format."}), 400
    except Exception as e:
        print(f"Error during chatbot response generation: {e}")
        return jsonify({"response": "Sorry, something went wrong."}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
