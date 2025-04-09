# MoodSync Chatbot

![Chatbot Banner]([assets/banner.png](https://github.com/huiguys/MoodSync-Chatbot/blob/main/assets/banner.jpg?raw=true))  
*A smart, mood-responsive chatbot built with Flask, TensorFlow, and GPT-Neo.*

## Overview

**MoodSync Chatbot** is an innovative conversational AI that adapts its tone based on detected mood. Combining rule-based intent recognition with generative AI, it provides dynamic responses.

### Key Features
- **Mood Detection**: Adjusts tone based on user sentiment
- **Hybrid AI**: TensorFlow intent classification + LLM fallback
- **Web Interface**: Flask-based chat UI
- **Easy Training**: Customize via `intents.json`

## Project Structure
MoodSync-Chatbot/
├── data/
│ └── intents.json
├── models/
│ ├── chatbot_model.h5
│ ├── words.pkl
│ └── classes.pkl
├── templates/
│ └── index.html
├── app.py
├── chatbot_trainer.py
├── requirements.txt
└── README.md

## Prerequisites
- Python 3.8+
- Git
- A GitHub account

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MoodSync-Chatbot.git
cd MoodSync-Chatbot

2. Install Dependencies
pip install -r requirements.txt

3. Download NLTK Data
python download_nltk_data.py

4. Train the Model
python chatbot_trainer.py

Note: This generates chatbot_model.h5, words.pkl, and classes.pkl in the models/ directory

5. Run the Chatbot
python app.py
    Open your browser to http://127.0.0.1:5000 and start chatting!

Usage
    Greet: "Hi" or "Hello" → "Hi there, how can I help?"
    Ask: "Who is Elon Musk?" → Trained response or LLM fallback.
    Mood Test: "I’m happy, who is Trump?" → "Yay, glad you're in a good mood! Donald Trump is..."

Customization
    Add Intents: Edit data/intents.json with new patterns and responses, then retrain with chatbot_trainer.py.
    Change LLM: Swap EleutherAI/gpt-neo-125m in app.py for another model (e.g., gpt2).
    Mood Keywords: Modify detect_mood() in app.py to tweak mood detection.

Dependencies
    Flask==3.0.3
    tensorflow==2.19.0
    tf-keras==2.19.0
    transformers==4.40.0
    nltk==3.8.1
    numpy==1.26.4
See requirements.txt for the full list.

Future Enhancements
    Fine-tune LLM responses for better accuracy.
    Add sentiment analysis with a pre-trained model (e.g., VADER).
    Deploy to a cloud platform like Heroku or Render.

Acknowledgements
    Built with  by Srinivasa P M
    Powered by TensorFlow, and Hugging Face Transformers.






