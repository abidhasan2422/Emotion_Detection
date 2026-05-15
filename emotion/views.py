import os
import re
import string
import pickle
from django.shortcuts import render

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

tfidf     = bundle['tfidf']
model     = bundle['model']
label_map = bundle['label_map']

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

EMOTION_META = {
    'joy':      {'emoji': '😄', 'color': '#f59e0b'},
    'sadness':  {'emoji': '😢', 'color': '#3b82f6'},
    'anger':    {'emoji': '😠', 'color': '#ef4444'},
    'fear':     {'emoji': '😨', 'color': '#8b5cf6'},
    'love':     {'emoji': '❤️',  'color': '#ec4899'},
    'surprise': {'emoji': '😲', 'color': '#10b981'},
}

# ── View ─────────────────────────────────────────────────────
def index(request):
    result     = None
    text_input = ''

    if request.method == 'POST':
        text_input = request.POST.get('text', '').strip()

        if text_input:
            cleaned  = preprocess(text_input)
            vec      = tfidf.transform([cleaned])
            pred     = model.predict(vec)[0]
            emotion  = label_map[pred]
            meta     = EMOTION_META.get(emotion, {'emoji': '🤔', 'color': '#6b7280'})

            result = {
                'emotion': emotion.capitalize(),
                'emoji':   meta['emoji'],
                'color':   meta['color'],
            }

    return render(request, 'emotion/index.html', {
        'result': result,
        'text':   text_input,
    })