import torch
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, render_template, jsonify

# Define supported language pairs
models = {
    'en-de': 'Helsinki-NLP/opus-mt-en-de',  # English to German
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr',  # English to French
    'en-es': 'Helsinki-NLP/opus-mt-en-es'   # English to Spanish
}

app = Flask(__name__)

# Load models at application startup
loaded_models = {}

def load_models():
    for lang_pair, model_name in models.items():
        try:
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            loaded_models[lang_pair] = (model, tokenizer)
            print(f"Model for {lang_pair} loaded successfully.")
        except Exception as e:
            print(f"Error loading model for {lang_pair}: {e}")

load_models()

def translate_text(input_text, model, tokenizer):
    try:
        tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation Error: {e}")
        return "Translation Error"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lang_pair = request.form['lang_pair']
        input_text = request.form['input_text']

        if lang_pair not in loaded_models:
            return jsonify({'error': f"Model for {lang_pair} not loaded."})

        model, tokenizer = loaded_models[lang_pair]
        translated_text = translate_text(input_text, model, tokenizer)

        return jsonify({'translated_text': translated_text})

    return render_template('index.html', language_pairs=models.keys())

if __name__ == "__main__":
    app.run(debug=True) 