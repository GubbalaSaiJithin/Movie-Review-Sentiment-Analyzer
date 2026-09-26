"""Reusable inference with the same tokenizer and sequence length as training."""
from pathlib import Path
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT = Path(__file__).resolve().parent
LABELS = ['negative', 'positive', 'neutral']

def load_bundle(artifact_dir=None):
    folder = Path(artifact_dir) if artifact_dir else ROOT / 'artifacts/trained'
    if (folder / 'metadata.json').is_file():
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        model = tf.keras.models.load_model(folder / 'model.keras', compile=False)
        tokenizer = tokenizer_from_json((folder / 'tokenizer.json').read_text(encoding='utf-8'))
        return {'model': model, 'tokenizer': tokenizer, 'max_length': metadata['max_length'],
                'labels': metadata['labels'], 'source': 'retrained artifact'}
    if artifact_dir:
        raise FileNotFoundError(f'Missing metadata.json in {folder}')
    model = tf.keras.models.load_model(ROOT / 'sentiment_analysis_model.h5', compile=False)
    # This is the trusted tokenizer shipped with this repository, not user input.
    with (ROOT / 'tokenizer.pkl').open('rb') as file:
        tokenizer = pickle.load(file)
    return {'model': model, 'tokenizer': tokenizer, 'max_length': 200, 'labels': LABELS,
            'source': 'original pretrained artifact; unbiased accuracy not reverified'}

def encode_review(review, bundle):
    if not isinstance(review, str) or not review.strip():
        raise ValueError('Enter a non-empty movie review.')
    if len(review) > 20000:
        raise ValueError('Keep the review below 20,000 characters.')
    sequence = bundle['tokenizer'].texts_to_sequences([review.strip()])
    if not sequence[0]:
        raise ValueError('The review contains no words recognised by this tokenizer.')
    return pad_sequences(sequence, maxlen=bundle['max_length'], padding='pre', truncating='pre')

def predict_sentiment(review, bundle):
    encoded = encode_review(review, bundle)
    # Reuse the cached model; clearing the global Keras session here is incorrect.
    probabilities = np.asarray(bundle['model'](encoded, training=False))[0]
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise ValueError('The model returned invalid class probabilities.')
    return {'sentiment': bundle['labels'][int(probabilities.argmax())],
            'scores': {label: float(score) for label,score in zip(bundle['labels'], probabilities)},
            'model_source': bundle['source']}
