"""Leakage-aware training: deduplicate text, split first, fit tokenizer on train only."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentiment_model import ROOT, LABELS

def prepare_data(path):
    path = Path(path)
    with path.open('r', encoding='utf-8-sig') as file:
        if file.readline().startswith('version https://git-lfs.github.com/spec/'):
            raise ValueError('This CSV is a Git LFS pointer, not training data. Supply the original CSV with --data.')
    frame = pd.read_csv(path)
    if not {'Reviews', 'Ratings'}.issubset(frame.columns):
        raise ValueError('CSV must contain Reviews and Ratings columns.')
    frame = frame[['Reviews', 'Ratings']].dropna().copy()
    frame['Reviews'] = frame.Reviews.astype(str).str.strip()
    frame['Ratings'] = pd.to_numeric(frame.Ratings, errors='coerce')
    frame = frame[frame.Reviews.ne('') & frame.Ratings.between(0, 10)].copy()
    frame['label'] = np.where(frame.Ratings >= 7, 1, np.where(frame.Ratings < 4, 0, 2))
    frame['text_key'] = frame.Reviews.str.casefold().map(lambda s: re.sub(r'\s+', ' ', s))
    counts = frame.groupby('text_key').label.nunique()
    conflicting = set(counts[counts > 1].index)
    before = len(frame)
    # Exclude contradictory duplicate labels instead of choosing one arbitrarily.
    frame = frame[~frame.text_key.isin(conflicting)].drop_duplicates('text_key').reset_index(drop=True)
    if set(frame.label.unique()) != {0, 1, 2} or frame.label.value_counts().min() < 10:
        raise ValueError('Need at least ten distinct reviews in each of the three sentiment classes.')
    return frame, {'valid_rows_before_deduplication': before, 'unique_nonconflicting_reviews': len(frame),
                   'conflicting_review_texts_removed': len(conflicting)}

def split_data(frame):
    development, test = train_test_split(frame, test_size=.2, random_state=42, stratify=frame.label)
    training, validation = train_test_split(development, test_size=.25, random_state=42, stratify=development.label)
    return training, validation, test

def fit_tokenizer(reviews, vocab_size=5000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def train(data, output=ROOT / 'artifacts/trained', epochs=10, max_length=200, units=32, limit_per_class=None):
    frame, statistics = prepare_data(data)
    if limit_per_class:
        frame = pd.concat([group.sample(min(limit_per_class, len(group)), random_state=42) for _,group in frame.groupby('label')])
    training, validation, test = split_data(frame)
    tf.keras.utils.set_random_seed(42)
    tokenizer = fit_tokenizer(training.Reviews)
    def encode(part):
        return pad_sequences(tokenizer.texts_to_sequences(part.Reviews), maxlen=max_length, padding='pre', truncating='pre')
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(5000, 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)),
        tf.keras.layers.Dense(3, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(encode(training), training.label.to_numpy(), validation_data=(encode(validation), validation.label.to_numpy()),
              epochs=epochs, batch_size=32, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])
    predicted = np.asarray(model(encode(test), training=False)).argmax(axis=1)
    metrics = {**statistics, 'train_rows': len(training), 'validation_rows': len(validation), 'test_rows': len(test),
        'test_accuracy': float(accuracy_score(test.label, predicted)),
        'test_macro_f1': float(f1_score(test.label, predicted, average='macro', zero_division=0)),
        'classification_report': classification_report(test.label, predicted, labels=[0,1,2], target_names=LABELS, output_dict=True, zero_division=0),
        'split': 'Deduplicated text; stratified 60/20/20 split. Tokenizer fitted on training only.',
        'tensorflow': tf.__version__, 'epochs_requested': epochs,
        'evaluation_scope': 'Dataset provided via --data; synthetic smoke fixtures do not measure real-world quality.'}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'model.keras')
    (output / 'tokenizer.json').write_text(tokenizer.to_json(), encoding='utf-8')
    metadata = {'labels': LABELS, 'max_length': max_length, 'vocab_size': 5000, 'metrics': metrics}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = {name:[hashlib.sha256(v.encode()).hexdigest() for v in part.text_key] for name,part in [('train',training),('validation',validation),('test',test)]}
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='classification_report'}, indent=2))
    return metrics

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/IMDB_reviews_dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/trained')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--limit-per-class', type=int)
    args=parser.parse_args()
    train(args.data,args.output,args.epochs,limit_per_class=args.limit_per_class)
