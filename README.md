# Movie Review Sentiment Analyzer

A Streamlit application that classifies English movie reviews as negative, positive or neutral using a TensorFlow/Keras LSTM model.

## Run locally

Tested on Windows, Python 3.10, TensorFlow 2.19 and Keras 3.9. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true
```

The original sentiment_analysis_model.h5 and tokenizer.pkl are included, so inference does not require the training CSV. All asset paths are anchored to the project directory. Only load pickle artifacts you trust; the app loads the tokenizer shipped with this repository, not user uploads.

## Repairs made in September 2026

- Load the saved model with compile=False for inference.
- Reuse the cached model instead of clearing Keras's global session after every prediction.
- Validate blank, punctuation-only, unrecognised and excessively long input.
- Keep the label order and padding/truncation settings consistent with training.
- Remove required decorative image dependencies, so missing background assets do not break model inference.
- Add separate, testable model/preprocessing functions.
- Add corrected training code that removes duplicate text, excludes conflicting duplicates, separates train/validation/test, and fits the tokenizer only on training data.
- Save new training bundles using consistent model.keras, tokenizer.json and metadata.json filenames.
- Update the notebook to use the same corrected training and inference functions. Its historical leaked accuracy output has been removed.

## Missing training data

The root IMDB_reviews_dataset.csv is only a 134-byte Git LFS pointer. It refers to a 384,061,983-byte object with SHA-256:

```
064d754f25565ffdf3d2f4ebab67809b759ebf90eb52c0ac101ca034db1c045d
```

On 26 September 2026, both the media download and GitHub's LFS batch endpoint returned 404; the latter reported that the object does not exist on the server. fetch_dataset.py documents the original URL and checksum, but it cannot restore an object absent from GitHub.

If you have the original CSV, place it at data/IMDB_reviews_dataset.csv, or pass its path explicitly:

```powershell
.\.venv\Scripts\python.exe train.py --data 'D:\path\IMDB_reviews_dataset.csv' --epochs 10
```

New models are saved in artifacts/trained/ and selected by the app on its next restart. The original pretrained artifacts are not overwritten. Training uses an explicit stratified 60/20/20 split and restores the best validation-loss epoch. To limit a development run, use --limit-per-class.

## What has and has not been verified

- The original model/tokenizer load and perform repeated inference.
- Five regression/interaction tests passed, including a Streamlit button click and validation errors.
- A real local Streamlit server passed its HTTP health check.
- The corrected training/save/reload path was exercised for one epoch using a 90-row synthetic fixture. This checks code execution only; its accuracy is not a portfolio metric.
- Full retraining and unbiased accuracy evaluation on real reviews remain unverified because the actual dataset is missing.

The original notebook fitted its tokenizer on all reviews, including the test partition. Therefore its historical test accuracy must not be presented as a fresh, leakage-free evaluation. Model scores in the app are not calibrated confidence estimates. Sarcasm, unfamiliar vocabulary and mixed sentiment can be misclassified.

Dataset licensing and source details need documenting when the original training data are recovered.
