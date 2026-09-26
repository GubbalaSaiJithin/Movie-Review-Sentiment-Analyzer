from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
from sentiment_model import load_bundle, predict_sentiment

st.set_page_config(page_title='Movie Review Sentiment Analyzer')
st.title('Movie Review Sentiment Analyzer')
st.write('Classify an English movie review as negative, positive or neutral.')

@st.cache_resource
def get_bundle():
    return load_bundle()

try:
    bundle = get_bundle()
except (OSError, ValueError, ModuleNotFoundError) as error:
    st.error(f'Could not load model assets: {error}')
    st.stop()

review = st.text_area('Enter the movie review', height=180, max_chars=20000)
if st.button('Analyze'):
    try:
        result = predict_sentiment(review, bundle)
        st.subheader(result['sentiment'].capitalize() + ' sentiment')
        st.bar_chart(pd.DataFrame.from_dict(result['scores'], orient='index', columns=['Model score']))
        st.caption('Model scores are not calibrated confidence estimates. Sarcasm and unfamiliar text can be misclassified.')
    except ValueError as error:
        st.error(str(error))

with st.expander('Model information'):
    st.write(bundle['source'])
    st.write('The original model can run without the training CSV. A fresh accuracy evaluation requires the real dataset; the repository CSV is a Git LFS pointer.')
