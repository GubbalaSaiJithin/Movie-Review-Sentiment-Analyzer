import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras import backend as K
from PIL import Image
import base64

# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_analysis_model.h5")

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

model = load_model()
tokenizer = load_tokenizer()

# Predict sentiment
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = ['negative', 'positive', 'neutral'][prediction.argmax()]
    K.clear_session()  # Free memory
    return sentiment

# Load image
def load_image(filepath: str):
    return Image.open(filepath)

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_data}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_background("assets/Images/bg_image.jpg")

# Streamlit app layout
st.title("Movie Review Sentiment Analyzer")
st.markdown("<h2>Analyze the sentiment of movie reviews and get a final recommendation!</h2>", unsafe_allow_html=True)

user_review = st.text_area("Enter the movie review:", height=200)

# Load 3D images and dialogues for sentiment
images = {
    'positive': {
        'image': load_image("assets/3D animated images/positive_review.png")
    },
    'neutral': {
        'image': load_image("assets/3D animated images/neutral_review.png")
    },
    'negative': {
        'image': load_image("assets/3D animated images/negative_review.png")
    }
}

# CSS to center elements
def center_content():
    st.markdown(
        """
        <style>
        .center {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .image {
            width: 300px;
        }
        .dialogue {
            margin-top: 10px;
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Inject CSS for centering
center_content()

# Analyze button action
if st.button("ANALYZE"):
    if user_review:
        sentiment = predict_sentiment(user_review)

        st.markdown('<div class="center">', unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center'>{sentiment.capitalize()} Sentiment</h1>", unsafe_allow_html=True)
        st.image(images[sentiment]['image'], use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("Please enter a review before analyzing.")