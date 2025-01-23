# Movie Review Sentiment Analyzer

This project is a **Movie Review Sentiment Analyzer** built using a Long Short Term Memory (LSTM) neural network for classifying movie reviews into **positive**, **negative**, and **neutral** sentiments. The model is trained on the IMDB reviews dataset, and the project also includes a **Streamlit web application** that allows users to enter movie reviews and get sentiment analysis results with visual feedback through 3D animated images.

## Project Overview

This project is divided into two parts:
1. **Model Training**: Training a sentiment analysis model using LSTM on a cleaned, balanced dataset.
2. **Web Application**: A Streamlit-based user interface to input movie reviews and display the predicted sentiment with an engaging visual representation.

## Dataset

The dataset used in this project is the **IMDB Movie Reviews dataset**, which contains reviews and corresponding ratings. To improve model performance, the ratings are converted into labels: 

* **Positive (1)**: Ratings ≥ 7
* **Negative (0)**: Ratings < 4
* **Neutral (2)**: Ratings between 4 and 6

Here are a few example entries from the dataset:

| **Ratings** | **Reviews**                                   | **Movies**  | **Resenhas**                         |
|-------------|-----------------------------------------------|-------------|--------------------------------------|
| 5           | The movie was okay, it's a one-time watch.     | ExampleMovie1 | Filme foi bom, vale ver uma vez.     |
| 10           | This movie was fantastic. I loved it!          | ExampleMovie2 | Este filme foi fantástico! Eu adorei!|
| 1          | This movie is terrible!                        | ExampleMovie3 | Este filme é terrível!               |

## Model Training

The model training process is documented in the **model_training.ipynb** file. Key steps include:

1. **Data Cleaning**: Removing missing values and unbalanced data distribution.
2. **Label Encoding**: Reviews are labeled as positive, negative, or neutral based on their rating.
3. **Data Preprocessing**: Tokenizing text data and padding sequences to a maximum length of 200 tokens.
4. **Model Architecture**: A **Bidirectional LSTM** model with:
    - An Embedding layer
    - Two LSTM layers for sequential processing
    - A Dense output layer with 3 nodes for multi-class classification
5. **Training**: The model is compiled using **sparse_categorical_crossentropy** and optimized using the **Adam optimizer**. It is trained for 20 epochs with **early stopping**.

The final model and tokenizer are saved for use in the web application.

## Streamlit Web Application

The web application provides a simple and intuitive user interface for predicting the sentiment of user-provided movie reviews. It displays visual feedback using **3D animated images** to represent the predicted sentiment (positive, neutral, or negative).

Key features:
* **Text input**: Enter a movie review.
* **Sentiment Prediction**: Based on the review, the model predicts whether the sentiment is positive, negative, or neutral.
* **3D Animated Feedback**: Corresponding to the prediction, an animated image is displayed.

The app is customizable, and the background can be changed to match the theme of your choice.

## How to Run

Follow these steps to run the project locally:

### Prerequisites

* Python 3.9+
* TensorFlow
* Streamlit
* Pandas
* Pillow
* scikit-learn
* pickle

### Step 1: Clone the Repository

```bash
git clone [https://github.com/GubbalaSaiJithin/Movie-Review-Sentiment-Analyzer].git
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Web Application

```bash
streamlit run app.py
```

## Step 4: Analyze Movie Reviews

1. **Enter your movie review** in the text area provided.
2. Click the **"ANALYZE"** button to see the predicted sentiment and corresponding visual feedback.

### Interface 
![1](https://github.com/user-attachments/assets/27511d2c-816e-4634-9da5-5c84d0b590a3)

## Model Training (Optional)

To retrain the model, you can run the `model_training.ipynb` notebook using Jupyter Notebook or any other IDE that supports `.ipynb` files.

## Results

The LSTM model achieved satisfactory accuracy and performed well on both the training and test datasets. You can enter various movie reviews and see how the model predicts sentiment based on the text content.

### Example Predictions:

* ### Positive Review:
  ![2](https://github.com/user-attachments/assets/71615837-4bee-49ae-a88f-bc1a8d426a69)
  ![3](https://github.com/user-attachments/assets/ccc8d514-b82d-4a5a-aa21-bd63475540f5)
* ### Neutral Review:
  ![4](https://github.com/user-attachments/assets/e5ea5061-8892-4e03-8829-e3a1b3b75316)
  ![5](https://github.com/user-attachments/assets/d9312169-14d3-48bc-825d-153c7306a419)
* ### Negative Review:
  ![6](https://github.com/user-attachments/assets/07780940-fe10-4b4c-ad7b-b5dfe2c1af7b)
  ![7](https://github.com/user-attachments/assets/75648ecc-3d23-4e4a-9564-7edabb78e90b)

## Technologies Used

* **Python**: Main programming language.
* **TensorFlow/Keras**: Used for building and training the LSTM model.
* **Pandas**: For data manipulation and analysis.
* **Streamlit**: For creating the web application.
* **PIL (Pillow)**: For handling images.
* **scikit-learn**: For data preprocessing and splitting.
* **LSTM (Long Short Term Memory)**: Deep learning model used for sentiment analysis.

## Acknowledgements

* The **IMDB Movie Reviews dataset** for providing the data to build this project.
* **TensorFlow** and **Keras** for providing an intuitive deep learning framework.
* **Streamlit** for creating an easy-to-use web interface.

## Author

This Movie Review Sentiment Analyzer was developed by :
-	[@GubbalaSaiJithin](https://github.com/GubbalaSaiJithin)
-	Repository : https://github.com/GubbalaSaiJithin/Movie-Review-Sentiment-Analyzer.git

## Feedback

For any feedback or queries, please reach out to me at saijithin09@gmail.com.
