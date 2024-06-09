from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\HateSpeechData.csv")

# Train TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    return text

# Preprocess the text data
df['tweet'] = df['tweet'].apply(preprocess_text)

# Split data into features and labels
X = df['tweet']
y = df['class']  # Assuming 'class' column denotes the label

# Vectorize preprocessed text
X_tfidf = tfidf_vectorizer.fit_transform(X)
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_tfidf, y)

# LSTM model
max_words = 1000
max_len = 150
embedding_dim = 100

lstm_model = Sequential()
lstm_model.add(Embedding(max_words, embedding_dim, input_length=max_len))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

# Train LSTM model
history_lstm = lstm_model.fit(pad_sequences(X_tfidf.toarray(), maxlen=max_len),
                              y, epochs=1, batch_size=32,
                              callbacks=[early_stop])

# CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(max_len, 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
history_cnn = cnn_model.fit(pad_sequences(X_tfidf.toarray(), maxlen=max_len).reshape(-1, max_len, 1),
                            y, epochs=1, batch_size=32,
                            callbacks=[early_stop])

# Function to classify input text using SVM
def classify_text_svm(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction_prob = svm_classifier.predict_proba(text_tfidf)
    return prediction_prob[0][1]  # Probability of class 1 (hate speech)

# Function to classify input text using LSTM
def classify_text_lstm(text):
    preprocessed_text = preprocess_text(text)
    text_sequence = tfidf_vectorizer.transform([preprocessed_text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len)
    prediction = lstm_model.predict(padded_sequence)
    return prediction[0][0]  # Probability of class 1 (hate speech)

# Function to classify input text using CNN
def classify_text_cnn(text):
    preprocessed_text = preprocess_text(text)
    text_sequence = tfidf_vectorizer.transform([preprocessed_text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len)
    prediction = cnn_model.predict(padded_sequence.reshape(-1, max_len, 1))
    return prediction[0][0]  # Probability of class 1 (hate speech)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        
        prediction_svm = classify_text_svm(input_text)
        
        # Generate result message
        if prediction_svm > 0.5:
            result_message = "This text contains hate speech. Percentage of hate speech is {:.2f}%".format(prediction_svm * 100)
        else:
            result_message = "This text does not contain hate speech."
        
        return render_template('result.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
