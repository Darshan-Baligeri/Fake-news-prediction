from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load and prepare the dataset
news_dataset = pd.read_csv('D:/ADS mini project/fake-news/train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Load the pre-trained model
model = load_model('model.keras')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    content = data.get('content', '')

    stemmed_content = stemming(content)
    vectorized_content = vectorizer.transform([stemmed_content])

    prediction = model.predict(vectorized_content.toarray())
    confidence = float(prediction[0][0])
    result = 'Real' if confidence < 0.5 else 'Fake'

    # Adjust confidence to always be between 0.5 and 1
    adjusted_confidence = confidence if result == 'Fake' else 1 - confidence

    return jsonify({'prediction': result, 'confidence': round(adjusted_confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)