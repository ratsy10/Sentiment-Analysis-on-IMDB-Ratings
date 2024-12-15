from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Home route (Optional)
@app.route('/', methods=['GET'])
def home():
    return "API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    
    # Preprocess the review text
    processed_text = preprocess_text(review)
    
    # Transform the review using the same vectorizer
    transformed_text = vectorizer.transform([processed_text])
    
    # Predict sentiment (0: Negative, 1: Positive)
    prediction = model.predict(transformed_text)
    
    # Return the result as JSON
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
