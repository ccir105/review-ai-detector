import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Remove HTML tags and punctuation
    text = re.sub('<[^>]*>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text and remove stopwords
    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(
        text) if word.lower() not in stop_words]

    # Join the tokens back into a string
    return ' '.join(tokens)
