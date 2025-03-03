import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalysis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('all')


df=pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
# print(df.head())
def preprocess_text(text):
    tokens=word_tokenize(text.lower())

    filtered_tokens=[token for token in tokens if token not in stopwords.words('english')]

    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(token) for token in tokens]

    processed_text=' '.join(lemmatized_tokens)
    return processed_text