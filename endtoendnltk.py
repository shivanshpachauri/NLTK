import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

df['reviewText']=df['reviewText'].apply(preprocess_text)
print(df)

analyzer=SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores=analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos']>0 else 0
    return sentiment

df['sentiment']=df['reviewText'].apply(get_sentiment)
print(df)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(df['Positive'],df['sentiment']))


from sklearn.metrics import classification_report
print(classification_report(df['Positive'],df['sentiment']))