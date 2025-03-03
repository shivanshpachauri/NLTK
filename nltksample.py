import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon

# lexicon by valence aware dictionary and sentiment reasoner
# list of lexical features that are labeled according to their sentiment(positive negative neutral) and intensity


nltk.download('vader_lexicon')

# Initialize the sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Sample text
text = "I love programming. It's so much fun and rewarding!"

# Perform sentiment analysis
sentiment_scores = sid.polarity_scores(text)

# Print the sentiment scores
print(sentiment_scores)