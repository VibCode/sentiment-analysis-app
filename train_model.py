import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Sample data
data = {
    "review": [
        "I love this product, it's amazing!",
        "Worst purchase ever, total waste of money.",
        "Absolutely fantastic! Will buy again.",
        "Terrible, it broke after one use.",
        "Really happy with the quality.",
        "Not worth the price.",
        "Exceeded my expectations!",
        "Completely disappointed.",
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]
}

df = pd.DataFrame(data)

# Model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and save model
pipeline.fit(df['review'], df['sentiment'])
joblib.dump(pipeline, 'naive_bayes_model.pkl')
print("Model trained and saved as naive_bayes_model.pkl")
