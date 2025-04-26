from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Initialize Flask
app = Flask(__name__)

# ========================
# DATASET & PREPROCESSING
# ========================

# Load your news dataset; make sure news_dataset.csv exists.
df = pd.read_csv("news_dataset.csv")

# For demonstration, we use the 'text' column.
# You could also use combined content if you have (e.g., title + text).
# If needed, adjust the column names according to your dataset.
df['text'] = df['text'].apply(lambda x: x.lower())
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop))
df.reset_index(drop=True, inplace=True)

# ================================================
# CLASSIFICATION MODEL - using a simple logistic model
# ================================================
X = df['text']
y = df['category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for classification
classifier_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=300))
])

classifier_pipeline.fit(X_train, y_train)

# =====================================
# RECOMMENDATION SYSTEM - content similarity
# =====================================
# Vectorize the news text with TF-IDF for recommendation purposes.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_news(article_id, top_n=5):
    """
    Recommend top_n similar news articles based on cosine similarity.
    """
    similarity_scores = list(enumerate(cosine_sim[article_id]))
    # Sort articles by similarity score (exclude the article itself at index 0)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i for i, score in similarity_scores[1:top_n+1]]
    return df.iloc[recommended_indices]

# =====================================
# FLASK ROUTES
# =====================================

@app.route('/')
def index():
    """
    Home page: Display all news headlines.
    """
    # Create a list of articles with id, title, and category (for display purposes)
    articles = []
    for idx, row in df.iterrows():
        articles.append({
            "id": idx,
            "title": row['title'],
            "category": row['category']
        })
    return render_template('index.html', articles=articles)

@app.route('/article/<int:article_id>')
def article(article_id):
    """
    Article detail page: Show article details, its predicted category,
    and recommended similar articles.
    """
    # Get the article details
    article = df.iloc[article_id]
    
    # Use the classifier to predict category for this article's text.
    predicted_category = classifier_pipeline.predict([article['text']])[0]
    
    # Get recommendations
    recommendations = recommend_news(article_id)
    rec_list = []
    for idx, row in recommendations.iterrows():
        rec_list.append({
            "id": idx,
            "title": row['title'],
            "category": row['category']
        })
    
    return render_template('article.html', article=article, predicted_category=predicted_category, recommended=rec_list)

if __name__ == '__main__':
    app.run(debug=True)
