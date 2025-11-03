"""
NLTK Tutorial: Movie Reviews Analysis
Complete walkthrough of NLTK features with movie reviews dataset
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import RegexpParser
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('movie_reviews.csv')
print("Dataset loaded successfully!")
print(df.head())


# ==================== 1. TEXT NORMALIZATION ====================
def normalize_text(text):
    """Convert text to lowercase, remove punctuation, and tokenize."""
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens


# Apply normalization
df['normalized'] = df['review'].apply(normalize_text)
print("\n1. TEXT NORMALIZATION COMPLETE")
print("Example:", df['normalized'][0][:10])


# ==================== 2. POS TAGGING & NER ====================
def get_pos_ner(text):
    """Get part-of-speech tags and named entities."""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return pos_tags, named_entities


# Apply POS tagging to first 5 reviews
df['pos_tags'] = df['review'].apply(lambda x: get_pos_ner(x)[0])
print("\n2. POS TAGGING COMPLETE")
print("Example POS tags:", df['pos_tags'][0][:5])


# ==================== 3. CHUNKING PATTERNS ====================
def chunk_text(text):
    """Extract noun phrases using chunking patterns."""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Define grammar for noun phrases
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    
    # Extract chunks
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            chunks.append(' '.join(word for word, tag in subtree.leaves()))
    
    return chunks


df['chunks'] = df['review'].apply(chunk_text)
print("\n3. CHUNKING COMPLETE")
print("Example chunks:", df['chunks'][0][:3])


# ==================== 4. SENTIMENT ANALYSIS (VADER) ====================
def analyze_sentiment(text):
    """Analyze sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound']


df['vader_score'] = df['review'].apply(analyze_sentiment)
print("\n4. VADER SENTIMENT ANALYSIS COMPLETE")
print("Example scores:", df[['review', 'vader_score']].head())


# ==================== 5. SUPERVISED CLASSIFICATION ====================
def train_classifier(df):
    """Train a Naive Bayes classifier for sentiment."""
    # Prepare data
    X = df['review']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = CountVectorizer(max_features=100)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, classifier, vectorizer


accuracy, clf, vec = train_classifier(df)
print(f"\n5. SUPERVISED CLASSIFICATION COMPLETE")
print(f"Model Accuracy: {accuracy:.2%}")


# ==================== 6. WORDNET SEMANTIC RELATIONS ====================
def explore_wordnet(word):
    """Get synonyms and definitions from WordNet."""
    synsets = wordnet.synsets(word)
    if synsets:
        syn = synsets[0]
        return {
            'definition': syn.definition(),
            'synonyms': [lemma.name() for lemma in syn.lemmas()][:5]
        }
    return None


# Explore key words
key_words = ['movie', 'fantastic', 'terrible', 'boring']
wordnet_data = {word: explore_wordnet(word) for word in key_words}
print("\n6. WORDNET SEMANTIC RELATIONS")
for word, data in wordnet_data.items():
    if data:
        print(f"{word}: {data['synonyms']}")


# ==================== 7. VISUALIZATION ====================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('NLTK Analysis Results - Movie Reviews', fontsize=16, fontweight='bold')

# Plot 1: Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()
axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
axes[0, 0].set_title('1. Sentiment Distribution')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')

# Plot 2: VADER Score Distribution
axes[0, 1].hist(df['vader_score'], bins=20, color='skyblue', edgecolor='black')
axes[0, 1].set_title('2. VADER Sentiment Scores')
axes[0, 1].set_xlabel('Compound Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Neutral')
axes[0, 1].legend()

# Plot 3: Average Token Length
df['token_count'] = df['normalized'].apply(len)
axes[0, 2].scatter(range(len(df)), df['token_count'], alpha=0.6, color='purple')
axes[0, 2].set_title('3. Token Count per Review')
axes[0, 2].set_xlabel('Review Index')
axes[0, 2].set_ylabel('Number of Tokens')

# Plot 4: POS Tag Distribution (top 10)
all_pos = [tag for tags in df['pos_tags'] for word, tag in tags]
pos_counts = pd.Series(all_pos).value_counts().head(10)
axes[1, 0].barh(pos_counts.index, pos_counts.values, color='coral')
axes[1, 0].set_title('4. Top 10 POS Tags')
axes[1, 0].set_xlabel('Count')

# Plot 5: Classification Accuracy
axes[1, 1].bar(['Model Accuracy'], [accuracy], color='lightgreen', width=0.4)
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_title('5. Classifier Performance')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].text(0, accuracy + 0.05, f'{accuracy:.2%}', ha='center', fontweight='bold')

# Plot 6: Average Chunks per Sentiment
df['chunk_count'] = df['chunks'].apply(len)
chunk_avg = df.groupby('sentiment')['chunk_count'].mean()
axes[1, 2].bar(chunk_avg.index, chunk_avg.values, color=['green', 'gray', 'red'])
axes[1, 2].set_title('6. Avg Noun Phrases by Sentiment')
axes[1, 2].set_xlabel('Sentiment')
axes[1, 2].set_ylabel('Average Chunks')

plt.tight_layout()
plt.show()


