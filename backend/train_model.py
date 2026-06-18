import pickle
import re
import warnings
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

# Suppress warnings for cleaner terminal execution output
warnings.filterwarnings('ignore')

# Download required NLTK resource packages for NLP processing
print("📥 Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize text preprocessing objects
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Executes professional natural language preprocessing (NLP) pipeline
    including case folding, token cleaning, regex operations, and lemmatization.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)                # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # Remove punctuation and symbols
    text = re.sub(r'\d+', '', text)                      # Remove numerical digits
    text = re.sub(r'\s+', ' ', text).strip()             # Standardize excessive whitespace
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ==============================================================================
# 📂 DATASET LOADING & STRUCTURING
# ==============================================================================
print("📂 Loading dataset from storage...")
df = pd.read_csv('../dataset/cyberbullying_tweets.csv')

print(f"✅ Dataset loaded successfully: {len(df)} samples detected.")
print(f"📊 Original dataset schema columns: {df.columns.tolist()}")

# Map raw dataset columns to standard processing target headers
df.rename(columns={
    'tweet_text': 'text',
    'cyberbullying_type': 'label'
}, inplace=True)

print(f"📊 Statistical distribution of target classes:")
print(df['label'].value_counts())

# Execute pre-processing pipeline over textual features
print("🧹 Running textual preprocessing and cleaning pipeline...")
df['cleaned_text'] = df['text'].apply(clean_text)

# ==============================================================================
# 🔧 FEATURE EXTRACTION & DATA SPLITTING
# ==============================================================================
print("🔧 Extracting features via TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Segment dataset partitions into training and evaluation sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 🤖 MODEL EVALUATION & BENCHMARKING
# ==============================================================================
print("\n🤖 Initializing comparative machine learning model evaluation...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

best_model = None
best_accuracy = 0
results = {}

for name, model in models.items():
    print(f"\nTraining Model: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✨ Evaluation Accuracy for {name}: {acc:.4f}")
    
    # Track the model yielding optimal performance metrics
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ==============================================================================
# 🏆 FINAL EVALUATION & EXPORT METRICS
# ==============================================================================
print("\n" + "="*50)
print(f"🏆 OPTIMAL CLASSIFICATION MODEL SELECTING DONE!")
print(f"🎯 Maximum Benchmarked Accuracy: {best_accuracy:.4f}")
print("="*50)

print("\n📊 Comprehensive Classification Metrics Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\n📊 Generated Confusion Matrix Breakdown:")
print(confusion_matrix(y_test, y_pred))

# Validate generalizability over unseen subsets using K-Fold Cross Validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\n🔄 5-Fold Cross-validation baseline distribution: {cv_scores}")
print(f"Mean CV Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Serializing structural weight components to pickle artifacts
print("\n💾 Serializing pipeline model configurations to binaries...")
pickle.dump(best_model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("\n✅ Training pipeline executed completely without exceptions!")
print(f"📁 Core Classifier Model saved to target path: model/model.pkl")
print(f"📁 Vocabulary Vectorizer saved to target path: model/vectorizer.pkl")