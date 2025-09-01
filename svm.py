import json
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
stop_words = list(set(stopwords.words('english')))


# -------------------------------
category_map = {
    'Bank/account scams': 'Bank scams',
    'Employment scams': 'Job scams',
    'Government/institution impersonation scams': 'Gov/Institution scams',
    'Other': 'Other',
    'Relationship/dating scams': 'Romance scams',
    'Technical support/e-commerce scams': 'Tech/E-commerce scams',
    'Transfer scams': 'Money transfer scams'
}

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                
                text = item['title'] + " " + item['text']
                label = category_map[item['category']]
                data.append({'text': text, 'label': label})
            except Exception as e:
                print(f"Error parsing line: {e}")
    return pd.DataFrame(data)


df = load_data('')


print(df['label'].value_counts())



# -------------------------------


if len(df) < 2:
    raise ValueError("wrong")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']  
)

print(f" trianNum {len(train_texts)}")
print(f" testNum {len(test_texts)}")


# -------------------------------


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=stop_words,
        max_df=0.95,
        min_df=2
    )),
    ('svm', SVC(
        kernel='linear',           
        C=1.0,                     
        probability=True           
    ))
])


# -------------------------------

print("train")
pipeline.fit(train_texts, train_labels)
print("complete")


# -------------------------------

print("test")
test_preds = pipeline.predict(test_texts)


print("\n (classification_report):")
print(classification_report(test_labels, test_preds))


# -------------------------------


joblib.dump(pipeline, 'scam_svm_model.pkl')
print(f"save as 'scam_svm_model.pkl'")
