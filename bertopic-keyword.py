import json
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# -------------------------------

def load_texts(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item['title'] + " " + item['content']
                texts.append(text)
            except Exception as e:
                print(f"Error parsing line: {e}")
    return texts


texts = load_texts('')
print(f" {len(texts)}  texts")


# -------------------------------


stop_words = set(stopwords.words('english'))


additional_stopwords = {
    'like', 'get', 'got', 'know', 'just', 'don', 'told', 'said', 'im', 'ive',
    'hes', 'shes', 'theyre', 'cant', 'wont', 'would', 'could', 'should',
    'also', 'maybe', 'well', 'really', 'actually', 'basically', 'literally'
}
stop_words.update(additional_stopwords)
stop_words = list(stop_words)



# -------------------------------

vectorizer_model = CountVectorizer(
    stop_words=stop_words,      
    ngram_range=(1, 2),         
    min_df=2,                   
    max_df=0.95                 
)


# -------------------------------

print("train")
topic_model = BERTopic(
    language="english",
    min_topic_size=5,
    nr_topics="auto",
    vectorizer_model=vectorizer_model  
)

topics, probs = topic_model.fit_transform(texts)

print("complete")
print(f" {len(topic_model.get_topics())} topcis")


# -------------------------------

all_keywords = []

for topic_id in topic_model.get_topics().keys():
    if topic_id == -1:  
        continue
    words_scores = topic_model.get_topic(topic_id)
    if words_scores:
        keywords = [word for word, score in words_scores if word not in stop_words]
        all_keywords.extend(keywords)


top_keywords = list(dict.fromkeys(all_keywords))[:100]  
print(f" {len(top_keywords)} words")
print(top_keywords)


# -------------------------------



flat_text = " ".join(texts).lower()  
words_in_corpus = [word.strip(".,!?\"'()[]{}:;") for word in flat_text.split()]

filtered_words = [word for word in words_in_corpus if word not in stop_words and word.isalpha()]

word_freq = Counter(filtered_words)


keyword_freq = {word: word_freq.get(word, 0) for word in top_keywords}
keyword_freq = {k: v for k, v in keyword_freq.items() if v > 0}



# -------------------------------


wc = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    max_words=100,
    relative_scaling=0.5,
    colormap='plasma',
    collocations=False  
)

wc.generate_from_frequencies(keyword_freq)

plt.figure(figsize=(15, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Keyword Cloud: BERTopic Keywords + Corpus Frequency (Stopwords Removed)", fontsize=16)
plt.tight_layout()


output_path = "bertopic_keyword_cloud_clean.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()



# -------------------------------

freq_df = pd.DataFrame(keyword_freq.items(), columns=['Word', 'Frequency'])
freq_df = freq_df.sort_values(by='Frequency', ascending=False)
freq_df.to_csv("bertopic_keyword_frequencies_clean.csv", index=False)
