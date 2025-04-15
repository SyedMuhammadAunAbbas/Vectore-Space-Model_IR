# Libraries
import os
import json
import math
import streamlit as st
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import string


CORPUS_PATH = "Preprocessed_Corpus"
INDEX_PATH = "inverted_index.json"

with open(INDEX_PATH, 'r', encoding="utf-8") as f:
    inverted_index = json.load(f)


N = len(os.listdir(CORPUS_PATH))


lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text, stopwords):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha() and w not in stopwords]
    pos_tags = pos_tag(words)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(p)) for w, p in pos_tags]
    return lemmas


with open("Stopword-List.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().split())


idf = {}
for term, posting_list in inverted_index.items():
    df = len(posting_list)
    idf[term] = math.log(N / df)


doc_vectors = defaultdict(dict)
for term, posting_list in inverted_index.items():
    for entry in posting_list:
        doc_id = entry["doc_id"]
        tf = entry["tf"]
        normalized_tf = 1 + math.log(tf)  # log tf
        tf_idf_weight = normalized_tf * idf[term]
        doc_vectors[doc_id][term] = tf_idf_weight


def cosine_similarity(query_vec, doc_vec):
    dot = sum(query_vec[t] * doc_vec.get(t, 0.0) for t in query_vec)
    norm_q = math.sqrt(sum(v ** 2 for v in query_vec.values()))
    norm_d = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


st.title("Vector Space Model For Information Retreival")
st.write("Search through the corpus using tf-idf and cosine similarity")

query_input = st.text_area("Enter your query here:")
threshold = st.slider("Threshold for similarity", min_value=0.0, max_value=1.0, value=0.05)

if st.button("Search"):
    query_terms = preprocess(query_input, stopwords)
    query_tf = Counter(query_terms)
    query_vector = {}
    for term, tf in query_tf.items():
        if term in idf:
            normalized_tf = 1 + math.log(tf)
            query_vector[term] = normalized_tf * idf[term]

    scores = []
    for doc_id, doc_vec in doc_vectors.items():
        sim = cosine_similarity(query_vector, doc_vec)
        if sim >= threshold:
            scores.append((doc_id, sim))

    ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)

    if ranked_results:
        st.success(f"{len(ranked_results)} relevant documents found:")
        for doc_id, sim in ranked_results:
            st.markdown(f"**Document {doc_id}** - Similarity: {sim:.4f}")
    else:
        st.warning("No relevant documents found with current threshold.")
