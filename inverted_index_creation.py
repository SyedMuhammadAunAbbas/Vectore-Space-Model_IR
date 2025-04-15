import os
import json
from collections import defaultdict, Counter

""" 
I have chosen inverted index as the appropriate data structure for holding posting lists and term frequency for each 
lexicon/vocabulary-term in our courpus. I made this choice to ensure fast lookup and retreival for effecient processing on query arrival.

For each lexicon, the inverted index will hold:

lexicon : (doc_id, tf) , .....

where doc_id is the document where the term appears in, and tf is the term frequency.
For idf calculation, the process becomes very straightforward. 
As df simply becomes len(posting-list(term)), and idf= log(N/df).
"""

RSLT_CORPUS = "Preprocessed_Corpus"

inverted_index = defaultdict(list)

documents = sorted(
    os.listdir(RSLT_CORPUS),
    key=lambda x: int(os.path.splitext(x)[0])
)

for doc_id, document in enumerate(documents, start=1):
    path = os.path.join(RSLT_CORPUS, document)

    with open(path, 'r', encoding="utf-8") as f:
        word_list = f.read().split()
        term_freq = Counter(word_list)  

    for word, tf in term_freq.items():
        inverted_index[word].append({"doc_id": doc_id, "tf": tf})

with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=4)
