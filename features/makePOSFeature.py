import numpy as np
import pandas as pd
import nltk
import re
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def pos_feature(text):

    all_text = []
    max_length = 0

    for line in text:
        token_line = nltk.word_tokenize(line)
        if len(token_line) > max_length:
            max_length = len(token_line)

    for line in text:
        token_line = nltk.word_tokenize(line)
        pos_sent = nltk.pos_tag(token_line)
        
        pos = []
        for pair in pos_sent:
            pos.append(pair[1])

        if len(pos) < max_length:
            pos += ["<PAD>" for pad in range(max_length - len(pos) - 1)]
            print(pos)
            print(len(pos))
        
        all_text.append(pos)

    print(all_text)
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(all_text)
    enc.transform(pos).toarray()


    #[word for word in reg.findall(string.lower()) if word in negations]

    """all_pos = set()
    pos = []
    for sentence in data:
        pos.append(nltk.pos_tag(sentence))

    for sent in pos:
        for pair in sent:
            all_pos.add(pair[1])

    return pos"""

def main():

    import loader
    text = loader.load_train()

    pos_feature(text["reviewText"][:5])
    

if __name__ == '__main__':
    main()


