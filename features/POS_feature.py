import numpy as np
import pandas as pd
import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')

import re
import matplotlib.pyplot as plt

def pos_feature(data):

    all_pos = set()
    pos = []
    for sentence in data:
        pos.append(nltk.pos_tag(sentence))

    for sent in pos:
        for pair in sent:
            all_pos.add(pair[1])

    return pos


#data = [["Hi",  "what", "is", "your", "name", "?"], ["My", "name", "is", "Frederik"]]

#print(pos_feature(data))


