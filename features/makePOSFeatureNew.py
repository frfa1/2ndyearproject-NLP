import numpy as np
import re
import spacy
import loader
from sys import argv

def get_pos_tags(data, export=False):
    """
    Gets POS tag for all tokens and converts into a list per sentence of the count of POS tags in the sentence.
    """
    nlp = spacy.load("en_core_web_sm")
    pos_tags = []
    
    for review in data:
        sentence = nlp(' '.join(review))
        tags = [token.pos_ for token in sentence]
        pos_tags.append(tags)
    
    tokens = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET",
              "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
              "SCONJ", "SYM", "VERB", "X", "EOL", "SPACE"]
    token2id = {token: i for i, token in enumerate(tokens)}
    
    tmp_list = np.zeros((len(data), len(tokens)))
    
    for idx, tags in enumerate(pos_tags):
        for tag in tags:
            tmp_list[idx, token2id[tag]] += 1
    
    if export:
        with open('../data/pos_count.txt','w') as pc:
            for line in tmp_list:
                #line = line.replace("[", "")
                #line = line.replace("]", "")
                #pc.write(str(line)+'\n')
                pc.write(', '.join(map(repr, line))+'\n')
 
    return tmp_list

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()
    
    if 'export' in args:
        review_length_out = get_pos_tags(data, export=True)


if __name__ == '__main__':
    main()