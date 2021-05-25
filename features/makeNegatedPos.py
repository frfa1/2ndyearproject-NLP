import loader
import pandas as pd
import re
from sys import argv

def create(docs: list, regex_pattern='\w+\'\w|\w\w+',mode='count', export=False):
    negations = [word.strip() for word in open('../data/negations.txt').readlines()]
    negative_words = [word.strip() for word in open('../data/positive.txt').readlines()]
    not_negative = set()
    for word in negative_words:
        for negation in negations:
            not_negative.add(' '.join([negation,word]))

    prog = re.compile(regex_pattern)

    def count_negations(search_string,re_object,mode):
        tokens = re_object.findall(search_string)
        pairs = []
        for i in range(len(tokens)-1):
            pairs.append(' '.join([tokens[i],tokens[i+1]]))
        negations_present = [word for word in pairs if word in not_negative]
        if mode=='count':
            return len(negations_present)
        elif mode=='binary':
            return 1 if negations_present else 0  

    n_negations = [count_negations(sentence,prog,mode) for sentence in docs]


    if export:
        if mode=='count':
            with open('../data/negated_positive_count_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/negated_positive_bin_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')

    return n_negations

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

   # search_tokens = [token.strip() for token in open('../data/non_neg.txt').readlines()]

    if 'binary' in args:
        out = create(data,export=True,mode='binary')
    if 'count' in args:
        out = create(data,export=True)

if __name__ == '__main__':
    main()