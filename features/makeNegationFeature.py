import pandas as pd
import re
import loader
from sys import argv

def create(docs: list, mode='count', regex_pattern='\w+\'\w|\w\w+', export=False) -> list:
    with open('../data/negations.txt') as f: # creates a set of negation words 
        negations = set(word.strip() for word in f.readlines())

    prog = re.compile(regex_pattern)

    def count_negations(search_string,re_object,mode):
        negations_present = [word for word in re_object.findall(search_string.lower()) if word in negations]
        if mode=='count':
            return len(negations_present)
        elif mode=='binary':
            return 1 if negations_present else 0  

    n_negations = [count_negations(sentence,prog,mode) for sentence in docs]

    if export:
        if mode=='count':
            with open('../data/negation_count_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/negation_bin_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')

    return n_negations
            

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    if 'binary' in args:
        out = create(data,export=True,mode='binary')
    if 'count' in args:
        out = create(data,export=True)


if __name__ == '__main__':
    main()