import pandas as pd
import re
import loader
from sys import argv

def create(docs: list, regex_pattern='\w+\'\w|\w\w+', export=False) -> list:
    with open('../data/negations.txt') as f: # creates a set of negation words 
        negations = set(word.strip() for word in f.readlines())

    prog = re.compile(regex_pattern)

    def count_negations(search_string,re_object):
        tokens = [word for word in re_object.findall(search_string.lower())]
        n_half = len(tokens) // 2
        first_half,last_half = tokens[:n_half],tokens[n_half:]
        first = 0
        last = 0
        for negation in negations:
            if negation in first_half:
                first = 1
                break
        for negation in negations:
            if negation in last_half:
                last = 1
        return (str(first),str(last))

    positions = [count_negations(sentence,prog) for sentence in docs]



    if export:
        with open('../data/negation_discourse_train.txt','w') as nf:
            for pair in positions:
                nf.write(','.join(pair)+'\n')

    return positions
            

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    #out = create(data,export=False)

    if 'export' in args:
        out = create(data,export=True)
    


if __name__ == '__main__':
    main()