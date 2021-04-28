import pandas as pd
import re
from sys import argv

"""
This script can run as is or with an optional 'binary' argument passed when running the script. 

It will by default create a .txt file based on the music reviews train dataset with rows as counts of
negation words for each review. If on the other han the script is run with the binary argument
it produces zeroes or ones based on the presence of any negation word in a given review.
"""


with open('../data/negations.txt') as f: # creates a set of negation words 
    negations = set(word.strip() for word in f.readlines())

def count_negations(string: str, negations: set, reg, mode='count') -> int: 
    negations_present = [word for word in reg.findall(string.lower()) if word in negations]
    if mode == 'count':
        return len(negations_present)
    elif mode == 'binary':
        return 1 if negations_present else 0  
    else:
        raise ValueError('mode parameter must be count or binary')


def main():
    args = set(argv)
    prog = re.compile('\w+\'\w|\w\w+') # this regex pattern searches for words with an apostrophe before the last character or plain words
    text = pd.read_json('../data/music_reviews_train.json', lines=True)['reviewText'].fillna(' ').tolist()
    
    if 'binary' in args:    
        n_negations = [count_negations(sentence,negations,prog,mode='binary') for sentence in text]
        with open('../data/negation_bin_train.txt','w') as nf:
            for line in n_negations:
                nf.write(str(line)+'\n')
    else:
        n_negations = [count_negations(sentence,negations,prog,mode='count') for sentence in text]
        with open('../data/negation_count_train.txt','w') as nf:
            for line in n_negations:
                nf.write(str(line)+'\n')

if __name__ == '__main__':
    main()