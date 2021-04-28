import pandas as pd
import re

with open('../data/negations.txt') as f:
    negations = set(word.strip() for word in f.readlines())

def count_negations(string,negations,reg,mode='count'):
    negations_present = [word for word in reg.findall(string.lower()) if word in negations]
    if mode == 'count':
        return len(negations_present)
    elif mode == 'binary':
        return 1 if negations_present else 0  
    else:
        raise ValueError('mode parameter must be count or binary')


def main():
    prog = re.compile('\w+\'\w|\w\w+') # this regex pattern searches for words with an apostrophe before the last character or plain words
    text = pd.read_json('../data/music_reviews_train.json', lines=True)['reviewText'].fillna(' ').tolist()
    n_negations = [count_negations(sentence,negations,prog,mode='binary') for sentence in text]
 
    with open('../data/negation_bin_train.txt','w') as nf:
        for line in n_negations:
            nf.write(str(line)+'\n')

if __name__ == '__main__':
    main()