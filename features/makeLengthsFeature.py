import pandas as pd
import re
import loader
from sys import argv


def get_review_length(data, regex_pattern="\w+\'\w|\w\w+", export=False):
    """
    Returns
        review_length: The length of the review in words (excludes punctuations)
    """
    re_object = re.compile(regex_pattern)
    
    review_length = [len([word for word in re.findall(regex_pattern, review)]) for review in data]
    
    if export:
        with open('../data/review_lengths.txt','w') as rl:
            for line in review_length:
                rl.write(str(line)+'\n')
    
    return review_length

def get_avg_word_length(data, regex_pattern="\w+\'\w|\w\w+", export=False):
    
    re_object = re.compile(regex_pattern)
    #review_length = [sum(len([word for word in re.findall(regex_pattern, review)])) for review in data]
    word_length = [sum(len(word) for word in re.findall(regex_pattern, review)) for review in data] #get length of all words
    review_length = [len([word for word in re.findall(regex_pattern, review)]) for review in data] #get length of review
    
    avg_word_length = []
    for i, j in zip(word_length, review_length):
        try:
            avg_word_length.append(i/j)
        except ZeroDivisionError:
            avg_word_length.append(0)    
        
    if export:
        with open('../data/avg_word_lengths.txt','w') as rl:
            for line in avg_word_length:
                rl.write(str(line)+'\n')
    
    return avg_word_length


def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()
    
    if 'export' in args:
        review_length_out = get_review_length(data, export=True)
        avg_token_length_out = get_avg_word_length(data, export=True)

if __name__ == '__main__':
    main()