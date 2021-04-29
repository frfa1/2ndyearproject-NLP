import pandas as pd
import re
import loader
from sys import argv

def create(docs: list, mode='count', regex_pattern='\w+\'\w|\w\w+', export=False) -> list:
    
    prog = re.compile(regex_pattern)
    regex_elongated = re.compile(r"(.)\1{2}")
    
    def count_elongated(search_string, re_object, mode):
        elongated_present = [word for word in re_object.findall(search_string.lower()) if regex_elongated.search(word)]
        if mode == 'count':
            return len(elongated_present)
        elif mode == 'binary':
            return 1 if elongated_present else 0  

    n_elongated = [count_elongated(sentence, prog, mode) for sentence in docs]

    if export:
        if mode=='count':
            with open('../data/elongated_count_train.txt','w') as ir:
                for line in n_elongated:
                    ir.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/elongated_bin_train.txt','w') as ir:
                for line in n_elongated:
                    ir.write(str(line)+'\n')

    return n_elongated
            

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    if 'binary' in args:
        out = create(data, export=True, mode='binary')
    if 'count' in args:
        out = create(data, export=True)


if __name__ == '__main__':
    main()