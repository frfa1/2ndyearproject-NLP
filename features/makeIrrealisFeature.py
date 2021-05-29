import pandas as pd
import re
import loader
from sys import argv

def create(docs: list, mode='count', regex_pattern='\w+\'\w|\w\w+', export=False) -> list:
    irrealis = set(['should','could','would'])
    prog = re.compile(regex_pattern)
    def count_irrealis(search_string, re_object, mode):
        irrealis_present = [word for word in re_object.findall(search_string.lower()) if word in irrealis]
        if mode == 'count':
            return len(irrealis_present)
        elif mode == 'binary':
            return 1 if irrealis_present else 0  
    n_irrealis = [count_irrealis(sentence, prog, mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/irrealis_count_train.txt','w') as ir:
                for line in n_irrealis:
                    ir.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/irrealis_bin_train.txt','w') as ir:
                for line in n_irrealis:
                    ir.write(str(line)+'\n')
    return n_irrealis
            

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    if 'binary' in args:
        out = create(data,export=True,mode='binary')
    if 'count' in args:
        out = create(data,export=True)


if __name__ == '__main__':
    main()