import pandas as pd
import loader
from sys import argv

def ShoutcaseCounter(data, export=False):
    """
    Returns -> shoutcase_count: Number of all-caps words of length 3 or more in the review
    """
    shoutcase_count = []
    for review in data:
        # remove words of length 2 or less (Don't want I, A, CD etc. to count as a shoutcase)
        new_review = ' '.join([i for i in review.split() if len(i)>2])

        tokens = new_review.split()
        count = sum(map(str.isupper, tokens))
        shoutcase_count.append(count)

    if export:
        with open('../data/shoutcase_count.txt','w') as scc:
            for line in shoutcase_count:
                scc.write(str(line)+'\n')

    return shoutcase_count

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()
    
    if 'export' in args:
        ShoutcaseCount_out = ShoutcaseCounter(data, export=True)

if __name__ == "__main__":
    main()
