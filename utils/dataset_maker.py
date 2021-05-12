import pandas as pd
import loader


all_handcrafted = [
    '../data/avg_word_lengths.txt',
    '../data/elongated_bin_train.txt',
    '../data/elongated_count_train.txt',
    '../data/emoticon_count.txt',
    '../data/exclamation_mark_count.txt',
    '../data/irrealis_bin_train.txt',
    '../data/irrealis_count_train.txt',
    '../data/negation_bin_train.txt',
    '../data/negation_count_train.txt',
    '../data/question_mark_count.txt',
    '../data/review_lengths.txt',
    '../data/shoutcase_count.txt'
]

def infer_col_name(path: str):
    return path.split('/')[-1].split('.')[0]

def combine_all(paths=all_handcrafted,labels=None) -> pd.DataFrame:
    dataframes = []
    for f in paths:
        dataframes.append(pd.read_csv(f, header=None,names = [infer_col_name(f)]))
    return pd.concat(dataframes, axis=1)




def main():
    train_labels = loader.load_train()['sentiment'].tolist()
    df = combine_all()
    labels = pd.DataFrame(data=train_labels,columns=['sentiment'])
    res = pd.concat([df,labels],axis=1)
    res.to_json('../data/handcrafted_train.json')

if __name__ == '__main__':
    main()