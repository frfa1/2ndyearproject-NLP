import pandas as pd

def load_train(drop=True):
    data = pd.read_json('../data/music_reviews_train.json', lines=True)[['reviewText','sentiment']]
    if drop:
        return data.dropna()
    else:
        return data.fillna('')

def load_dev(drop=True):
    data = pd.read_json('../data/music_reviews_dev.json', lines=True)[['reviewText','sentiment']]
    if drop:
        return data.dropna()
    else:
        return data.fillna('')


def main():
    dat = load_train(drop=True)


if __name__ == '__main__':
    main()