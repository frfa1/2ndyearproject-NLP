import pandas as pd


def load_train(drop=True,balance=True,shuffle=True) -> pd.DataFrame:
    data = pd.read_json('../data/music_reviews_train.json', lines=True)[['reviewText','sentiment']]

    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_train_handcrafted(balance=False):
    data = pd.read_json('../data/train_handcrafted.json')
    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)
    return data


def load_dev(drop=True,balance=False) -> pd.DataFrame:
    data = pd.read_json('../data/music_reviews_dev.json', lines=True)[['reviewText','sentiment']]

    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)

    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_dev_handcrafted(balance=False):
    data = pd.read_json('../data/dev_handcrafted.json')
    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)
    return data

def load_test(drop=True,balance=False) -> pd.DataFrame:
    data = pd.read_json('../data/music_reviews_test_masked.json', lines=True)[['reviewText','sentiment']]
    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_hard(drop=True,balance=False):
    data = pd.read_json('../data/phase_2.json', lines=True)[['reviewText','sentiment']] 
    if balance:
        pos = data.loc[data["sentiment"] == "positive"]
        neg = data.loc[data["sentiment"] == "negative"]
        if len(pos) > len(neg):
            bigger = pos
            smaller = neg
        elif len(neg) > len(pos):
            bigger = neg
            smaller = pos

        bigger = bigger.iloc[:len(smaller)].reset_index(drop=True)
        data = pd.concat([bigger, smaller]).reset_index(drop=True)
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_movies(drop=True):
    """
    Movie review dataset - already balanced
    """
    
    data = pd.read_json('../data/target_data.json')[['reviewText','sentiment']] 
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')


def main():
    train = load_train()
    dev = load_dev()
    test = load_test()
    hard = load_hard()
    movies = load_movies()

    train_handcrafted = load_train_handcrafted()
    dev_handcrafted = load_dev_handcrafted()


if __name__ == '__main__':
    main()