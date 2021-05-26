import pandas as pd


def load_train(drop=True,balance=True,shuffle=True) -> pd.DataFrame:
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/music_reviews_train.json?token=AAAATMWWTTKZQPE6TATZIMLAW5RQI', lines=True)[['reviewText','sentiment']]

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

def load_train_handcrafted():
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/train_handcrafted.json?token=AAAAWL33WXEDWUYWAQ74Q43AW3FHI')
    return data


def load_dev(drop=True) -> pd.DataFrame:
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/music_reviews_dev.json?token=AAAAWLYHC7D2KFTGTBH6TXDAVYG5O', lines=True)[['reviewText','sentiment']]
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_dev_handcrafted():
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/dev_handcrafted.json?token=AAAAWL4UXHGVLSV4KHWBZB3AW3FPO')
    return data

def load_test(drop=True) -> pd.DataFrame:
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/music_reviews_test_masked.json?token=AAAAWL7FGWYWSROIES2BJZTAVYG6O', lines=True)[['reviewText','sentiment']]
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')     


def main():
    train = load_train()
    dev = load_dev()
    test = load_test()

    train_handcrafted = load_train_handcrafted()
    dev_handcrafted = load_dev_handcrafted()


if __name__ == '__main__':
    main()