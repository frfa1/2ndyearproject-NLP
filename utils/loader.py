import pandas as pd


def load_train(drop=True) -> pd.DataFrame:
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/music_reviews_train.json?token=AAAAWLY264NANGVT3WFCFDLAVYG2A', lines=True)[['reviewText','sentiment']]
    if drop:
       return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

def load_dev(drop=True) -> pd.DataFrame:
    data = pd.read_json('https://github.itu.dk/raw/frph/2ndyearproject/master/data/music_reviews_dev.json?token=AAAAWLYHC7D2KFTGTBH6TXDAVYG5O', lines=True)[['reviewText','sentiment']]
    if drop:
        return data.dropna().reset_index(drop=True)
    else:
        return data.fillna('')

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

if __name__ == '__main__':
    main()