import pandas as pd
import nltk
from string import punctuation
from sys import argv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump

"""
Run this e.g. with 
>>> python NaiveBayesNGrams.py 2 save
for a 
"""


class NaiveBayesClassifier:
    def __init__(self,dump_model=False, write_csv=False):
        self.dump_model = dump_model
        self.write_csv = write_csv
        self.is_fitted = False
        self.model = None
        self.bow_transformer = None
        self.tfidf_transformer = None

    def _preprocess_text(self, text: str) -> list:
        puncs = punctuation
        stops = stopwords.words('english')
        wo_punctuation = ''.join([char.lower() for char in text if char not in puncs])
        wo_stops = ' '.join([word for word in wo_punctuation.split() if word not in stops])
        return wo_stops

    def fit(self, text: pd.Series, labels: pd.Series, n: int) -> None:
        preprocessed_text = text.apply(self._preprocess_text).tolist()
        self.bow_transformer = CountVectorizer(ngram_range=(1,n)).fit(preprocessed_text)
        train_bow = self.bow_transformer.transform(preprocessed_text)
        self.tfidf_transformer = TfidfTransformer().fit(train_bow)
        train_tfidf = self.tfidf_transformer.transform(train_bow)
        self.model = MultinomialNB().fit(train_tfidf,labels)
        self.is_fitted = True
        if self.dump_model:
            self.export(self)

    def predict(self, text: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError('Model not fitted')
        preprocessed_text = text.apply(self._preprocess_text).tolist()
        test_bow = self.bow_transformer.transform(preprocessed_text)
        test_tfidf = self.tfidf_transformer.transform(test_bow)
        predictions = self.model.predict(test_tfidf)
        if self.write_csv:
            self.write_predictions(predictions)
        return predictions

    def export(self, model, name: str = 'ngramBaselineNB') -> None:
        if not self.is_fitted:
            raise ValueError('Model not fitted')
        dump(self, name+'.joblib')

    def write_predictions(self, predictions: pd.Series) -> None:
        label2idx = {'negative': 0, 'positive': 1}
        predictions = list(map(label2idx.get, predictions))
        df = pd.DataFrame(predictions, columns=['prediction'])
        df.to_csv('NaiveBayesPredictions.csv',index_label='id')


def main():
    train_path = argv[1]
    ngram = int(argv[2])
    if (len(argv) == 4 and argv[3].lower() == 'save'):
        save_model = argv[3]
    else:
        save_model = False

    train = pd.read_json(train_path, lines=True)[['reviewText','sentiment']].dropna()
    clf = NaiveBayesClassifier(dump_model=save_model)
    clf.fit(train['reviewText'],train['sentiment'],ngram)

if __name__ == '__main__':
    main()
