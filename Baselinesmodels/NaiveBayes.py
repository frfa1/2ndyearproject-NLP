import pandas as pd
import nltk
import string
from sys import argv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump

train_path = argv[1]
if (len(argv) == 3 and argv[2] == 'save'):
    save_model = argv[2]
else:
    save_model = False

class NaiveBayesClassifier:
    def __init__(self,dump_model=False, write_csv=False):
        self.dump_model = dump_model
        self.write_csv = write_csv
        self.is_fitted = False
        self.model = None
        self.bow_transformer = None
        self.tfidf_transformer = None

    def _preprocess_text(self, text: str) -> list:
        puncs = string.punctuation
        stops = stopwords.words('english')
        wo_punctuation = ''.join([char.lower() for char in text if char not in puncs])
        wo_stops = ' '.join([word for word in wo_punctuation.split() if word not in stops])
        return wo_stops

    def fit(self, text: pd.Series, labels: pd.Series) -> None:
        preprocessed_text = text.apply(self._preprocess_text).tolist()
        self.bow_transformer = CountVectorizer(min_df=2).fit(preprocessed_text)
        train_bow = self.bow_transformer.transform(preprocessed_text)
        self.tfidf_transformer = TfidfTransformer().fit(train_bow)
        train_tfidf = self.tfidf_transformer.transform(train_bow)
        self.model = MultinomialNB().fit(train_tfidf,labels)
        self.is_fitted = True
        if self.dump_model:
            self.export(self.model)

    def predict(self, text: pd.Series):
        if not self.is_fitted:
            raise ValueError('Model not fitted')
        preprocessed_text = text.apply(self._preprocess_text).tolist()
        test_bow = self.bow_transformer.transform(preprocessed_text)
        test_tfidf = self.tfidf_transformer.transform(test_bow)
        predictions = self.model.predict(test_tfidf)
        if self.write_csv:
            self.write_predictions(predictions)
        return predictions

    def export(self, model, name='baselineNB'):
        if not self.is_fitted:
            raise ValueError('Model not fitted')
        dump(model, name+'.joblib') 

    def write_predictions(self, predictions) -> None:
        label2idx = {'negative': 0, 'positive': 1}
        predictions = list(map(label2idx.get, predictions))
        df = pd.DataFrame(predictions, columns=['prediction'])
        df.to_csv('NaiveBayesPredictions.csv',index_label='id')


def main():
    train = pd.read_json(train_path, lines=True)[['reviewText','sentiment']].dropna()
    clf = NaiveBayesClassifier(dump_model=save_model)
    clf.fit(train['reviewText'],train['sentiment'])

    test_path = '../data/music_reviews_dev.json'
    test = pd.read_json(test_path, lines=True)[['reviewText','sentiment']].fillna(' ')
    predictions = clf.predict(test['reviewText'])
    clf.write_predictions(predictions)

if __name__ == '__main__':
    main()