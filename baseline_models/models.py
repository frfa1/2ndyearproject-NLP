import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sys import argv
from loader import load_train, load_dev

class LogisticRegressionBOW:
    def __init__(self) -> None:
        self.bow_transformer = None
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, text: pd.Series, labels: pd.Series) -> None:
        self.bow_transformer = CountVectorizer(min_df=2).fit(text)
        train_bow = self.bow_transformer.transform(text)
        self.model.fit(train_bow,labels)

    def predict(self, text: pd.Series) -> list:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        return predictions

    def score(self, text: pd.Series,labels: pd.Series) -> float:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        return accuracy_score(labels,predictions)

    def export_predict(self, text: pd.Series) -> None:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        out = [0 if item == 'negative' else 1 for item in predictions]
        df = pd.DataFrame(out, columns=['prediction'])
        df.to_csv('LogisticRegressionPredictions.csv',index_label='id')

    def report(self, text: pd.Series, labels: pd.Series):
        return classification_report(labels, self.predict(text),digits=3)



class LogisticRegressionBOWandHand:
    def __init__(self) -> None:
        self.bow_transformer = None
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, features: pd.Series, labels: pd.Series) -> None:
        self.bow_transformer = CountVectorizer(min_df=2).fit(features['reviewText'])
        train_bow = self.bow_transformer.transform(features['reviewText'])
        all_feat = pd.concat([train_bow,])
        self.model.fit(train_bow,labels)

    def predict(self, text: pd.Series) -> list:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        return predictions

    def score(self, text: pd.Series,labels: pd.Series) -> float:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        return accuracy_score(labels,predictions)

    def export_predict(self, text: pd.Series) -> None:
        test_bow = self.bow_transformer.transform(text)
        predictions = self.model.predict(test_bow)
        out = [0 if item == 'negative' else 1 for item in predictions]
        df = pd.DataFrame(out, columns=['prediction'])
        df.to_csv('LogisticRegressionPredictions.csv',index_label='id')

    def report(self, text: pd.Series, labels: pd.Series):
        return classification_report(labels, self.predict(text),digits=3)


    

def main():
    if 'ut' in argv:
        clf = LogisticRegressionBOW()
        train = load_train(balance=True)
        dev = load_dev() 
        clf.fit(train['reviewText'],train['sentiment'])
        y_pred = clf.predict(dev['reviewText'])
        acc_score = clf.score(dev['reviewText'],dev['sentiment'])
        clf.export_predict(dev['reviewText'])
        print('Unit test finished successfully')
    #else:
       # pass

    




if __name__ == '__main__':
    main()