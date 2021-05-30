import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sys import argv
from loader import load_train, load_dev, load_train_handcrafted, load_dev_handcrafted
import scipy

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
    def __init__(self,verbose=False) -> None:
        self.bow_transformer = None
        self.model = LogisticRegression(max_iter=1000)
        self.verbose = verbose

    def fit(self, features: pd.Series, labels: pd.Series) -> None:
        if self.verbose:
            print('Preprocessing for fitting...')
        if self.verbose:
            print('\tFitting BOW-object...')    
        self.bow_transformer = CountVectorizer(min_df=2).fit(features['reviewText'])
        if self.verbose:
            print('\tTransforming train data...')    
        train_bow = self.bow_transformer.transform(features['reviewText'])
        if self.verbose:
            print('\tConverting BOW to dense...')    
        bow_numpy = train_bow.todense()
        features_numpy = features.iloc[:,1:].to_numpy()
        if self.verbose:
            print('\tConcatenating BOW and features...')    
        all_features = np.concatenate((bow_numpy,features_numpy),axis=1)
        if self.verbose:
            print('\tConverting to sparse matrix...')
        all_features_sparse = scipy.sparse.csr_matrix(all_features)
        if self.verbose:
            print('Fitting model...')    
        self.model.fit(all_features_sparse,labels)

    def predict(self, features: pd.Series) -> list:
        if self.verbose:
            print('Preprocessing for predicting...')
        if self.verbose:
            print('\tCreating test BOW...')
        test_bow = self.bow_transformer.transform(features['reviewText'])
        if self.verbose:
            print('\tConverting BOW to dense...')  
        test_bow_numpy = test_bow.todense()
        features_numpy = features.iloc[:,1:].to_numpy()
        if self.verbose:
            print('\tConcatenating BOW and features...') 
        all_features = np.concatenate((test_bow_numpy,features_numpy),axis=1)
        if self.verbose:
            print('\tConverting to sparse matrix...')
        all_features_sparse = scipy.sparse.csr_matrix(all_features)
        if self.verbose:
            print('Fitting model...')  
        predictions = self.model.predict(all_features_sparse)
        if self.verbose:
            print('Finished!')  
        return predictions

    def score(self, features: pd.Series,labels: pd.Series) -> float:
        test_bow = self.bow_transformer.transform(features['reviewText'])
        test_bow_numpy = test_bow.todense()
        features_numpy = features.iloc[:,1:].to_numpy()
        all_features = np.concatenate((test_bow_numpy,features_numpy),axis=1)
        all_features_sparse = scipy.sparse.csr_matrix(all_features)
        predictions = self.model.predict(all_features_sparse)
        return accuracy_score(labels,predictions)

    def export_predict(self, features: pd.Series) -> None:
        test_bow = self.bow_transformer.transform(features['reviewText'])
        test_bow_numpy = test_bow.todense()
        features_numpy = features.iloc[:,1:].to_numpy()
        all_features = np.concatenate((test_bow_numpy,features_numpy),axis=1)
        all_features_sparse = scipy.sparse.csr_matrix(all_features)
        predictions = self.model.predict(all_features_sparse)
        out = [0 if item == 'negative' else 1 for item in predictions]
        df = pd.DataFrame(out, columns=['prediction'])
        df.to_csv('LogisticRegressionPredictions.csv',index_label='id')

    def report(self, features: pd.Series, labels: pd.Series):
        return classification_report(labels, self.predict(features),digits=3)




class LogisticRegressionFeatures:
    def __init__(self,verbose=False) -> None:
        self.bow_transformer = None
        self.model = LogisticRegression(max_iter=1000)
        self.verbose = verbose

    def fit(self, features, labels: pd.Series) -> None:
        
        print('Fitting model...')    
        self.model.fit(features,labels)

    def predict(self,features) -> list:
        if self.verbose:
            print('Predicting...')  
        predictions = self.model.predict(features)
        return predictions

    def score(self, features: pd.Series,labels: pd.Series) -> float:
        predictions = self.model.predict(features)
        return accuracy_score(labels,predictions)

    def export_predict(self, text,features: pd.Series) -> None:
        test_bow = self.bow_transformer.transform(text)
        test_bow_numpy = test_bow.todense()
        features_numpy = features.to_numpy()
        all_features = np.concatenate((test_bow_numpy,features_numpy),axis=1)
        all_features_sparse = scipy.sparse.csr_matrix(all_features)
        predictions = self.model.predict(all_features_sparse)
        out = [0 if item == 'negative' else 1 for item in predictions]
        df = pd.DataFrame(out, columns=['prediction'])
        df.to_csv('LogisticRegressionPredictions.csv',index_label='id')

    def report(self, features: pd.Series, labels: pd.Series):
        return classification_report(labels, self.predict(features),digits=3)


    

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
    else:
        pass



if __name__ == '__main__':
    main()