from sklearn.feature_extraction.text import CountVectorizer
from models import LogisticRegressionBOW
from loader import load_train, load_dev

def run_logistic(train_features,train_labels,test_features,test_labels=None,return_pred=False):
    clf = LogisticRegressionBOW()
    clf.fit(train_features,train_labels)
    print(clf.score(test_features,test_labels))
    if return_pred:
        return clf.predict(test_features)


def main():
    train = load_train(balance=True)
    dev = load_dev()
    run_logistic(train['reviewText'],train['sentiment'],dev['reviewText'],test_labels=dev['sentiment'])


if __name__ == '__main__':
    main()