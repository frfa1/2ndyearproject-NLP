from sklearn.feature_extraction.text import CountVectorizer
from models import LogisticRegressionBOW
from loader import load_train, load_dev, load_hard
from sklearn.metrics import classification_report

def run_logistic(train_features,train_labels,test_features,test_labels=None,return_pred=False):
    clf = LogisticRegressionBOW()
    clf.fit(train_features,train_labels)
    print(classification_report(test_labels,clf.predict(test_features)))
    if return_pred:
        return clf.predict(test_features)


def main():
    train = load_train(balance=True)
    dev = load_dev()
    dev_hard = load_hard()
    print('Logistic regression trained on BOW of music reviews train, tested on BOW of music reviews dev:\n')

    run_logistic(train['reviewText'],train['sentiment'],dev['reviewText'],test_labels=dev['sentiment'])

    print('\nLogistic regression trained on BOW of music reviews train, tested on BOW of hard cases:\n')

    run_logistic(train['reviewText'],train['sentiment'],dev_hard['reviewText'],test_labels=dev_hard['sentiment'])




if __name__ == '__main__':
    main()