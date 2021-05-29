from sklearn.feature_extraction.text import CountVectorizer
from models import LogisticRegressionBOW
from loader import load_train, load_dev, load_hard, load_movies
from sklearn.metrics import classification_report

def run_logistic(train_features,train_labels,test_features,test_labels=None,return_pred=False):
    clf = LogisticRegressionBOW()
    clf.fit(train_features,train_labels)
    print(clf.report(test_features,test_labels))
    if return_pred:
        return clf.predict(test_features)


def main():
    train = load_train(balance=True)
    dev = load_dev(balance=True)
    dev_hard = load_hard(balance=True)
    movies = load_movies()
    print('Logistic regression trained on BOW of balanced music reviews train, tested on balanced BOW of music reviews dev:\n')
    run_logistic(train['reviewText'],train['sentiment'],dev['reviewText'],test_labels=dev['sentiment'])
    print('\nLogistic regression trained on balanced BOW of music reviews train, tested on balanced BOW of hard cases:\n')
    run_logistic(train['reviewText'],train['sentiment'],dev_hard['reviewText'],test_labels=dev_hard['sentiment'])
    print('\nLogistic regression trained on balanced BOW of music reviews train, tested on balanced BOW of movie reviews:\n')
    run_logistic(train['reviewText'],train['sentiment'],movies['reviewText'],test_labels=movies['sentiment'])


if __name__ == '__main__':
    main()