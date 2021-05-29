from sklearn.feature_extraction.text import CountVectorizer
from models import LogisticRegressionBOW, LogisticRegressionBOWandHand
from loader import load_train, load_dev, load_hard, load_movies, load_train_handcrafted, load_dev_handcrafted, load_hard_handcrafted, load_movies_handcrafted
from sklearn.metrics import classification_report

def run_logisticBOW(train_features,train_labels,test_features,test_labels=None,return_pred=False):
    clf = LogisticRegressionBOW()
    clf.fit(train_features,train_labels)
    print(clf.report(test_features,test_labels))
    if return_pred:
        return clf.predict(test_features)

def run_LogisticBOWandHand(train_features,train_labels,test_features,test_labels=None,return_pred=False):
    clf = LogisticRegressionBOWandHand()

    clf.fit(train_features,train_labels)
    print(clf.report(test_features,test_labels))
    if return_pred:
        return clf.predict(test_features)

def main():
    train = load_train(balance=True)
    dev = load_dev(balance=True)
    dev_hard = load_hard(balance=True)
    movies = load_movies()

    train_handcrafted = load_train_handcrafted()
    dev_handcrafted = load_dev_handcrafted()
    hard_handcrafted = load_hard_handcrafted()
    movies_handcrafted = load_movies_handcrafted()



    print('\nLOGISTIC REGRESSION ON BOW\n\n')
    print('Logistic regression trained on BOW of balanced music reviews train, tested on balanced BOW of music reviews dev:\n')
    run_logisticBOW(train['reviewText'],train['sentiment'],dev['reviewText'],test_labels=dev['sentiment'])
    print('\nLogistic regression trained on balanced BOW of music reviews train, tested on balanced BOW of hard cases:\n')
    run_logisticBOW(train['reviewText'],train['sentiment'],dev_hard['reviewText'],test_labels=dev_hard['sentiment'])
    print('\nLogistic regression trained on balanced BOW of music reviews train, tested on balanced BOW of movie reviews:\n')
    run_logisticBOW(train['reviewText'],train['sentiment'],movies['reviewText'],test_labels=movies['sentiment'])

    print('LOGISTIC REGRESSION ON BOW AND HANDCRAFTED FEATURES\n\n')
    print('Logistic regression trained on BOW of balanced music reviews train AND handcrafted features, tested on balanced music reviews dev:\n')
    run_logisticBOW(train_handcrafted['reviewText'],train_handcrafted['sentiment'],dev_handcrafted['reviewText'],test_labels=dev_handcrafted['sentiment'])
    print('Logistic regression trained on BOW of balanced music reviews train AND handcrafted features, tested on hard cases:\n')
    run_logisticBOW(train_handcrafted['reviewText'],train_handcrafted['sentiment'],hard_handcrafted['reviewText'],test_labels=hard_handcrafted['sentiment'])
    print('Logistic regression trained on BOW of balanced music reviews train AND handcrafted features, tested on movie reviews:\n')
    run_logisticBOW(train_handcrafted['reviewText'],train_handcrafted['sentiment'],movies_handcrafted['reviewText'],test_labels=movies_handcrafted['sentiment'])


if __name__ == '__main__':
    main()