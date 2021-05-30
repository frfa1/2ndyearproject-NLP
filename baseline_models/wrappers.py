from sklearn.feature_extraction.text import CountVectorizer
from models import LogisticRegressionBOW, LogisticRegressionBOWandHand, LogisticRegressionFeatures
from loader import load_train, load_dev, load_hard, load_movies, load_train_handcrafted, load_dev_handcrafted, load_hard_handcrafted, load_movies_handcrafted
from sklearn.metrics import classification_report
from sys import argv

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

def run_ablation_Logistic(train_features,train_labels,test_features,test_labels=None,return_pred=False,verbose=False):
    
    n_handcrafted = train_features.shape[1]
    assert train_features.shape[1] == test_features.shape[1]

    x = [17]
    accuracies = [0.818]

    for i in range(n_handcrafted):
        if verbose:
            print(':: Running ablation study ::')
            print('\nEvaluated features:',n_handcrafted-i)
        train_X = train_features.iloc[:,i:]
        test_X = test_features.iloc[:,i:]
        if verbose:
            print()
        clf = LogisticRegressionFeatures(verbose=True)
        clf.fit(train_X,train_labels)
        if verbose:
            print()
            print(clf.report(test_X,test_labels))
            print()
        x.append(n_handcrafted-i)
        accuracies.append(clf.score(test_X,test_labels))

    return x,accuracies

def main():
    train = load_train(balance=True)
    dev = load_dev(balance=True)
    dev_hard = load_hard(balance=True)
    movies = load_movies()

    train_handcrafted = load_train_handcrafted()
    dev_handcrafted = load_dev_handcrafted()
    hard_handcrafted = load_hard_handcrafted()
    movies_handcrafted = load_movies_handcrafted()

    if 'ut' in argv:
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



    features_to_use = ['irrealis_words_are_present','num_words','elongated_words_are_present',
    'avg_word_length','negation_is_present','num_negations','negation_density','negated_positive',
    'negated_negative','negation_in_first_half','negation_in_first_half','negation_in_second_half',
    'num_question_marks','num_exclamation_marks','emoticon_sentiment','shoutcase_count']

    n,accuracies = run_ablation_Logistic(train_handcrafted[features_to_use],train_handcrafted['sentiment'],movies_handcrafted[features_to_use] ,test_labels=movies_handcrafted['sentiment'],verbose=True )

    print(n)
    print(accuracies)

if __name__ == '__main__':
    main()