import pandas as pd
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics

def run_logistic_regression(train,test):
    train_X,train_y = train.iloc[:,:-1], train['sentiment']
    test_X,test_y = test.iloc[:,:-1], test['sentiment']
    clf = sklearn.linear_model.LogisticRegression(max_iter=10000)
    clf.fit(train_X,train_y)
    y_pred = clf.predict(test_X)
    print('Logistic regression accuracy:',round(sklearn.metrics.accuracy_score(test_y,y_pred)*100,2),'%')

def run_multionomial_NB(train,test):
    train_X,train_y = train.iloc[:,:-1], train['sentiment']
    test_X,test_y = test.iloc[:,:-1], test['sentiment']
    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(train_X,train_y)
    y_pred = clf.predict(test_X)
    print('Multinomial NB accuracy:',round(sklearn.metrics.accuracy_score(test_y,y_pred)*100,2),'%')


def main():
    handcrafted_train = pd.read_json('../data/train_handcrafted.json')#[['negation_count','sentiment']]
    handcrafted_dev = pd.read_json('../data/dev_handcrafted.json')#[['negation_count','sentiment']]
    hard = pd.read_json('../data/hard.json')

    # standard logistic regression trained on music reviews, tested on music reviews dev
    run_logistic_regression(handcrafted_train, handcrafted_dev)

    # standard logistic regression trained on music reviews, tested on all hard cases
    run_logistic_regression(handcrafted_train, hard)

    train_labels = handcrafted_train['sentiment']
    

if __name__ == '__main__':
    main()

