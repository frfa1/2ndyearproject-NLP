import pandas as pd
from sys import argv
from joblib import load
from NaiveBayes import NaiveBayesClassifier
from sklearn.metrics import accuracy_score

def main():    
    if len(argv) == 2:
        test_path = argv[1]
    else:
        test_path = '../data/music_reviews_dev.json'

    data = pd.read_json(test_path, lines=True)[['reviewText','sentiment']].fillna(' ')
    model = load('baselineNB.joblib')

    predictions = model.predict(data['reviewText'])
    model.write_predictions(predictions)
    #print(accuracy_score(data['sentiment'],predictions))
 
if __name__ == '__main__':
    main()