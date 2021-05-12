import makeElongatedFeature
import makeEmoticonFeature
import makeIrrealisFeature
import makeLengthsFeature
import makeNegationFeature
import makePunctuationsFeature
import makeShoutcaseFeature
import loader
import pandas as pd
from datetime import datetime

"""
The featurizer library enables the creation of our handcrafted features from any dataset that contains 
text. It is important the the dataset is initially loaded as a pandas DataFrame and that the make_all 
function recieves its features and labels from this dataset as series. 

"""

def validate_features(features: dict) -> bool:
    lengths = set()
    for value in features.values():
        lengths.add(len(value))
    return len(lengths) == 1

def safe_filename():
    now = str(datetime.now().strftime("%b-%d-%H.%M.%S"))
    return 'dataset_at_' + now

def write_error_log(features: dict) -> None:
    with open('feature_count_log.txt','w') as f:
        for key,val in features.items():
            f.write(''.join([key,' has ',str(len(val)),' observations\n']))

def make_all(docs, labels, use_all=True, error_info=False, export=False, export_name=None):
    """
    Feed docs and labels as pd.Series objects from a pd.DataFrame and remember to run the
    pd.DataFrame.reset_index(drop=True) method on the dataframe before feeding it to this function!
    """
    features = {}
    features['elongated_count'] = makeElongatedFeature.create(docs,mode='count')
    features['elongated_binary'] = makeElongatedFeature.create(docs,mode='binary')
    features['n_emoticons'] = makeEmoticonFeature.EmoticonSentiment(docs)
    features['irrealis_count'] = makeIrrealisFeature.create(docs,mode='count')
    features['irrealis_binary'] = makeIrrealisFeature.create(docs,mode='binary')
    features['avg_review_length'] = makeLengthsFeature.get_review_length(docs)
    features['avg_word_length'] = makeLengthsFeature.get_avg_word_length(docs)
    features['negation_count'] = makeNegationFeature.create(docs,mode='count')
    features['negation_binary'] = makeNegationFeature.create(docs,mode='binary')
    features['exclamation_mark_count'] = makePunctuationsFeature.get_exclamation_marks(docs)
    features['question_mark_count'] = makePunctuationsFeature.get_question_marks(docs)
    features['shoutcase_count'] = makeShoutcaseFeature.ShoutcaseCounter(docs)

    if not validate_features(features):
        if error_info:
            write_error_log(features)
        raise ValueError('Inconsistent number of observations in features.')

    if use_all:
        dataframes = []
        for feature_name,values in features.items():
            dataframes.append(pd.DataFrame(data=values,columns=[feature_name]))
        dataframes.append(labels)
        final = pd.concat(dataframes, axis=1)
        if export:
            if not export_name:
                path = '../data/'+safe_filename()+'.json'
                final.to_json(path)
            else:
                final.to_json(export_name)
        return final


def main():
    dataset = loader.load_hard()
    features,labels = dataset['reviewText'], dataset['sentiment']
    data = make_all(features, labels,export=True,export_name='../data/hard.json')



if __name__ == '__main__':
    main()