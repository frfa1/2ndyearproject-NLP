import pandas as pd
from scipy.sparse import data
from handcrafted import *
import loader
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from datetime import datetime
from sys import argv

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

def make_all(docs,labels, scale=True,keep_text=True, error_info=False, export=False, export_name=None):
    """
    Feed docs and labels as pd.Series objects from a pd.DataFrame and remember to run the
    pd.DataFrame.reset_index(drop=True) method on the dataframe before feeding it to this function!
    - but if you use the loader, this is already done :) 
    """
    features = {}
    features['elongated_words_are_present'] = create_elongated(docs,mode='binary')
    features['irrealis_words_are_present'] = create_irrealis(docs,mode='binary')
    features['num_words'] = create_review_length(docs)
    features['avg_word_length'] = create_avg_word_length(docs)
    features['negation_is_present'] = create_negations_present(docs,mode='binary')
    features['num_negations'] = create_negations_present(docs,mode='count')
    features['negation_density'] = create_negation_density(docs)
    features['negated_positive'] = create_negated_positives(docs)
    features['negated_negative'] = create_negated_negatives(docs)
    features['negation_in_first_half'] = create_negation_discourse(docs)[0]
    features['negation_in_second_half'] = create_negation_discourse(docs)[1]
    features['num_exclamation_marks'] = create_exclamation_marks(docs)
    features['num_question_marks'] = create_question_marks(docs)
    features['emoticon_sentiment'] = create_emoticon(docs)
    features['shoutcase_count'] = create_shoutcase_count(docs)


    if not validate_features(features):
        if error_info:
            write_error_log(features)
        raise ValueError('Inconsistent number of observations in features.')


    if keep_text:
        dataframes = [docs]
    else:
        dataframes = []
    
    tmp = []
    for feature_name,values in features.items():
        tmp.append(pd.DataFrame(data=values,columns=[feature_name]))
    just_features = pd.concat(tmp,axis=1)


    if scale:
        mapper = DataFrameMapper([(just_features.columns, StandardScaler())])
        scaled_features = mapper.fit_transform(just_features.copy(), 4)
        scaled_features_df = pd.DataFrame(scaled_features, index=just_features.index, columns=just_features.columns)
        if keep_text:
            final = pd.concat([docs,scaled_features_df,labels],axis=1) 
        else:
            final = pd.concat([scaled_features_df,labels],axis=1)

    else:
        dataframes.extend(tmp)
        dataframes.append(labels)
        final = pd.concat(dataframes,axis=1)
    if export:
        if not export_name:
            path = '../data/'+safe_filename()+'.json'
            final.to_json(path)
        else:
            final.to_json('../data/'+export_name+'.json')
    return final



def main():
    dev = loader.load_movies()

    if 'ut' in argv:

    
        tmp = make_all(dev['reviewText'],dev['sentiment'],scale=True,keep_text=True,export=False)

        print(tmp['shoutcase_count'])
    else:
        make_all(dev['reviewText'],dev['sentiment'],scale=True,export=True,export_name='movies_handcrafted')

if __name__ == '__main__':
    main()