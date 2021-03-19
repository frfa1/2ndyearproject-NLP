# %%

import pandas as pd
import nltk
import re
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics.scores import accuracy
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('PorterStemmer ')
#nltk.download('averaged_perceptron_tagger')


class UnigramModel:
    def __init__(
            self, 
            stem_words=False
        ):
        self.stem_words: bool = stem_words
        self.all_words: list = None
        self.samples: list = None
        self.word_dist: dict = None
        self.most_frequent = None
        self.docs: list = None
        self.stop_words = list(set(stopwords.words('english')))
        self.porter = PorterStemmer()
        self.feature_sets: list = None
        self.model = None

    def _tokenize(self, text: str) -> list:
        cleaned = re.sub(r'[^(a-zA-Z)\s]','', text)
        tokens = word_tokenize(cleaned)
        if self.stem_words:
            return [self.porter.stem(word) for word in tokens if word not in self.top_words]
        else:
            return [word.lower() for word in tokens if word not in self.stop_words]

    def _make_docs(
            self,
            text,
            labels
        ):
        docs, tokens = [], []
        for index, value in text.items():
            docs.append((value,labels[index]))
            tokens.extend(self._tokenize(value))
        return docs, tokens

    def _get_word_distribution(self, words: list) -> None:
        distribution = nltk.FreqDist(words)
        return distribution

    def _get_n_most_frequent(self, n: int):
        return sorted(self.word_dist.items(), key=lambda item: item[1],reverse=True)[:n]

    def _get_features(self, text):
        if not self.most_frequent:
            raise ValueError('Missing frequency distribution')
        words = self._tokenize(text)
        features = {}
        for word in words:
            features[word] = (word in self.most_frequent)
        return features

    def fit(
            self, 
            text, 
            labels, 
            n: int = 5000
        ):
        print('Fitting model ...\n\tCreating doc list and tokenizing all words ...')
        self.docs, self.all_words = self._make_docs(text, labels)
        print('\tCalculating word distribution ...')
        self.word_dist = self._get_word_distribution(self.all_words)
        print('\tRetriving the',n,'most frequent words ...')
        self.most_frequent = self._get_n_most_frequent(n)
        print('\tCreating feature sets ...')
        self.feature_sets = [(self._get_features(review), sentiment) for (review,sentiment) in self.docs]
        print('\tFitting model ...')
        self.model = NaiveBayesClassifier.train(self.feature_sets)
        print('Finished fitting successfully!')

    def predict(self, text):
        test_feature_sets = []
        for _, value in text.items():
            test_feature_sets.append(self._get_features(value))
        return self.model.classify_many(test_feature_sets)


def  main():
    data = pd.read_json('data/music_reviews_train.json', lines=True)[['reviewText','sentiment']].dropna()
    reviews, targets = data['reviewText'], data['sentiment']

    clf = UnigramModel()
    clf.fit(reviews, targets)

    use_test = True
    if use_test:
        unseen = pd.read_json('data/music_reviews_test.json', lines=True)[['reviewText','sentiment']].dropna()
    else:
        unseen = pd.read_json('data/music_reviews_dev.json', lines=True)[['reviewText','sentiment']].dropna()
    new_reviews, new_targets = unseen['reviewText'], unseen['sentiment']

    y_pred = clf.predict(new_reviews)

    print(accuracy(new_targets,y_pred))

if __name__ == '__main__':
    main()

# %%
