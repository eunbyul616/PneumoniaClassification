import argparse
from utils.embedding import EmbeddingTransformation
import json
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
import nltk
from collections import Counter
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessing:
    def __init__(self, config):
        self.config = config

    def read_data(self, file, encoding='euc-kr'):
        self.data = pd.read_csv(file, encoding=encoding)

    def apply_preprocess(self, col_name='description'):
        self.data.reset_index(drop=True, inplace=True)
        self.data['preprocessed'] = self.data[col_name].map(lambda x: self.preprocessing(x))
        print('After preprocessing: ', self.data['preprocessed'].iloc[0])

    def get_max_len(self, col_name='description'):
        self.max_len = max([len(doc) for doc in self.data[col_name]])
        print('Max length: ', self.max_len)

        return self.max_len

    def get_transformed_embedding(self):
        self.source_vec, self.target_vec = EmbeddingTransformation(self.config)

        return self.source_vec, self.target_vec

    def apply_embedding(self):
        self.source_vec, self.target_vec = self.get_transformed_embedding()

        print('Apply Embedding')
        x = np.zeros(self.data.shape[0], self.max_len, self.config['embedding']['embedding_dim'])

        for k, sent in enumerate(self.data['description'].tolist()):
            if k % 500 == 0:
                print('\t -- train {} row ....'.format(k))

            for i, word in enumerate(sent):
                if re.search('[ㄱ-ㅣ가-힣]+', word):
                    x[k][i] = self.source_vec[word]
                else:
                    x[k][i] = self.target_vec[word]
        self.x = x

    def get_x(self):
        return self.x

    def get_y(self, col_name='label'):
        self.y = np.array(self.data[col_name])

        return self.y

    def save(self, file_x, file_y=None):
        np.save(file_x, self.x)

        if file_y:
            y = self.get_y()
            np.save(file_y, y)

        print('Save file')

    def preprocessing(self, text, n=1, stopwords_dict=None, encoding='utf-8', min_token_count=3):
        # Add a space after the period
        text = text.replace('.', '. ').strip()
        sentence = str(text).lower()

        # replace
        # ->, r/o to therefore
        sentence = re.sub('([=r-]+)>|r/o', 'therefore', sentence)
        # date
        sentence = re.sub('[\(]*[0-9]+-[0-9]+-[0-9]+[\)]*', '', sentence)

        # tokenize
        words = []
        for token in sent_tokenize(sentence.replace('.', '. ').strip()):
            words.append(word_tokenize(token))

        # remove stopwords
        stopwords = []
        if stopwords_dict:
            s_list = []
            with open(stopwords_dict, "r", encoding=encoding) as f:
                for line in f.readlines():
                    s_list.append(line.strip())

            stopwords = set(s_list)

        preprocessed_tokens = []
        for ws in words:
            # Remove Korean particles after English
            ws = [re.sub('([a-zA-Z]+)([가-힣]+)', r'\1', w) for w in ws]
            # Remove including numbers
            ws = [w for w in ws if not re.search('[0-9]+', w)]
            preprocessed_tokens.extend([
                re.sub('[-=+,#?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', WordNetLemmatizer().lemmatize(w, pos="v"))
                for w in ws
                if not w in stopwords if len(w) > 1])

        if n != 1:
            ngram_words = []
            if n == 2:
                ngram_words.extend([w[0] + '_' + w[1] for w in ngrams(preprocessed_tokens, n)])
            elif n == 3:
                ngram_words.extend([w[0] + '_' + w[1] for w in ngrams(preprocessed_tokens, n)])
                ngram_words.extend([w[0] + '_' + w[1] + '_' + w[2] for w in ngrams(preprocessed_tokens, n)])

            counter = Counter(ngram_words)
            for k, v in counter.items():
                if v >= min_token_count:
                    if len(k) > 1:
                        preprocessed_tokens.append(k)

        return preprocessed_tokens

    def get_preprocessed_data(self, file, is_save=True, file_x=None, file_y=None):
        self.read_data(file=file)
        self.apply_preprocess()
        self.get_max_len()
        self.get_transformed_embedding()
        self.apply_embedding()

        x = self.get_x()
        y = self.get_y()

        if is_save:
            self.save(file_x, file_y)

        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default='config.json')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    train_data = DataPreprocessing(config)
    train_x, train_y = train_data.get_preprocessed_data(file=config['dataset']['raw']['train_file'])

    test_data = DataPreprocessing(config)
    test_x, test_y = test_data.get_preprocessed_data(file=config['dataset']['raw']['test_file'])
