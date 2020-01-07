from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle


def clean_texts(texts):
    texts = [text.lower() for text in texts if text != '']
    # remove hyperlink references e.g. [3]
    texts = [re.sub('(\[\d+\])', '', text) for text in texts]
    # remove special characters like \t
    texts = [re.sub('[^A-Za-z0-9 \.\,]+', ' ', text) for text in texts]
    return texts


def train_model():
    apple_company = open('../data/apple-computers.txt').read()
    apple_fruit = open('../data/apple-fruit.txt').read()

    apple_company = apple_company.split('\n')
    apple_fruit = apple_fruit.split('\n')



    apple_company = clean_texts(apple_company)
    apple_fruit = clean_texts(apple_fruit)

    df = pd.DataFrame(zip(apple_company + apple_fruit,
                          len(apple_company) * ['computer-company'] + len(apple_fruit) * ['fruit']),
                      columns=['text', 'label'])

    from sklearn.svm import SVC

    X = df['text'].values
    y = df['label'].values

    model = SVC(C=0.9, class_weight={'computer-company': 1, 'fruit': 1.5}, kernel='linear')

    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    X = tfidf.fit_transform(X)
    model.fit(X, y)

    pickle.dump([model, tfidf], open('../data/model_trainedv2.pic', 'wb'))

