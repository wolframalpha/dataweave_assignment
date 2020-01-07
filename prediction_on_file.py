retrain_model = True
import pickle
import re

model, tfidf = pickle.load(open('data/model_trainedv2.pic', 'rb'))


def clean_texts(texts):
    texts = [text.lower() for text in texts if text != '']
    # remove hyperlink references e.g. [3]
    texts = [re.sub('(\[\d+\])', '', text) for text in texts]
    # remove special characters like \t
    texts = [re.sub('[^A-Za-z0-9 \.\,]+', ' ', text) for text in texts]
    return texts


def get_predictions(texts):
    X_test = texts
    X_test = clean_texts(X_test)
    X_test = tfidf.transform(X_test)
    y_pred = model.predict(X_test)
    return y_pred


input_filename = "input_test_sample.txt"
output_filename = "output_test_sample.txt"


import argparse

parser = argparse.ArgumentParser(description='prediction args')
parser.add_argument("--ifp", default=input_filename, help="input file path")
parser.add_argument("--ofp", default=output_filename, help="input file path")

args = parser.parse_args()

preds = get_predictions(open(args.ifp, 'r').read().split('\n'))
open(args.ofp, 'w').write('\n'.join(preds))