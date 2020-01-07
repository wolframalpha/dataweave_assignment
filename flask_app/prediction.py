retrain_model = True
import pickle
from train import train_model, clean_texts

if retrain_model:
    train_model()
model, tfidf = pickle.load(open('../data/model_trainedv2.pic', 'rb'))


def get_single_prediction(text):
    X_test = [
        text,
    ]
    X_test = clean_texts(X_test)
    X_test = tfidf.transform(X_test)
    y_pred = model.predict(X_test)
    return y_pred[0]

