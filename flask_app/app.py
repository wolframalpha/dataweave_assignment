from flask import Flask, render_template, url_for, request

import pandas as pd
import pickle
import sklearn
from prediction import get_single_prediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        pred = get_single_prediction(message)

    print('hola', pred)
    return render_template('home.html', prediction=pred)




if __name__ == '__main__':
    app.run(debug=True)