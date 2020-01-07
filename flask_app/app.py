from flask import Flask, render_template, url_for, request

import pandas as pd
import pickle
import sklearn
from prediction import get_single_prediction
import sqlite3
import datetime

# conn = sqlite3.connect('database.db')
#
# conn.execute('CREATE TABLE WEB_LOGS (datetime TEXT, text TEXT, pred TEXT)')
#
# conn.commit()
# conn.close()
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        pred = get_single_prediction(message)
        try:
            conn = sqlite3.connect('database.db')
            cur = conn.cursor()
            cur.execute("INSERT INTO WEB_LOGS (datetime, text, pred) VALUES(?, ?, ?)", (datetime.datetime.now(), message, pred) )

            conn.commit()
            msg = "inserted"

        except:
            conn.rollback()
            msg = "error"

        finally:
            print(msg)
            conn.close()

    return render_template('home.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)