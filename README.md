# Dataweave Assignment

Install the requirements from `./requirements.txt` file

### 1. Prediction on files:
**filename**: `prediction_on_file.py`  
**usage**:  `python prediction_on_file.py --ifp input_test_sample.txt --ofp output_test_sample.txt`
### 2. Webapp:
- API created on flask (`./flask_app/`)  
- run the flask server using `python ./flask_app/app.py`  
- go to http://127.0.0.1:5000/ for the UI
- the web app uses SQLite db and the file is located at `./flask_app/database.db`
### 3. Training notebook:
located at `./train.ipynb`
#### training process (self explanatory details in the notebook):
- The wikipedia text was split and the model was trained on this splits 
- SVM with linear kernel was used along with class weights to address the imbalance problem 



