from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
import pandas as pd
import re

# Set the flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Load the model
mlModel = pickle.load(open('models/nbModel.pkl', 'rb'))
# Tfidf algorithm
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
# CountVectorizer algorithm
countVec = pickle.load(open('models/countVectorizer.pkl', 'rb'))

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# App page
@app.route('/tantei', methods=['GET'])
def tantei():
    return render_template('tantei.html')

# Prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        message = request.form['message']
        myPrediction = mlModel.predict([message])

        probability = mlModel.predict_proba([message])

        maxProba = np.amax(probability)
        maxProba = format(maxProba, ".2%")

        print(maxProba)
    
    return render_template('tantei.html', prediction = myPrediction, accuracy = maxProba)

# Run the app
if __name__== '__main__':
    # Run the app
    app.run()