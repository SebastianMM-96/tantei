from flask import Flask, render_template, url_for, request
import pickle

# Set the flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Load the model
mlModel = pickle.load(open('models/nbModel.pkl', 'rb'))

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
    message = request.form['message']
    myPrediction = mlModel.predict([message])
    return render_template('tantei.html', prediction = myPrediction)

# Run the app
if __name__== '__main__':
    # Run the app
    app.run()