import os
from flask import Flask, render_template, request
import joblib

from flask_cors import CORS
app = Flask(__name__)


# Load the trained model
model = joblib.load('random_forest_model_Heart Attack Risk2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]  # Extracting feature values from the form
        prediction = model.predict([features])[0]  # Making prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)),
            threaded=True, debug=False)
