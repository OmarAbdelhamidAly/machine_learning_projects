import os
from flask import Flask, render_template, request,jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('random_forest_model_stroke.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get JSON data from the request body
            data = request.get_json()
            
            # Extract features from JSON data
            features = [float(data[key]) for key in data]
            
            # Make prediction
            prediction = model.predict([features])[0]
            
            # Return prediction result as JSON
            return jsonify({'prediction': int(prediction)})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)),
            threaded=True, debug=False)
