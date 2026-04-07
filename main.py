from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the data to get unique locations
data = pd.read_csv('cleaned_bangalore_house_data.csv')
model = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location').strip()
    bhk = int(float(request.form.get('bhk')))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)[0]
    return str(np.round(max(0, prediction), 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
