from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('gs_log_reg_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')

def home():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])

def predict():
    try:
        #Get data from form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        # Create a feature array with numpy
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, diabetes_pedigree, age ]])
        
        prediction = model.predict(input_data)
        result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

        return render_template('index.html', prediction_text=f"The patient is: {result}")
    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid numbers for all fields")
    
if __name__== "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)