from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('lgbm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Capture input values from the form
            age = float(request.form['Age'])
            cholesterol = float(request.form['Cholesterol'])
            heart_rate = float(request.form['Heart Rate'])
            bmi = float(request.form['BMI'])
            triglycerides = float(request.form['Triglycerides'])
            exercise_hours = float(request.form['Exercise Hours Per Week'])
            activity_days = float(request.form['Physical Activity Days Per Week'])
            stress_level = float(request.form['Stress Level'])
            sedentary_hours = float(request.form['Sedentary Hours Per Day'])
        except ValueError:
            return render_template('result.html', errors=["Invalid input values"], risk_status=None)

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[age, cholesterol, heart_rate, bmi, triglycerides, 
                                    exercise_hours, activity_days, stress_level, sedentary_hours]], 
                                  columns=['Age', 'Cholesterol', 'Heart Rate', 'BMI', 'Triglycerides', 
                                           'Exercise Hours Per Week', 'Physical Activity Days Per Week', 
                                           'Stress Level', 'Sedentary Hours Per Day'])

        # Scale the features
        input_data = scaler.transform(input_data)

        # Predict the risk
        prediction = model.predict(input_data)
        risk_status = "At Risk" if prediction[0] == 1 else "Not At Risk"

        # Return the result page
        return render_template('display.html', errors=None, risk_status=risk_status)

if __name__ == '__main__':
    app.run(debug=True)




#http://127.0.0.1:5000/

