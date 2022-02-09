
# import the required packages
from flask import Flask, render_template, request
import joblib
import pandas as pd

# instantiate the web-app
app = Flask(__name__)

# load our model pipeline object
model = joblib.load("heart_disease_model.joblib")

# outline the homepage or "default" page
# when a user visits this page, the home function will be run
@app.route("/")
def home():
    return render_template("index.html")

# outline the prediction page
# when a user visits the /predict page, the predict function will be run
@app.route('/predict', methods=['POST'])
def predict():
    
    # get input variables from form
    age = request.form.get('age')
    sex = request.form.get('sex')
    chest_pain_type = request.form.get('chest_pain_type')
    resting_blood_pressure = request.form.get('resting_blood_pressure')
    serum_cholestoral = request.form.get('serum_cholestoral')
    fasting_blood_sugar = request.form.get('fasting_blood_sugar')
    resting_ecg = request.form.get('resting_ecg')
    max_hr = request.form.get('max_hr')
    exercise_induced_angina = request.form.get('exercise_induced_angina')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')
    new_data = pd.DataFrame({"age" : [age], "sex" : [sex], "chest_pain_type" : [chest_pain_type],"resting_blood_pressure" : [resting_blood_pressure],"serum_cholestoral" : [serum_cholestoral],"fasting_blood_sugar" : [fasting_blood_sugar],"resting_ecg" : [resting_ecg],"max_hr" : [max_hr],"exercise_induced_angina" : [exercise_induced_angina],"oldpeak" : [oldpeak],"slope" : [slope],"ca" : [ca],"thal" : [thal]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # render the page using result.html and include the predicted probability
    return render_template("result.html", prediction_text = f"{pred_proba:.0%}")
    
if __name__ == "__main__":
    app.run(debug=True)