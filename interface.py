from flask import Flask , jsonify ,render_template , request
import pandas as pd
import numpy as np
import pickle

with open('Log_model.pkl','rb') as f :
    log_clf = pickle.load(f)


app = Flask(__name__)


@app.route('/')

@app.route('/Homepage')
def Homepage():
    return render_template('home.html')



@app.route('/predict_class',methods=['POST'])
def get_wine_class():
       
    data = request.form
    Glucose                         = eval(data['Glucose '])
    BloodPressure                   = eval(data['BloodPressure '])
    SkinThickness                   = eval(data['SkinThickness '])
    Insulin                         = eval(data['Insulin '])
    BMI                             = eval(data['BMI '])
    DiabetesPedigreeFunction        = eval(data['DiabetesPedigreeFunction '])
    Age                             = eval(data['Age '])

    test_array=np.zeros(7) 
    test_array[0]= Glucose
    test_array[1]= BloodPressure
    test_array[2]= SkinThickness
    test_array[3]= Insulin
    test_array[4]= BMI
    test_array[5]= DiabetesPedigreeFunction
    test_array[6]= Age

    print('Test Array :',test_array)
    predict = log_clf.predict([test_array])[0]
    return render_template('after.html', data=predict)


    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080, debug= False)

