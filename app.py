#libraries
from flask import Flask, request, render_template
import pickle
import time
import pandas as pd
from pycaret.regression import *
import numpy as np
#Flask app defining object
app = Flask(__name__)
#every def are function
#this one turning inputs to integer
def parseInt(x):
    return int(x) if x.isdigit() else 0

#home page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

#for upload csv files
@app.route('/test', methods=['GET'])
def test():
    a = pd.read_csv('test.csv')
    return render_template('test.html', model=a)

#about page
@app.route('/about')
def about():
    return render_template('about.html')

#This part is for posting form values into inputList
@app.route('/getResponseLinearReg', methods=["GET", "POST"])
def getResponseLinearReg():
    time.sleep(1)
    #These are user inputs, also model's features
    
    
    DEWPOINT = parseInt(request.form["DEWPOINT"])
    GHI = parseInt(request.form["GHI"])
    OZONE = parseInt(request.form["OZONE"])
    RELATIVEHUMADITY = parseInt(request.form["RELATIVEHUMADITY"])
    SURFACEALBEDO = parseInt(request.form["SURFACEALBEDO"])
    PRECIPITABLEWATER = parseInt(request.form["PRECIPITABLEWATER"])
    CLOUDTYPE = parseInt(request.form["CLOUDTYPE"])
    #Pipeline, we need to convert number to index and give it 1 when others get 0 for onehot-encoder
    CLOUDTYPE_ARR = [0]*12

    if CLOUDTYPE in range(0,12):
        CLOUDTYPE_ARR[CLOUDTYPE] = 1
    del CLOUDTYPE_ARR[5]
    del CLOUDTYPE_ARR[10:12]

    #this is input list which will be model's input
    inputList = [DEWPOINT,
                 GHI,
                 OZONE,
                 RELATIVEHUMADITY, 
                 SURFACEALBEDO, 
                 PRECIPITABLEWATER] + CLOUDTYPE_ARR

    # inputList = [0] * 27

    # pickled_model = pickle.load(open('trainedmodelasd11', 'rb'))

    #pickled model
    with open("validatedlastversionjune", 'rb') as file:
        pickle_model = pickle.load(file)
        #Trained model predict from input list
        y_pred_from_pkl = pickle_model.predict([inputList])
    
    return str(round(y_pred_from_pkl[0]))


if __name__ == '__main__':
    app.run(debug=True)
 