from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import csv

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['post'])
def predict():
    date =request.form['date']
    
    with open('mout.csv','rt') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if row[0] == date:
                output = row[1]
                output=float(output)
                output = round(output, 6)
                #print(output)

                
                return render_template('result.html',prediction_text ="Predicted Energy Usage is {} kW".format(output))
if __name__=="__main__":
    app.run(debug = True)