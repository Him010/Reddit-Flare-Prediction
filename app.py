# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask,render_template,request,json,send_file
import pickle
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import requests
import numpy as np
import tensorflow as tf
import keras
import os


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)
model = joblib.load(open('NN_model.pkl','rb')) 
model._make_predict_function() 
tfidf = joblib.load(open('Tf-Idf.pkl','rb'))
headers = {'User-Agent': 'Microsoft Edge 41.16299.1004.0'}

def predict(l = ""):
    if request.method == 'POST' and l=="":
        link = request.form["Link"]
    else:
        link = l
    page = requests.get(link, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    title = [soup.find('h1',class_ = "_eYtD2XCVieq6emjKBH3m").text]
    title_vec = tfidf.transform(title)
    with session.as_default():
            with session.graph.as_default():
                prediction = model.predict(title_vec)
    dic = {0: "Business/Finance", 1: "Coronavirus", 2: "Politics", 3: "Non-Political", 
           4: "AskIndia", 5: "Policy/Economy" }
    
    return (title,dic[np.argmax(prediction)])

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/submit',methods = ['POST'])
def submit():
    output = predict()
    title = " ".join(output[0][0].split(" "))
    return render_template('index.html',title='Title : {}'.format(title), prediction_text=' Predicted Flare: {}'.format(output[1]))

@app.route('/automated_testing')
def upload():
    return render_template("test.html")

@app.route('/success',methods = ['POST'])
def file_predict():
    #render_template("test.html")
    if request.method == 'POST':
        file = request.files['files']
        #filename = secure_filename(file.filename)
        
        file.save(os.path.join(r"C:\Users\admin","Downloads",file.filename))
        
        f = open(file.filename,'r').readlines()
        res = {}
        for x in f:
            res[x] = predict(x)[1]
        
        with open('res.json', 'w') as fp:
            json.dump(res, fp)
            
        return send_file('res.json',as_attachment=True)
        

if __name__ == '__main__':
    app.run(debug = True)