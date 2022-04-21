from flask import Flask,render_template,request
import os
import shutil
from random import shuffle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
app=Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")



@app.route('/predict', methods= ['POST'])
def predict():
    IMG_SIZE = 128
    imgfile=request.files['imagefile']
    #img_path = "img\cov2.jpeg"
    #img_path = "img\\norm2.jpeg"
    image_path="./img/"+ imgfile.filename
    img = cv2.resize(cv2.imread(image_path),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
    model1 = keras.models.load_model("VGG_model3_128_20ep.h5")
    img = img.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    img = img/255
    img_final = img.reshape(-1,128,128,3)
    preds   = model1.predict(img_final)
    res = np.argmax(preds, axis=1)
    return f"{preds} + \n {res}"



if __name__ == '__main__':
    app.run(port=3000,debug=True)