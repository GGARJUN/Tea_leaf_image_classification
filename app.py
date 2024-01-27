import numpy
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
app=Flask(__name__)

model=load_model("vgg_16_Tea_leaves_diseases_model.h5",compile=False)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def index1():
    return render_template("index.html")

@app.route('/about')
def about():
        return render_template('about.html')
    
@app.route('/teahome')
def teahome():
        return render_template('teahome.html')

@app.route('/pred')
def pred():
    return render_template("pred.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(256,256))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']
        text="The Classified TEA LEAF is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run(debug=False, port=8000)