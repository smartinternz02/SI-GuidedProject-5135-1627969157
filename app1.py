
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("prestone.h5")
                 
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        
        basepath = os.path.dirname(__file__)
        
        filepath = os.path.join(basepath,'uploads',f.filename)
        
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)

        preds = model.predict_classes(x)
      
        index = ['Amethyst','Jade','Rose Quartz','Sapphire']
        text = "The jewellery consists of the precious stone " + str(index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
