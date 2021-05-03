import numpy as np
import os
import requests
from flask import Flask, request, jsonify, render_template, abort, flash
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import cv2
import time
import imghdr

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['jpg', 'png', 'jpeg']
app.config['UPLOAD_FOLDER'] = "Edge_detect\static" 

def validate(url):
    filename = url.split("/")[-1]
    res = filename.split('.')[-1]
    if res not in app.config["UPLOAD_EXTENSIONS"]:
        flash("Enter valid image url or check for typos")
        abort(404)
    return filename

def edge_detection(path, thresh1, thresh2):
    img = cv2.imread(path, 0)
    img = np.uint8(img)
    img = cv2.blur(img, (3,3))
    canny = cv2.Canny(img, thresh1, thresh2)
    return canny

@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html', img_in = None, img_out = None)
# test link https://i.ytimg.com/vi/hIRjlG-gbuI/maxresdefault.jpg

@app.route('/detect', methods = ['POST'])
def get_data():
    thr_hi = int(request.form['uthresh'])
    thr_low = int(request.form['lthresh'])
    url = request.form['url']
    # Using image url
    if url:
        byte_img = requests.get(url).content
        file = Image.open(BytesIO(byte_img)).convert("RGB")
        filename = validate(url)
        print(filename)
    else:
        file = request.files['image']
        filename = secure_filename(file.filename)
        print(filename)
    # Store uploaded file into 'static' folder 
    filepath = os.path.join(\
                            os.getcwd(), app.config['UPLOAD_FOLDER'],filename)
    print(filepath, thr_hi, thr_low)
    file.save(filepath)

    # Perform edge detection internally and return transofrmed image
    out = edge_detection(filepath, thr_hi, thr_low)
    out_name = "out_" + filename
    outpath = os.path.join(\
                            os.getcwd(), app.config['UPLOAD_FOLDER'],out_name)
    cv2.imwrite(outpath, out)
    
    return render_template('index.html', img_in = filename,  img_out = out_name)
    
    
    
if __name__=='__main__':
    app.run()