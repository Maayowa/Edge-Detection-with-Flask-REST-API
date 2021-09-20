"""
app.py
=========================
This is an endpoint for the generation of cartoonized images given an input image

Note: The model used for this endpoint is extracted from the White-box facial image cartoonization repo.
Check https://github.com/SystemErrorWang/FacialCartoonization to read about the paper and the model architecture

"""

from PIL import Image
import cv2 as cv
import os
import base64
import io
from flask.json import jsonify
import numpy as np
from flask import Flask, send_file, Response, request, render_template, make_response
from flask_restful import Api, Resource
import requests
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin 

import torch
from network.inference import SimpleGenerator
from utils import infer, cloud_upload

FILE_UPLOAD_TYPES = ["png", 'jpg', "jpeg"]

model = SimpleGenerator()
wgt_path = os.path.join("network", 'weight.pth')
model.load_state_dict(torch.load(wgt_path, map_location='cpu'))
model.eval()

def send_file(image, public_id = None, send = False):
    """
    Collects a PIL.Image object, converts to byte image and sends it to the cloudinary storage
    """
    fileobj = io.BytesIO()
    image.save(fileobj, format= "PNG", quality = 100)
    fileobj.seek(0)
    if send:
        file_upload = cloud_upload(fileobj, public_id)
        print(file_upload)
        return file_upload
    return fileobj



app = Flask(__name__)
CORS(app)
api = Api(app)



class Homepage(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers)


class Cartoonize(Resource):

    def post(self):
        
        image = request.files["image"]
        filename = secure_filename(image.filename)
        name, ext = filename.split(".")[-2:]
        if len( name ) > 8: name = name[:8]
        

        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:
            file = Image.open( image.stream )
            size = file.size
            file_byte = io.BytesIO()
            file.save(file_byte, "PNG")
            original = base64.b64encode( file_byte.getvalue() ).decode()
            # Pass image through model for an output
            faceimage = infer(model, image)
            faceimage = faceimage.resize(size, Image.LANCZOS)
            file_link = send_file(faceimage, name + "_Cartoon", True)
            

            
            headers = {'Content-Type': 'text/html'}
            return make_response(render_template('result.html', original = "data:image/png;base64," + original, result = file_link),200,headers)

            
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")
        

class GrayCartoonize(Resource):

    def post(self):
        
        image = request.files["image"]
        filename = secure_filename(image.filename)
        name, ext = filename.split(".")[-2:]
        if len( name ) > 8: name = name[:8]
        print("Image Validated")

        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:
            # Pass image through model for an output and convert to grayscale
            file = Image.open( image.stream )
            size = file.size
            file_byte = io.BytesIO()
            file.save(file_byte, "PNG")
            original = base64.b64encode( file_byte.getvalue() ).decode()
            faceimage = infer(model, image)
            print("Transformed")
            faceimage = faceimage.resize(size, Image.LANCZOS)
            cv_img = np.array(faceimage)
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
            image = Image.fromarray(cv_img)
            file_link = send_file(image, name + "_bwcartoon", True)

            
            headers = {'Content-Type': 'text/html'}
            return make_response(render_template('result.html', original = "data:image/png;base64," + original, result = file_link),200,headers)

            #return send_file(fileobj, mimetype="image/PNG")

        
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")

routes = ["/", "/home"]
api.add_resource(Homepage, *routes)
api.add_resource(Cartoonize, "/cartoonize")
api.add_resource(GrayCartoonize, "/bwcartoonize")


if __name__ == "__main__":
    app.run(port= 8080, debug= True)




