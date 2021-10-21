"""
app.py
=========================
This is an endpoint for the generation of cartoonized images given an input image

"""

from PIL import Image
import cv2
import os
import json
import base64
import io
from flask.json import jsonify
import numpy as np
from flask import Flask, send_file, Response, request
from flask_restful import Api, Resource
import requests
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin 

from cartooncv import cartoonize, read_image

FILE_UPLOAD_TYPES = ["png", 'jpg', "jpeg"]


def uploader(image, public_id = None, send = False):
    """
    Collects a PIL.Image object, converts to byte image and sends it to the cloudinary storage
    """
    fileobj = io.BytesIO()
    image.save(fileobj, format= "PNG", quality = 100)
    
    if send:
        fileobj.seek(0)
        #file_upload = cloud_upload(fileobj, public_id)
        #print(file_upload)
        #return file_upload
    
    return fileobj



app = Flask(__name__)
CORS(app)
api = Api(app)



class Cartoonize(Resource):


    def post(self):
        
        file = request.files["image"]
        image = file.read()
        filename = secure_filename(file.filename)
        name, ext = filename.split(".")[-2:]
        if len( name ) > 8: name = name[:8]


        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:  
            #convert string data to numpy array
            npimg = np.fromstring(image, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

            out = cartoonize(image)
            res = Image.fromarray(out)
            
            file_byte = uploader(res)
            file_encode = base64.b64encode( file_byte.getvalue() ).decode()

            
            return Response(
                response=json.dumps({"img": file_encode, "warning" : ""}),
                status=200,
                mimetype="application/json"
            )
            
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")
        

class GrayCartoonize(Resource):

    def post(self):

        file = request.files["image"]
        image = file.read()
        filename = secure_filename(file.filename)
        name, ext = filename.split(".")[-2:]
        if len( name ) > 8: name = name[:8]
        

        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:  
            #convert string data to numpy array
            npimg = np.fromstring(image, np.uint8)
            # convert numpy array to image
            image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED) 

            out = cartoonize(image, grey = True)
            res = Image.fromarray(out)
            
            file_byte = uploader(res)
            file_encode = base64.b64encode( file_byte.getvalue() ).decode()

            
            return Response(
                response=json.dumps({"img": file_encode, "warning" : ""}),
                status=200,
                mimetype="application/json")
            
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")


api.add_resource(Cartoonize, "/cartoonize")
api.add_resource(GrayCartoonize, "/bwcartoonize")


if __name__ == "__main__":
    app.run(port= 8080, debug= True)




