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
import io
import numpy as np
from flask import Flask, send_file, Response, request
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

import torch
from network.inference import SimpleGenerator
from utils import infer

FILE_UPLOAD_TYPES = ["png", 'jpg', "jpeg"]

model = SimpleGenerator()
wgt_path = os.path.join("network", 'weight.pth')
model.load_state_dict(torch.load(wgt_path, map_location='cpu'))
model.eval()



app = Flask(__name__)
api = Api(app)



class Cartoonize(Resource):

    def post(self):
        
        image = request.files["image"]
        filename = secure_filename(image.filename)
        ext = filename.split(".")[-1]

        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:
            # Pass image through model for an output
            faceimage = infer(model, image)
            fileobj = io.BytesIO()
            faceimage.save(fileobj, format= "PNG", quality = 100)
            fileobj.seek(0)

            return send_file(fileobj, mimetype="image/PNG")

        
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")
        

class GrayCartoonize(Resource):

    def post(self):
        
        image = request.files["image"]
        filename = secure_filename(image.filename)
        ext = filename.split(".")[-1]

        if ext.lower() not in FILE_UPLOAD_TYPES:
            return Response({"error": "Unsupported file type"}, 200, mimetype="application/json")

        try:
            # Pass image through model for an output and convert to grayscale
            faceimage = infer(model, image)
            cv_img = np.array(faceimage)
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
            image = Image.fromarray(cv_img)
            fileobj = io.BytesIO()
            image.save(fileobj, format= "PNG", quality = 100)
            fileobj.seek(0)

            return send_file(fileobj, mimetype="image/PNG")

        
        except:
            return Response({"error": "Incompatible image format type"}, 200, mimetype="application/json")


api.add_resource(Cartoonize, "/cartoonize")
api.add_resource(GrayCartoonize, "/bwcartoonize")


if __name__ == "__main__":
    app.run(port= 8080, debug= True)




