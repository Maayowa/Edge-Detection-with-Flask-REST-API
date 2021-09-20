import requests
import io
from PIL import Image
import base64

host = "http://127.0.0.1:5000/cartoonize"

headers = {"content_type":"multipart/form-data"}
files = {"image": ("Ronaldo.jpg", open("./Ronaldo.jpg", "rb"))}

response = requests.post(host, files=files)
print(type(response))


cue = response.json()

img = cue["img"]
img = base64.b64decode(img)
img_bytes = io.BytesIO(img)
img = Image.open(img_bytes)
img.show()

"Torch-Flask-ImageCartoonization/Cartoonization API"
