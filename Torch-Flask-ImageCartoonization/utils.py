import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image

import os
import cloudinary
from dotenv import load_dotenv
load_dotenv()
import cloudinary.uploader 




def cloud_upload(file, public_id, resource = "image"):
    """
    ### Uploads file by resource type to the Cloudinary cloud server for storage
    

    file: resource to be uploaded on the Cloudinary server. 
    resource_type: "image" (default) - Soecifies the type of file to be stored in cloud
    """

    cloudinary.config( 
            cloud_name = os.getenv("CLOUD_NAME"), 
            api_key = os.getenv("API_KEY"), 
            api_secret = os.getenv("API_SECRET")
            )
    print("Configuration Complete")

    upload_result = cloudinary.uploader.upload(file, public_id = public_id, resource_type = resource)
    print("File Sent")

    return upload_result["secure_url"]



def infer(model, img_path, load_size = 256):
    
    # Load in image
    input_image = Image.open(img_path).convert("RGB")

    input_image = input_image.resize((load_size, load_size), Image.BICUBIC)
    raw_image = np.asarray(input_image)

    # Preprocess image for transformation
    image = raw_image/127.5 - 1
    image = image.transpose(2, 0, 1)
    #image = image[:, :, [2, 1, 0]]
    image = torch.tensor(image).unsqueeze(0)
    image = Variable(image).float()

    # Cartoonize
    with torch.no_grad():
        output = model(image)

    # Convert array to image
    output = output.squeeze(0).detach().numpy()
    output = output.transpose(1, 2, 0)
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Plot image side by side
    #output = np.concatenate([raw_image, output], axis=1)
    #cv2.imwrite(save_path, output)
    output = Image.fromarray(output)

    return output
