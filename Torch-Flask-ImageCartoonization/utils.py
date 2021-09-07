import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image



def infer(model, img_path, load_size = 256):
    
    # Load in image
    input_image = Image.open(img_path).convert("RGB")

    # resize, retain aspect ratio
    h, w = input_image.size

    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int( h * 1.0/ w)
    else:
        w = load_size
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
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
    output = np.concatenate([raw_image, output], axis=1)
    #cv2.imwrite(save_path, output)
    output = Image.fromarray(output)

    return output
