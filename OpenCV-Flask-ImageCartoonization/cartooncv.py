import cv2
import sys
import numpy as np
import argparse
from arrays_img import array_display

parser = argparse.ArgumentParser(description="Takes argument for Image cartoonization",
                                    epilog="\u00A9 Oluwabunmi Iwakin")

# Adding arguments
parser.add_argument('path',type=str, help=': image dir')
parser.add_argument('-g', action= 'store_true', help=': option for grayscale image')


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)

    #cv2.imshow("Image", img)
    return img

def edge_mask(img: np.ndarray, line_size: int, nblur: int):

    # Extracts edges from an image with specified line size by
    # first converting to a binary image, reducing image noise before extracting egdes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, nblur)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_size, nblur)
    return edges

def edge_mask2(image, sigma=0.33):
    
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    edged = (255-edged)
    
    return edged


def color_quantization(img, n):
    """
        Performs color quantization. In simpler terms, this module the Kmeans algorithm to cluster 
        all colors into k total colors, thereby quantizing or removing some colors in the process
    """

    # Convert to float to make clustering faster and reshape into 1D array of pixels and 3 color channels
    qimg = np.float32(img).reshape((-1, 3))

    # Splitting criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    ret, label, center = cv2.kmeans( qimg, n, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS) # KMEANS_RANDOM_CENTERS, KMEANS_PP_CENTERS
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


def cartoonize(img, grey = False):
    # Combining all modules together

    cartoon_edge = edge_mask(img, 5, 7)
    # cartoon_edge = edge_mask2(img)  # appx. 4x faster but edge is not as pronounced as desired
    
    cartoon_colors = color_quantization(img, 9)
    filter_colors = cv2.bilateralFilter(cartoon_colors, 5, 100, 150)

    # Combining lines and colors
    cartoon = cv2.bitwise_and(filter_colors, filter_colors, mask = cartoon_edge)
    
    if grey:
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
    else:       
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    
    return cartoon


if __name__ == "__main__":
    
    args= parser.parse_args()
    
    path = args.path
    grey = args.g

    img = read_image(path)
    out = cartoonize(img, grey)
    array_display(out)