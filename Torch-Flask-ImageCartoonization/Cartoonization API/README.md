# ImageCartoonization-with-Flask-REST-API

This Image Cartoonization API is based off the implementation of the White-box Facial Cartoonization model. The API provides two endpoints for colored and grayscale representation of the cartoonized for of the input image

The Image Cartoonization API converts image to cartoon form using the [FaceCartoonization](https://github.com/SystemErrorWang/FacialCartoonization) pytorch framework.

Provides support for both:

- Image cartoon
- Grayscale cartoon

### POST	Cartoonize Endpoint

```
https://maayowa-torchtoon.herokuapp.com/cartoonize
```

**API Arguments:**

```
image: image to be converted to cartoon form
```


<br>

### POST	GrayscaleCartoon Endpoint

Returns a grayscale (black and white) cartoon version of the input image

```
https://maayowa-torchtoon.herokuapp.com/cartoonize
```

**API Arguments:**

```
image: image to be converted to cartoon form
```
