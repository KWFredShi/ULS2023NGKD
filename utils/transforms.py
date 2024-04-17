import numpy as np
import imgaug.augmenters as iaa

def flip_horizontal(percentage): return iaa.Fliplr(percentage)

def flip_vertical(percentage): return iaa.Flipud(percentage)

def rotation(dg1,dg2): return iaa.Rotate((dg1,dg2))

def gaussian_blur(sig1,sig2): return iaa.GaussianBlur(sigma=(sig1,sig2))

def translate(x1,x2,y1,y2): return iaa.Affine(translate_percent={"x": (x1, x2), "y": (y1, y2)})

def scale(size1,size2): return iaa.Affine(scale=(size1,size2))