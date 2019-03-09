# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np

def process_image(image_path, means, stds):
    '''
        Process a PIL image into an object that can be used as input to a trained model. Returns numpy
    '''
    
    img = Image.open(image_path)
    img = img.resize((256, 256))
    
    # Crop image
    width, height = img.size
    
    left_pos = (width/2)-(224/2)
    right_pos = (width/2)+(224/2)
    bottom_pos = (height/2)-(224/2)
    top_pos = (height/2)+(224/2)
    
    img = img.crop((left_pos, bottom_pos, right_pos, top_pos))

    np_img = (np.array(img) / 255)

    np_img = (np_img - means)/stds
    
    np_img = np_img.transpose((2, 0, 1))
    return np_img