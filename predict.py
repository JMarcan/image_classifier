'''
    Module is a console application
    that predicts flower for an input image based on trained deep learning model
'''
#  ======== Imports ======== 
import torch
from torch import optim
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
import pandas as pd

import argparse

import model_lib
import image_utils_lib

# ======== Settings ======== 
debug_mode = False #set True if you want to see debug messages

# ======== Utilities ======== 
def print_debug(debug_message):
    if debug_mode == True:
        print (debug_message)
 
# ======== Code ========        
class Predict:
    def __init__(self, checkpoint_path, category_names, device):
        '''
            Innitialize trained model
            Input: 
                checkpoint_path  = path to a trained model to be used
                category_names   = path to mapping between class_ids and class_names
                device           = load for 'cpu' or 'cuda' computation             
        '''
        self._checkpointInitialized = False
        self.device = device
        
        self.model = model_lib.load_checkpoint(checkpoint_path, self.device)
        if self.model == 0:
            print ("LoadCheckpoint: ERROR - Checkpoint load failed")
        else:
            self._checkpointInitialized = True
            print ("LoadCheckpoint: Checkpoint loaded")
            
        with open(category_names, 'r') as f:
            self.label_map = json.load(f)
    
    def predict_image(self, image_path, device, top_k):
        '''
        Classify object with its probabilities
        Input: 
            imgage_path  = path to the image to be classified
            device       = computation to be done on 'cpu' or 'cuda'
            top_k =      = return top K probabilities
        Output:
            printed results: class_ids, class_names, probabilities
                
        '''
        top_probs, top_classes = self.predict(image_path, device, top_k)
        
        #Get names for predicted class_ids
        class_names = self.get_class_name(top_classes)

        print("==== Result for: {} ====".format(image_path))
        print("[Displayed |{0}| most probable classes]\n".format(top_k))
        print("ClassesIDs: {0}".format(top_classes))
        print("Classes_names: {0}".format(class_names))
        print("Probabilities: {0}".format(top_probs))  
    def predict(self, image_path, device, top_k):
        '''
        Classify object with its probabilities
        Input: 
            imgage_path  = path to the image to be classified
            device       = computation to be done on 'cpu' or 'cuda'
            top_k =      = return top K probabilities
        Output:
            top_probs[], top_classes[]
                
        '''
        if self._checkpointInitialized == False:
            print ("ERROR predict_flower: checkpoint is not innitialized")
            return
        
        img = image_utils_lib.process_image(image_path, model_lib.means, model_lib.stds)
        top_probs, top_classes = self._get_probs_classes(img, top_k)
        return top_probs, top_classes

    def _get_probs_classes(self, img, top_k):
        '''
        Classify object with its probabilities
        Input: 
            img     = Numpy image
            top_k   = return top K probabilities
        Output:
            top_probs[], top_classes[]
                
        '''
        #Convert numpy to tensor
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        image_tensor.to(self.device)
        
        #Add information about batch size - 1
        model_input = image_tensor.unsqueeze(0)
    
        model_input.to(self.device)
        self.model.to(self.device)
        self.model.type(torch.FloatTensor)  #Fix from https://knowledge.udacity.com/questions/23625 | Otherwise RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
       
        print_debug("Model input:")
        print_debug(model_input)

        probs_log = self.model(model_input)
        probs = torch.exp(probs_log) 
        
        print_debug("Probs:")
        print_debug(probs)

        
        top_probs, top_classes = probs.topk(top_k)

        print_debug("Top_probs:")
        print_debug(top_probs)
        top_probs = top_probs.detach().numpy().tolist()[0] 
        top_classes = top_classes.detach().numpy().tolist()[0]
        

        print_debug("Top_probs po uprave:")
        print_debug(top_probs)
        
        return top_probs, top_classes
    
    def get_class_name(self, class_ids):
        #Get name(s) for predicted class_id(s)
        idx_to_class = {val: key for key, val in    
                                      self.model.class_to_idx.items()}
        flower_names = []
        for c in class_ids:
            flower_names.append(self.label_map[idx_to_class[c]])
            
        return flower_names
    
    def sanity_check(self, device):
        ''' 
        Test on few samples if predict function/model returns expected flower_id
        '''
        test_results = []
    
        image_path = 'flowers/train/1/image_06734.jpg' #image class 1
        prob, obj_id = p.predict(image_path, device, 1)
        if obj_id[0] == 1:
            test_results.append(True)
        else:
            test_results.append(False)

        image_path = 'flowers/train/2/image_05089.jpg' #image class 2
        prob, obj_id = p.predict(image_path, device, 1)
        if obj_id[0] == 2:
            test_results.append(True)
        else:
            test_results.append(False)

        image_path = 'flowers/train/82/image_01586.jpg' #image class 82
        prob, obj_id = p.predict(image_path, device, 1)
        if obj_id[0] == 82:
            test_results.append(True)
        else:
            test_results.append(False)
        
        print ("=== Sanity test if the model correctly predicts three sample 3 images")
        print ("Each row represents a test result (True/False):")
        for i in test_results:
            print (i)
        print ("=== [/test results end]")
    
# # ======== Main ======== 
        
parser = argparse.ArgumentParser() 
parser.add_argument("input", help="path to an image you want to classify")  
parser.add_argument("checkpoint", help="path to a model checkpoint you want to load. The image is then predicted based on this model")  
parser.add_argument("--top_k", type=int, default=1, help="specify number of top K most likely classes. Default is one")  
parser.add_argument("--category_names", default="cat_to_name.json", help="specify path to your custom mapping of categories to real names. Default is 'cat_to_name.json'")  
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use GPU during the computation. Default is CPU")  


args = parser.parse_args()

image_path = args.input
checkpoint_path = args.checkpoint
top_k = args.top_k
category_names = args.category_names
device = args.gpu

p = Predict(checkpoint_path, category_names, device) 
p.predict_image(image_path, device, top_k)

if debug_mode == True: 
    p.sanity_check(device)




