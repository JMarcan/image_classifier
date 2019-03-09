# -*- coding: utf-8 -*-
# Function that loads a checkpoint and rebuilds the model

import torch
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
        
def save_checkpoint(self, model, checkpoint_path):
        #Save trained model
        model.class_to_idx = self.test_data.class_to_idx  
        model.cpu()
        torch.save({'arch': 'vgg16',
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx},
                checkpoint_path)
        
def load_checkpoint(filepath, device='cuda'):
    
    check = torch.load(filepath, map_location=device)
    
    if check['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif check['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        print("Error: LoadCheckpoint - Model not recognized")
        return 0   
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = check['class_to_idx']
    model.classifier = load_classifier()
        
    model.load_state_dict(check['state_dict'])

    return model

def load_classifier(hidden_units=4096):
    
    #Load classifier for vgg16
    '''
        # VGG16 classifier structure:
        
        (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
    '''
    
    #Classifier parameters
    input_size = 25088 #input layer of vgg16- MUST be 25088
    hidden_sizes = hidden_units #4096 default model value
    output_size = 102 #102 flower categories in the dataset
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_sizes, output_size)),
        ('output', nn.LogSoftmax(dim=1))]))

    return classifier


   

    
