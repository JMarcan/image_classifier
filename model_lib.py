# -*- coding: utf-8 -*-
# Function that loads a checkpoint and rebuilds the model

import torch
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models
        
def save_checkpoint(model, checkpoint_path):
        #Save trained model
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

def load_classifier(model, output_categories):
    
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
    classifier_input = model.classifier[0].in_features #input layer of vgg16- has 25088
    classifier_hidden_units = 4096 # 4096 default model value
    
    classifier = nn.Sequential(
        nn.Linear(classifier_input, classifier_hidden_units, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(classifier_hidden_units, output_categories),
        nn.LogSoftmax(dim=1) 
        # Log softmax activation function ensures that sum of all output probabilities is 1 \
        # - With that we know the confidence the model has for a given class between 0-100%
    
    )

    return classifier


   

    
