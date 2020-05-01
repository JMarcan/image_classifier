# -*- coding: utf-8 -*-
# Function that loads a checkpoint and rebuilds the model

import torch
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models
        
def save_checkpoint(model, checkpoint_path, output_categories):
        '''
        Save the trained deep learning model

        Args:
            model: trained deep learning model to be saved
            checkpoint_path(str): file path where model will be saved
            output_categories(int): number of output categories recognized by the model

        Returns:
            None
        '''
        model.cpu()
        torch.save({'arch': 'vgg16',
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx,
                'output_categories': output_categories
                },checkpoint_path)
        
def load_checkpoint(checkpoint_path, device='cuda'):
    '''
    Loads trained deep learning model

    Args:
        checkpoint_path(str): file path where model will be saved

    Returns:
        model: loaded deep learning model
    '''
    check = torch.load(checkpoint_path, map_location=device)
    
    if check['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif check['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        print("Error: LoadCheckpoint - Model not recognized")
        return 0   
    
    output_categories = 2
    
    try:
        if check['output_categories'] >= 2:
            output_categories = check['output_categories']
        else:
            print("Error: LoadCheckpoint - Saved model output categories has invalid value ({0}). Value needs to be 2 or higher.".format(check['output_categories']))
            return 0
    except Exception as e: # when ['output_categories'] is not part of save model
        print("Error: LoadCheckpoint - Saved model does not contain information about output categories: {0}".format(e))
        return 0
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = check['class_to_idx']
    model.classifier = load_classifier(model, output_categories)
        
    model.load_state_dict(check['state_dict'])

    return model

def load_classifier(model, output_categories):
    '''
    Loads the classifier that we will train

    Args:
        model: deep learning model for which we create the classifier
        output_categories(int): number of output categories 
                                recognized by the model

    Returns:
        classifier: loaded classifier for a given model
    '''
    
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


   

    
