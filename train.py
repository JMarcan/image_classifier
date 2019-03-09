import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.optim import lr_scheduler

import argparse
import time
import copy

import model_lib

class Train:
    def __init__(self, data_dir, model_name, hidden_units, device):
        
        # Define  transforms for the training, validation, and testing sets

        train_transforms = transforms.Compose([transforms.RandomRotation(45), #Image augmentation
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(model_lib.means,
                                                            model_lib.stds)])

        testing_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(model_lib.means,
                                                            model_lib.stds)])

        validation_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(model_lib.means,
                                                            model_lib.stds)])
    
        # Load the datasets with ImageFolder
        
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        self.train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)
        self.validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

        # Using the image datasets and the trainforms, define the dataloaders

        self.dataloaders = {'train': torch.utils.data.DataLoader(self.train_data, batch_size=16, num_workers=0, shuffle=True),
              'test': torch.utils.data.DataLoader(self.test_data, batch_size=16, num_workers=0),
               'validation': torch.utils.data.DataLoader(self.validation_data, batch_size=16, num_workers=0)
              }
        self.dataset_sizes = {'train': len(self.dataloaders['train']),
              'test': len(self.dataloaders['test']),
               'validation': len(self.dataloaders['validation'])}
        
        self.model = self._load_model(model_name)
        
        #Freeze features parameters
        for param in self.model.parameters():
            param.require_grad = False
            
        #feed-forward network
        self.model.classifier = model_lib.load_classifier(hidden_units)
        self.model.class_to_idx = self.train_data.class_to_idx
        
    def _load_model(self, model_name):
        if (model_name == "vgg16"):
            return models.vgg16(pretrained=True)
        elif model_name == "vgg13":
            return models.vgg13(pretrained=True)
        else:
            raise "Error _load_model: Only vcgg16 and vcgg13 models are supported"
                

    def train_model(self, learning_rate, epochs, device):
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr = 0.003)

        # Udacity - PyTorch V2 Part 8 Solution V1 - https://www.youtube.com/watch?v=4n6T93hKRD4&feature=youtu.be

        self.model.to(device)
    
        steps = 0
        running_loss = 0
        print_every = 20
    
        for epoch in range(epochs):
            for images, labels in self.dataloaders["train"]:
                steps += 1
            
                self.model.train()
            
                images, labels = images.to(device), labels.to(device)
            
                optimizer.zero_grad()
            
                logps = self.model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
            
                if steps % print_every == 0:
                    self.model.eval()
                
                    test_loss = 0
                    accuracy = 0
                
                    for images, labels in self.dataloaders["validation"]:
                    
                        images, labels = images.to(device), labels.to(device)

                        logps = self.model(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))
                    
                    print("Epoch {0}/{1}".format(epoch+1, epochs))
                    print("Steps {0}/{1}".format(steps, self.dataset_sizes["train"]))
                    print("Train loss {:.2}".format(running_loss/print_every))
                    print("Validation loss {:.2}".format(test_loss/self.dataset_sizes["validation"]))
                    print("Validation accuracy {:.2%}".format(accuracy/self.dataset_sizes["validation"]))
        
        self.save_checkpoint(self.model, self.checkpoint_path)
        print ("Checkpoint saved")
        self._validation(device)
        
        def _validation(self, device):
            self.model.to(device)
    
            self.model.eval()
                
            test_loss = 0
            accuracy = 0
    
            with torch.no_grad(): # Turn off gradients for validation, saves memory and computations
                for images, labels in dataloaders["test"]:
                    
                    images, labels = images.to(device), labels.to(device)

                    logps = self.model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
                    
            print("Validation loss {:.2}".format(test_loss/dataset_sizes["test"]))
            print("Validation accuracy {:.2%}".format(accuracy/dataset_sizes["test"])) 
# ============= main ==============
parser = argparse.ArgumentParser() 
parser.add_argument("data_directory", help="path to an folder containing data folders (training, validation, testing)")  
parser.add_argument("--save_dir", default="model/checkpoint_script.pth", help="location where to save a model checkpoint. Default is 'model/checkpoint_script.pth'")  
parser.add_argument("--arch", default="vgg16", choices=["vgg16", "vgg13"], help="Select model for transfer learning. Default is vgg16")  
parser.add_argument("--learning_rate", type=float, default=0.01, help="specify learning rate for the model. Default is 0.01")  
parser.add_argument("--hidden_units", type=int, default=4096, help="specify number of hidden layers for the model. Default is 4096")  
parser.add_argument("--epochs", type=int, default=20, help="specify number of epochs for training. Default is 20")  
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use GPU during the computation. Default is CPU")  


args = parser.parse_args()

data_directory = args.data_directory
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epoch = args.epochs
device = args.gpu

train_cl = Train(data_directory, arch, hidden_units, device)    
train_cl.train_model(learning_rate, epoch, device)
        