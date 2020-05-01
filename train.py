'''
    Module is a console application that trains deep learning model 
    to predict class for an input image
'''
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
    def __init__(self, data_dir, output_categories, means, stds, model_name):
        
        '''
            Initialize deep learning model to be trained
            
            Args:
                data_dir(str): directory containing:
                    1. folder 'train' with training data
                    2. folder 'valid' with validation data used during training
                    3. folder 'test' with testing data used after training
                    4.  file 'cat_to_name.json' with mapping between label ids and label names
                output_categories(int): number of output labels that our model shall identify
                means(array): means for the dataset e.g. [0.485, 0.456, 0.406]
                stds(array): stds for the dataset e.g. [0.229, 0.224, 0.225]
                model_name(str): Chose model to be trained ("vgg16" or "vgg13")
                    
            Returns:
                None
        '''
        # Define folders
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Define number of possible categories that our network must recognize in your data
        self.labels_output_categories = output_categories

        # File with maping between label ids and their names
        self.label_map_file = data_dir + "/cat_to_name.json"

        # Define means and stds for images
        self.means = means
        self.stds = stds
        
        # Define  transforms for the training, validation, and testing sets
        train_transforms = transforms.Compose([
            # data augmentation
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            # data normalization
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])

        testing_transforms = transforms.Compose([
            # data normalization
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])

        validation_transforms = transforms.Compose([
            # data normalization
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])
    
        # Load the datasets with ImageFolder

        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)
        validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)


        # Define the dataloaders using the image datasets and image transformations 

        self.dataloaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True),
              'test': torch.utils.data.DataLoader(test_data, batch_size=16, num_workers=0),
               'validation': torch.utils.data.DataLoader(validation_data, batch_size=16, num_workers=0)
              }
        self.dataset_sizes = {'train': len(self.dataloaders['train']),
              'test': len(self.dataloaders['test']),
               'validation': len(self.dataloaders['validation'])}
        
        self.model = self._load_model(model_name)
        
        # Turn off gradients for our model
        for param in self.model.parameters():
            param.require_grad = False
            
        #feed-forward network
        self.model.classifier = model_lib.load_classifier(self.model, self.labels_output_categories)
        
        # Load naming of output categories
        self.model.class_to_idx = train_data.class_to_idx
        
    def _load_model(self, model_name):
        '''
            Initialize pretrained deep learning model (vgg13 or vgg16)
            
            Args:
                mode_name(str): 'vgg16' or 'vgg13'
                    
            Returns:
                model: pretrained deep learning model (vgg13 or vgg16)
        '''
        if (model_name == "vgg16"):
            return models.vgg16(pretrained=True)
        elif model_name == "vgg13":
            return models.vgg13(pretrained=True)
        else:
            raise "Error _load_model: Only vgg16 and vgg13 models are supported"
                

    def train_model(self, learning_rate, epochs, device, save_checkpoint_path):
        '''
        Train the deep learning model

        Args:
            learning_rate(float): learning rate for the optimizer e.g. 0.003
            epochs(int): number of training epochs that will be executed
            device(str): 'cuda' for using gpu, 'cpu' to run without gpu (not recommended, slow) 

        Returns:
            None
        '''
        print ("train_model start. device: \'{0}\' | learnin_rate: {1} | epochs: {2}"\
               .format(device, learning_rate, epochs))
            
        start = time.process_time()
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.classifier.parameters(), lr = learning_rate)

        self.model.train()
        self.model.to(device)
    

        running_loss = 0
        print_every = 100 # Define how often the intermediate training status will be printed \
        # - and the model tested against the validation set
    
        for epoch in range(epochs):
            steps = 0
            for images, labels in self.dataloaders["train"]:
                steps += 1
            
                self.model.train()
            
                images, labels = images.to(device), labels.to(device)
            
                optimizer.zero_grad()
            
                logps = self.model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
            
                if steps % print_every == 0:
                    
                
                    test_loss = 0
                    accuracy = 0
                    
                    # model intermediate evaluation during the training
                    self.model.eval()
                    
                    # deactivate autograd engine during validation. 
                    # - it will reduce memory usage 
                    # - it will speed up computations for model validation 
                    # - it will disable backpropagation we don't want to have during model validation anyway
                    with torch.no_grad(): 
                        for images, labels in self.dataloaders["validation"]:
                    
                            images, labels = images.to(device), labels.to(device)

                            logps = self.model.forward(images)
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
                    print("Validation accuracy {:.2}\n".format(accuracy/self.dataset_sizes["validation"]))
        
        
        
                    running_loss = 0
                    epoch_end = time.process_time()
                    print("Training runtime so far {0}".format(epoch_end - start))
                
        end = time.process_time()
        print("Training finished. Run time: {0}".format(end - start))    
                    
        model_lib.save_checkpoint(self.model, save_checkpoint_path, self.labels_output_categories)
        print ("Checkpoint saved into file \'{0}\'".format(save_checkpoint_path))
        
    def test_model(self, device):
        self.model.to(device)
    
        self.model.eval()
        
        criterion = nn.NLLLoss()

                
        test_loss = 0
        accuracy = 0
        
        steps = 0
        print_every = 100  # Define how often the intermediate training status will be printed \
    
        # deactivate autograd engine during validation. 
        # - it will reduce memory usage 
        # - it will speed up computations for model validation 
        # - it will disable backpropagation we don't want to have during model validation anyway
        with torch.no_grad(): # Turn off gradients for validation, saves memory and computations 
            
            for images, labels in self.dataloaders["test"]:
                steps += 1

                if steps % print_every == 0:
                    print("Steps {0}/{1}".format(steps, self.dataset_sizes["train"]))
                    
                images, labels = images.to(device), labels.to(device)

                logps = self.model.forward(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()
                    
                # Calculate accuracy
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                print("Validation loss {:.2}".format(test_loss/self.dataset_sizes["test"]))
                print("Validation accuracy {:.2%}".format(accuracy/self.dataset_sizes["test"])) 

# ============= main ==============
parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", required=True, help="path to an folder containing data folders \ (training, validation, testing) and file cat_to_name.json containing translation between output categories and their names")  
parser.add_argument("--output_cat", required=True, type=int, help="specify number of output categories for the model")  
parser.add_argument("--means", required=False, default=[0.485, 0.456, 0.406], help="image dataset means (provided with dataset)'")  
parser.add_argument("--stds", required=False, default=[0.229, 0.224, 0.225], help="image dataset stds (provided with dataset)'")  
parser.add_argument("--arch", required=False, default="vgg16", choices=["vgg16", "vgg13"], help="Select model for transfer learning. Default is vgg16")  


parser.add_argument("--save_path", required=False, default="assets/model/checkpoint_script.pth", help="location where to save a model checkpoint. Default is 'assets/model/checkpoint_script.pth'")  
parser.add_argument("--learning_rate", required=False, type=float, default=0.03, help="specify learning rate for the model. Default is 0.03")  
parser.add_argument("--epochs", required=False, type=int,  default=7, help="specify number of epochs for training. Default is 7")  
parser.add_argument("--device", required=False, default="cpu",choices=["cpu", "gpu"], help="use GPU during the computation. Default is CPU")  


args = parser.parse_args()

data_directory = args.data_dir
output_categories = args.output_cat
means = args.means
stds = args.stds

save_path = args.save_path
arch = args.arch
learning_rate = args.learning_rate
epoch = args.epochs
device = args.device

train_cl = Train(data_directory, output_categories, means, stds, arch)    
train_cl.train_model(learning_rate, epoch, device, save_path)
train_cl.test_model(device)
        