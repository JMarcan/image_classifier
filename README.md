# Image Classifier

## Project description
In the project I've used Deep Learning to train image classifier to categorize images. 

I've trained and tested it on a dataset of flowers as part of my graduation from [Udacity](https://www.udacity.com/) `Data Science Nanodegree`. 
One of the model applications could be that you point your smartphone on a flower and phone app will tell you what flower it is. 
However, the model can be retrained on any dataset of your choice. You can learn it to recognize cars, point your picture on the car, and let the application to tell you what the make and model it is.

In the project:
- First, I've loaded and reprocessed the image dataset
- Second, I've trained the image classifier on my dataset.
- Third, I've used the trained classifier to predict image content.

In the step loading and reprocessing the image dataset:
- I've used `torchvision` to load the data. 
- Next, I've split the data between training, validation, and testing. For training I've augmented data by applying transformations such as random scaling, cropping and flipping to ensure that network is generalized and provides better performance.
- Lastly, I've normalized means and standard deviations of the images to fit what the pre-trained network expects.

In the second step, to train the image classifier:
- I've loaded a pre-trained network (VGG16) from `torchvision`
- Next, I've defined a new, untrained `feed-forward network` as the classifier, using `ReLU activations` and `dropout`.
- Next, I've trained the classifier layers using `backpropagation` using the pre-trained network to get the features. I've tracked the loss and accuracy on the validation set to determine the best hyperparameters. 
- The training required usage of `GPU computing` for which is torch perfect fit.


In the last step, to use the trained classifier to predict image content:
- I've preprocessed input images to fit the trained model, so they are resized and normalized first.
- Lastly, I've written a function to predict the top 5 most probable classes with their probabilities. Results are visualized to the user in the form of a barplot graph.


![classification_sample](classification_sample.png)

## Usage

1. Download [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories

2. Run `train.py' to train the model
- Basic usage: `python train.py data_directory`
- Options:
- Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
- Choose architecture: `python train.py data_dir --arch "vgg13"`
- Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Use GPU for training: `python train.py data_dir --gpu`

3. Run `predict.py` to classify your image
- Basic usage: `python predict.py /path/to/image checkpoint`
- Options: 
- Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
- Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
- Use GPU for inference: `python predict.py input checkpoint --gpu`

## Libraries used
Python 3
- pandas
- numpy
- torch
- torchvision
- matplotlib
- seaborn
- PIL

## Files in the repository
- `Image Classifier Project.ipynb.py`: Jupyter Notebook File containing the whole code
- `train.py`: contains functionality to retrain model for a dataset of your choice
- `predict.py`: contains functionality to classify an image
- `cat_to_name.json`: contains the mapping between image categories and their real name