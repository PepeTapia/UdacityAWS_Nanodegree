import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import random
import os
from utility import training_input_args, testing_input_args
import json

# PROGRAMMER: Jose Tapia
# DATE CREATED: 08/26/2022
# REVISED DATE: 08/26/2022
# PURPOSE: Tests a saved model against a novel image


def main():
    
    in_args = testing_input_args()
    
    #print(in_args)
    #return
    #Load model from model checkpoint path
    model_checkpoint = in_args.checkpoint
    model = load_checkpoint(model_checkpoint)
    checkpoint = torch.load(model_checkpoint)
    #Load_state_dict LOADS the model and attaches it

    
    #Ensure device is specified and active
    if (in_args.gpu == True) and torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #Send model to the device for consistency in GPU/CPU usage
    model.to(device)
    
    #Get label map and load it
    label_map = in_args.category_names
    with open(label_map,'r') as f:
        cat_to_name = json.load(f)
    
    #Prepare final variables image_path and topk, both from in_args
    #model has been set above
    image_path = in_args.path_to_image
    topk = in_args.top_k

    #Begin testing sequence
    img = process_image(image_path)
    img = torch.tensor(img, device=device).float()
    
    #Adds a batch of size 1 to image
    model_input = img.unsqueeze(0)
    #Sends input to device to be read as tensor cuda
    true_input = model_input.to(device)
    
    #Predict top k probabilties
    ps = torch.exp(model(true_input))
    top_p, top_class = ps.topk(topk)
    top_p = top_p.detach().tolist()[0]
    top_class = top_class.detach().tolist()[0]
    
    #Creates lasts for the topk values, ensuring that a value of K>=1 is output
    idx_to_class = {val: key for key, val in    
                            model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_class]
    top_classes = [cat_to_name[idx_to_class[lab]] for lab in top_class]
    print("Probabilties respective to labels: ", top_p)
    print("Top classes, flower names, respective to the probabilities shown above:", top_classes)
    
    #Return the actual flower name
    flower_num = image_path.split('/')[-2]
    
    print("The flower is actually: ", cat_to_name[flower_num])
    return
    
def load_checkpoint(checkpoint_path):
    """
    This function loads a checkpoint and recreates the model as it was initially created, loading the classifier, load_state_dict, and class_to_idx into the model
    """
    #Loads checkpoint path
    checkpoint = torch.load(checkpoint_path)
    #Loads model to prepare for transfering of model attributes
    model = models.vgg11(pretrained=True)
    
    #Freeze parameters that do not require grads
    for param in model.parameters():
        param.requires_grad = False
    
    #Model classifier as was made during training
    model.classifier = nn.Sequential(nn.Linear(25088,2056),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(2056,102),
                           nn.LogSoftmax(dim=1))
    #Load state dict and class mapping
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model.class_to_idx = checkpoint['class_mapping']

    return model

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Size of 256,256 for the image, preparing for resizing
    size = 256,256
    with Image.open(image) as im:
        #Resize image
        im.thumbnail(size)

        # Get dimensions of image to formulate into a cropped image 
        width, height = im.size   
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2

        im = im.crop((left, top, right, bottom))

        #Normalize the image
        im = np.array(im)/255
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        im = (im - mean)/std

        #Transpore image
        im = im.transpose((2,0,1))
        
        #Return fully processed image to be accepted by the model requirements
        return im

        
if __name__ == "__main__":
    main()   
   
   