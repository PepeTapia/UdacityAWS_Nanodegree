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
# PURPOSE: Classifies flower images using a pretrained CNN model and compares them to the 
#          classifications of the true image label for that flower image.
#          Returns and prints out training loss, validation loss, and validation accuracy
def main():
    
    #Creates in_args from utility.py and returns a collection of the command line arguments input
    in_args = training_input_args()
    
    print(in_args)
    
    
    #Set up folder accessibility variables for train,valid, and test dir from root data_dir
    data_dir = in_args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/train'
    
    #Tranform train,valid, and testing sets into the ImageNet dataset compositions originally used when created
    
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])

    #Open each image folder and apply the transformations
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    #Initialize dataloaders from each transforms dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    #Label mapping category label to category name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    #------------Create a training model network------------#
    
    #Retrives model architechture name from arg input
    model_type = in_args.arch
    
    #Loads pretrained model architechture
    if model_type == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        print("Add functionality to accept other models")


    
    
    #Freeze parameters that do not require gradient
    for param in model.parameters():
        param.requires_grad = False
        
    #Ensure device is specified and active
    if in_args.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #Send model to the device for consistency in GPU/CPU usage
    model.to(device)
    
    #Create a Forward Feeding Network
    
    #Create classifier. In this case, use the classifier made in the Image Classifier Part 1
    classifier = nn.Sequential(nn.Linear(25088,2056),
                               nn.ReLU(),
                               nn.Dropout(0.1),
                               nn.Linear(2056,102),
                               nn.LogSoftmax(dim=1))
    
    #Attach the created classifier to the model, swapping the classifiers
    model.classifier = classifier
    #NLLLoss is used with LogSoftMax
    criterion = nn.NLLLoss()
    
    #Using adam optimizer for the momentum feature
    optimizer = optim.Adam(model.classifier.parameters(),lr=in_args.learning_rate)
    #Send model to the device for consistency in GPU/CPU usage
    model.to(device)
     
    #Training the model
    #After multiple tests and attempts, I settled with the above parameters in the classifier
    #The best results were found at 5 epochs, so I will only run 5.
    epochs = in_args.epochs
    steps = 0
    print_every = 1
    train_losses, valid_losses = [],[]
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps+= 1
            #Ensures the tensors go to cuda
            images,labels = images.to(device), labels.to(device)

            #Record the log probabilties
            log_ps = model(images)

            loss = criterion(log_ps,labels)
            #Set gradients to zero each loop during training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0

            #Model evaluation requires torch.no_grad() and switching to model.eval()
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images,labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    loss = criterion(log_ps,labels)
                    valid_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            model.train()

            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
    #Validation on the test set
    epochs = 1
    steps = 0
    print_every = 1
    model.to(device)
    for epoch in range(epochs):
        if steps % print_every == 0:
            #Switch model to evaluation mode
            model.eval()
            test_loss = 0
            accuracy = 0
            #Turn of gradients
            with torch.no_grad():
                for inputs, labels, in testloader:
                    steps +=1
                    inputs,labels = inputs.to(device), labels.to(device)


                    logps = model(inputs)
                    loss= criterion(logps,labels)

                    test_loss += loss.item()

                    #Grab probabilities and label of the top class
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Test loss: {test_loss/len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(testloader):.3f}")
    #Save model checkpoint
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_mapping': train_data.class_to_idx},
                f'{in_args.save_dir}/checkpoint_image.pth')
        
if __name__ == "__main__":
    main()
     
   