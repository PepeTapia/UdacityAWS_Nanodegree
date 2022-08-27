# AI Programming with Python Project


PROGRAMMER: Jose Tapia
DATE CREATED: 08/17/2022
REVISED DATE: 08/26/2022

Project code for Udacity's AI Programming with Python Nanodegree program. 


NOTE:
-Environment yml file included
-saved model checkpoints are not available due to the size restriction of github
Includes two image classifiers of the same purpose but different file formats:
1. ipynb file created on Jupyter Notebook
2. Command Line Interface using three files ['predict.py,'train.py','utility.py'] that classifies this dataset:

102 Category Flower Dataset
by Maria-Elena Nilsback and Andrew Zisserman
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


```
Epoch: 1/5..  Training Loss: 2.228..  Validation Loss: 0.612..  Validation Accuracy: 0.827
Epoch: 2/5..  Training Loss: 0.848..  Validation Loss: 0.429..  Validation Accuracy: 0.873
Epoch: 3/5..  Training Loss: 0.702..  Validation Loss: 0.442..  Validation Accuracy: 0.885
Epoch: 4/5..  Training Loss: 0.590..  Validation Loss: 0.327..  Validation Accuracy: 0.893
Epoch: 5/5..  Training Loss: 0.547..  Validation Loss: 0.331..  Validation Accuracy: 0.917
```

Image here



Network Tested:
```
Epoch 1/1.. Test loss: 0.020.. Test accuracy: 0.034
Epoch 1/1.. Test loss: 0.049.. Test accuracy: 0.061
Epoch 1/1.. Test loss: 0.069.. Test accuracy: 0.093
Epoch 1/1.. Test loss: 0.097.. Test accuracy: 0.123
...
...
...
Epoch 1/1.. Test loss: 0.405.. Test accuracy: 0.806
Epoch 1/1.. Test loss: 0.407.. Test accuracy: 0.845
Epoch 1/1.. Test loss: 0.454.. Test accuracy: 0.873
```

Command Line Usage:

python train.py A:/CodingProjects/Udacity_AWS_Nanodegree/flowers --save_dir A:/CodingProjects/Udacity_AWS_Nanodegree/saved_models  --gpu True 
python predict.py A:/CodingProjects/Udacity_AWS_Nanodegree/flowers/test/1/image_06743.jpg  A:/CodingProjects/Udacity_AWS_Nanodegree/saved_models/checkpoint_image.pth --gpu True 



While ideal to stop at Epoch 4, I included it for the purpose of training and the understanding that epoch 4 would have been the better stopping place.
My next iteration would test 4 epochs instead of 5 
```
Epoch: 1/5..  Training Loss: 2.002..  Validation Loss: 0.607..  Validation Accuracy: 0.831
Epoch: 2/5..  Training Loss: 0.860..  Validation Loss: 0.534..  Validation Accuracy: 0.855
Epoch: 3/5..  Training Loss: 0.713..  Validation Loss: 0.426..  Validation Accuracy: 0.882
Epoch: 4/5..  Training Loss: 0.629..  Validation Loss: 0.326..  Validation Accuracy: 0.910
Epoch: 5/5..  Training Loss: 0.575..  Validation Loss: 0.382..  Validation Accuracy: 0.901
```


```
Epoch 1/1.. Test loss: 0.001.. Test accuracy: 0.005
Epoch 1/1.. Test loss: 0.001.. Test accuracy: 0.009
Epoch 1/1.. Test loss: 0.001.. Test accuracy: 0.014
Epoch 1/1.. Test loss: 0.004.. Test accuracy: 0.018
Epoch 1/1.. Test loss: 0.005.. Test accuracy: 0.023
Epoch 1/1.. Test loss: 0.005.. Test accuracy: 0.028
Epoch 1/1.. Test loss: 0.010.. Test accuracy: 0.031
Epoch 1/1.. Test loss: 0.013.. Test accuracy: 0.035
Epoch 1/1.. Test loss: 0.013.. Test accuracy: 0.040
Epoch 1/1.. Test loss: 0.013.. Test accuracy: 0.045
...
...
...
Epoch 1/1.. Test loss: 0.113.. Test accuracy: 0.946
Epoch 1/1.. Test loss: 0.113.. Test accuracy: 0.951
Epoch 1/1.. Test loss: 0.114.. Test accuracy: 0.956
Epoch 1/1.. Test loss: 0.114.. Test accuracy: 0.960
Epoch 1/1.. Test loss: 0.114.. Test accuracy: 0.965

```
Result: 
python predict.py A:/CodingProjects/Udacity_AWS_Nanodegree/flowers/test/1/image_06743.jpg  A:/CodingProjects/Udacity_AWS_Nanodegree/saved_models/checkpoint_image.pth --gpu True 

```
Probabilties respective to labels: [0.9642105102539062, 0.03567475080490112, 0.00010268234473187476]
Top classes, flower names, respective to the probabilities shown above: ['pink primrose', 'tree mallow', 'hippeastrum']
The flower is actually:  pink primrose
````