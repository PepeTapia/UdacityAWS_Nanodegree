# Udacity - AI Programming with Python Course Project


# Why I took this course
In the Summer of 2022 I applied for the [AWS AI/ML Scholarship Program](http://awsaimlscholarship.com/) with the goal of improving my fundamental knowledge of AI/ML in practice. This was a great opportunity to prepare myself for my Fall semester of 2022, in which I took two academic courses - Machine Learning(CPSC 483) and Artificial Intelligence(CPSC 491). I'm grateful for the opportunities that happened shortly after this course and it will forever be a reminder that chasing opportunities to improve yourself can open many doors.


## The application process
The application process included chapter modules that covered the common ML pipeline of defining a problem, building a dataset, model training, model evaluation, and model inference. 


## Eligibility for the scholarship
After studying these modules, you compete in the AWS DeepRacer Student League. Eligibility for the scholarship was dependent on competing in the DeepRacer Student League and finishing a time trial lap in less than 3 minutes on any single leaderboard.</p>

As you can see with this given model provided by AWS with no parameters changed, I would not have made the cut! But, there is room to improve with the 10 hours of free training that AWS provides you with. 




##

PROGRAMMER: Jose Tapia
DATE CREATED: 08/17/2022
REVISED DATE: 08/26/2022


Project code for Udacity's AI Programming with Python Nanodegree program. 


NOTE:

-Environment yml file included
```
Includes two image classifiers of the same purpose but different file formats:
1. ipynb file created on Jupyter Notebook
2. Command Line Interface using three files ['predict.py,'train.py','utility.py'] that classifies this dataset:
```
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
Training and Validation loss from this run
<img width="378" alt="train_valid_lossPlot" src="https://user-images.githubusercontent.com/22277499/187011858-33637d60-7125-45a9-9422-c46c29541956.png">



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
