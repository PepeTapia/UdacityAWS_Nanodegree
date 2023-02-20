# Udacity - AI Programming with Python Course Project


# Why I took this course
In the Summer of 2022 I applied for the [AWS AI/ML Scholarship Program](http://awsaimlscholarship.com/) with the goal of improving my fundamental knowledge of AI/ML in practice. This was a great opportunity to prepare myself for my Fall semester of 2022, in which I took two academic courses - Machine Learning(CPSC 483) and Artificial Intelligence(CPSC 491). I'm grateful for the opportunities that happened shortly after this course and it will forever be a reminder that chasing opportunities to improve yourself can open many doors.


## The application process
The application process included chapter modules that covered the common ML pipeline of defining a problem, building a dataset, model training, model evaluation, and model inference. 

![AWS Chapter modules that cover a variety of ML topics](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/assets/AWS_Scholarship_ChapterModules.png)

## Eligibility for the scholarship
After studying these modules, you compete in the AWS DeepRacer Student League. Eligibility for the scholarship was dependent on competing in the DeepRacer Student League and finishing a time trial lap in less than 3 minutes on any single leaderboard.

### DeepRacer Student League
The DeepRacer Student League was a race car model that uses Reinforcement Learning to drive a virtual car around a virtual track. The applicant would be able to choose the track, choose the algorithm type, and customize the code and reward function! After that you choose your duration of training for your model.

![Overview of the DeepRacer Model setup](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/assets/DeepRacer_Model.png)

### Custom model adjustments within the DeepRacer Student League Models!

The user is provided with default models and algorithms, but can customize certain variables to help improve their model.

![Example of the parameter adjustions an applicant can make](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/assets/DeepRacer_RewardFunction.png)


As you can see with this given model provided by AWS with no parameters changed, I would not have made the cut! But, there is room to improve with the 10 hours of free training that AWS provides you with. 

![AWS model with no parameters changed resulted in a time of 05:30](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/assets/my_Model.png)



# Project code for Udacity's AI Programming with Python Nanodegree program. 

Includes two image classifiers of the same purpose but different file formats:
1. ipynb file created on Jupyter Notebook
2. Command Line Interface using three files ['predict.py,'train.py','utility.py']


## This model was built using:

102 Category Flower Dataset
by Maria-Elena Nilsback and Andrew Zisserman
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


## IPYNB, Python Notebook version

This is a beginner friendly course that helps students understand the structure of building the model from start to finish. Only few code bits were given to students, but most of it was presented throughout the course with the goal that a student would confidently be able to fill out the majority of the code. I recommend checking out the [IPYNB code here](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/Image%20Classifier%20Project.ipynb)


### Details of learning PyTorch and understanding the fundamentals of models

During the course you are taught the fundamentals and importance of Linear Algebra and Calculus within ML Models, specifically Neural Networks. To help create this ML pipeline, the student is also taught NumPy, Pandas, and Matplotlib. 

Referencing the aforemention [IPYNB code](https://github.com/PepeTapia/UdacityAWS_Nanodegree/blob/main/Image%20Classifier%20Project.ipynb):

We see that the output of Cell 5 is the full VGG-11 architecture. The scholarship program taught students about each individual component and why each layer is placed at their location. The code below includes the first few layers.

Conv2d 
: is a convulation layer that has an input channel, output channel, and other parameters to create a total 2d layer.

ReLU
: short for Rectified Linear Unit, this is the activation function. It outputs the input directly if it is positive, otherwise it is zero

MaxPool2d
: calculates the maximum of a specified patch of an array, in this case it's defined by the stride
```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
```


For this course we used Transfer Learning, which takes a pretrained model and adds custom layers. Below we add sequential layers including a Linear Transformation, ReLU, and Dropout. For our last layer we included a LogSoftmax layer along the first dimension.

Our loss criterion is the negative log likelihood loss, which helps us understand how our model is performing.
The optimizer chosen was the Adam optimzer, a form of stochastic gradient descent.

```

classifier = nn.Sequential(nn.Linear(25088,2056),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(2056,102),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)

```

The results of the model running. 

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

# Command Line Implementation

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



