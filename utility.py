import argparse

def training_input_args():
    """
    Retrieves and parses the command line arguments outlined in the Image Classifier instructions.
    Arguements include:
    --save_dir: Path to save checkpoints to
    --arch: Model Architechture
    --learning_rate: Defines learning rate
    --hidden_units: Defines hidden units
    --epochs: Defines number of epochs to run
    --gpu: Using this argument enables GPU for training
    
    """
    #Create Argument Parser class
    parser = argparse.ArgumentParser()
    
    #Add argument of Data Directory to use for main files
    parser.add_argument('data_dir', 
                       help = 'Path to data directory of flower images')
   
    #Add arguments save_dir for save_directory, default saved_models
    parser.add_argument('--save_dir', type = str, default = 'saved_models/',
                       help='path to save models')
    
    #Add argument to set model architechture, default vgg11
    parser.add_argument('--arch', default='vgg11',
                       help='Select model architechture')
    
    #Add argument to set learning rate, default of 0.01
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Define learning rate')
    
    #Add argument to set hidden_units, default 512
    #For this project I will not be using hidden_units because I am not sure where it was in the learning material. I will study this and come back to it after the program
    parser.add_argument('--hidden_units', default='512',
                      help='Define hidden units')
    
    #Add argument to set number of epochs to run, default 5
    parser.add_argument('--epochs', type=int, default=5,
                      help='Define number of epochs')
    
    #Add argument to turn on GPU for training. Including the argument enables by default.
    parser.add_argument('--gpu', type = bool, default=False, help='Use gpu argument to enable gpu for training')
    
    in_args = parser.parse_args()
    
    #Returns a collection of arguments
    return in_args

def testing_input_args():
    #Create Argument Parser class
    parser = argparse.ArgumentParser()
    
    #Add argument for a path to a single image
    parser.add_argument('path_to_image', type=str, help='Path to single flower image')
   
    #Add argument for a model checkpoint
    parser.add_argument('checkpoint', type=str, default='saved_models/checkpoint_image.pth',
                        help='Path to model checkpoint file')
    #Add argument to choose the top K most likely classes to return
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top K most likely classes to return')
    #Add argument for mapping of categories to real names:
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                       help='Category name mapping json file')
    #Add argument for GPU access
    parser.add_argument('--gpu', type=bool, default=False, help='Use gpu argument to enable gpu for testing')                  
    in_args = parser.parse_args()
    
    #Returns a collection of arguments
    return in_args