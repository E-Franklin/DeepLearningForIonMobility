# DeepLearningForIonMobility

## Installation
Begin by downloading the repo. TODO: better instructions here  

Create a virtual environment using the following command and replacing 
"environment_name" with whatever you would like your environment to be called. 
```
# Linux and macOS
python -m venv environment_name  
# Windows
python -m venv venv environment_name
``` 

To activate the environment so that you can install and use packages run:
```
# Linux and macOS
source virtual_environment_name/bin/activate
# Windows
.\virtual_environment_name\Scripts\activate
```

Now you can install the required packages from the requirements.txt file 
provided.
```
pip3 install -r requirements.txt
```

You can exit the environment using the 'deactivate' command.
```
# Linux, macOS, and Windows
deactivate
```

## Training models
Some default parameters are specified in configs-default.yaml. Other parameters
will be specified by a config file that you create. You may also overwrite the 
default values in your own config file.
```
python run_training.py <path to training data> <path to testing data> <path to config file>
```


## Making predictions using pre-trained models
###Formatting data
Peptides are represented as strings of single letter amino acid codes. 
Supported modified amino acids include .... represented as .... respectively.

###Download the model 


###Make predictions 
You can use the following command to make predictions.  
```python predict.py <path to data file>```
