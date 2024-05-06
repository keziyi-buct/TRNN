1、The code is divided into two libraries, for TRNN1-1, which is a single-task, single-output model, the relevant code is stored in the folder OpenSA1, and for the multi-task models TRNN M-1, M-M, the relevant code is stored in OpenSA. Running the model ensures that the corresponding folder name is OpenSA to ensure that the path can be read correctly.

2、The LUCAS data is in a several hundred MB file, we processed the dataset to weed out missing data and other data not relevant to the experiment, this file is not in the folder, you can download it at the link below. 
https://pan.baidu.com/s/1ONRBRJP9pRSfC2z_-sdoag Extract Password: eaxf
After downloading, store the data in OpenSA/Data/Rgs to make sure the code works.

3、We have saved the trained models for the seven soil properties corresponding to TRNN1-1, but they take up too much memory, and we have uploaded them to this link. Using the trained models can speed up the training process. If you need to use a specific model, just put the corresponding model into the model folder. If you get a path error, check the OpenSA/Regression/CNN.py file for the code that reads the path to the model.

