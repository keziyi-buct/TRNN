1、The code is divided into two libraries, for TRNN1-1, which is a single-task, single-output model, the relevant code is stored in the folder OpenSA1, and for the multi-task models TRNN M-1, M-M, the relevant code is stored in OpenSA. Running the model ensures that the corresponding folder name is OpenSA to ensure that the path can be read correctly.

2、The LUCAS data is in a several hundred MB file, we processed the dataset to weed out missing data and other data not relevant to the experiment, this file is not in the folder, you can download it at the link below. 
https://pan.baidu.com/s/1ONRBRJP9pRSfC2z_-sdoag Extract Password: eaxf
After downloading, store the data in OpenSA/Data/Rgs to make sure the code works.

3、We have saved the trained models for the seven soil properties corresponding to TRNN1-1, but they take up too much memory, and we have uploaded them to this link. Using the trained models can speed up the training process. If you need to use a specific model, just put the corresponding model into the model folder. If you get a path error, check the OpenSA/Regression/CNN.py file for the code that reads the path to the model.

4、When all these preparations are done, you can go to the OpenSA folder and run the example.py file to start the training, and the relevant model weights, the training process, and the performance of the parameters are all recorded. For Linux users, you can directly type in the terminal

Python example.py 

to run the code. It is important to note that you need to make sure that the relative path to the code is in the current OpenSA folder.

5、Regarding the parameter configuration of the training process, we train with a GPU graphics card with 24GB of memory, if your graphics card does not have enough memory to support the experiment, an error will occur.There are no CPU requirements, but high-performance hardware will speed up the training process.

6、The code for the model interpretability methods is still being sorted out, and will be sorted out and uploaded to the platform some time later. Using these methods requires a high graphics card, so it is recommended to use a 48G memory graphics card or two 24G graphics cards. The code provided here is for a single graphics card, if you need to use multiple GPUs, you need to add some code.
