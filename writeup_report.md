#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./before_bias_removal.png "Histogram-before removing bias around ZERO steering angle"
[image2]: ./after_bias_removal.png "Histogram-after removing bias around ZERO steering angle"
[image3]: ./nvidia "Architecture"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

Project includes the following files:
* model.py containing the script to create and train the model
* model.html containg snapshot of training.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network weights 
* model.json  containing a trained convolution meural network model
* writeup_report.md  summarizing the results
* video.mp4 Video recording of autonomous driving using drive.py and above mentioned model

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```
docker run -it --rm -p 4567:4567 -v 'pwd':/src udacity/carnd-term1-starter-kit python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

There are three primary functions written 
- trans_image - function used for image translation
- draw_histogram - function to view data distribution
- remove _steering - function used to remove bias caused due to too many small steering angles around ZERO
- preprocess - function primarily to cropping, reshaping, smoothing and color channel adjustment.

These functions are called in pre-processing step before training the model.



###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
    Tried modeling using LeNet and Nvidia architecture. Nvidia gave better results than LeNet. Nvidia model was slightly modified to avoid overfitting. Below is model summary at each layer.

---------------------------------------------------------------------------------------------------
Layer (type)                     Output Shape          Param #     Connected to                     
--------------------------------------------------------------------------------------------------------
lambda_7 (Lambda)                (None, 66, 200, 3)    0           lambda_input_7[0][0]             
____________________________________________________________________________________________________
convolution2d_31 (Convolution2D) (None, 31, 98, 24)    1824        lambda_7[0][0]                   
____________________________________________________________________________________________________
convolution2d_32 (Convolution2D) (None, 14, 47, 36)    21636       convolution2d_31[0][0]           
____________________________________________________________________________________________________
convolution2d_33 (Convolution2D) (None, 5, 22, 48)     43248       convolution2d_32[0][0]           
____________________________________________________________________________________________________
convolution2d_34 (Convolution2D) (None, 3, 20, 64)     27712       convolution2d_33[0][0]           
____________________________________________________________________________________________________
convolution2d_35 (Convolution2D) (None, 1, 18, 64)     36928       convolution2d_34[0][0]           
____________________________________________________________________________________________________
flatten_7 (Flatten)              (None, 1152)          0           convolution2d_35[0][0]           
____________________________________________________________________________________________________
dense_31 (Dense)                 (None, 1164)          1342092     flatten_7[0][0]                  
____________________________________________________________________________________________________
dropout_13 (Dropout)             (None, 1164)          0           dense_31[0][0]                   
____________________________________________________________________________________________________
dense_32 (Dense)                 (None, 100)           116500      dropout_13[0][0]                 
____________________________________________________________________________________________________
dropout_14 (Dropout)             (None, 100)           0           dense_32[0][0]                   
____________________________________________________________________________________________________
dense_33 (Dense)                 (None, 50)            5050        dropout_14[0][0]                 
____________________________________________________________________________________________________
dropout_15 (Dropout)             (None, 50)            0           dense_33[0][0]                   
____________________________________________________________________________________________________
dense_34 (Dense)                 (None, 10)            510         dropout_15[0][0]                 
____________________________________________________________________________________________________
dense_35 (Dense)                 (None, 1)             11          dense_34[0][0]                   
----
Total params: 1,595,511
Trainable params: 1,595,511
----

Model was modified as below
- Lambda layer added in the front for Normalizing the image
- Dropout added after each Fully connected layer to avoid overfitting.




####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py). 

Model has been trained with different datasets to avoid overfitting and below are different training dataset used

1. Central driving: data collected with one complete lap driving in center of track.
2. Central reverse driving : data collected with one complete lap driving in center of track in reverse direction.
3. Recovery driving: data collected with one complete lap driving in recovery mode. This is very stressful. Recovery mode - *Record is done only whil driving car away from shoulder lines and recording is not done while driving towards shoulder lines in track.*
4. Recovery reverse driving : data collected with one complete lap driving in recovery mode in reverse direction. Recovery mode - *This is very stressful. Record is done only whil driving car away from shoulder lines and recording is not done while driving towards shoulder lines in track.*
5. Problemetic Areas - During model testing, it was identified that certain locations of tracks are causing problem and pushing car away from the track. All these areas were identified and car was trained in recovery mode in specific locations of track. This improved autonomous driving mode a lot.
6. Udacity sample data: Udacity sample data was also used for training.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was set to 0.001 in model.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the above section mentioning training data approaches. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to LeNet. This model was pushing car out of track after few seconds. Then, i changed to Nvidia's model and still car was not staying in track. Along with car not staying in track, my validation loss was raising after some time showing signs of overfitting.

I did following steps to fix problem:

1. instead of collection data for center lane driving, recovery lane driving in single directly, i started collecting them in different directories. I have created four directories(central, central_reverse, recovery, recovery_reverse). In addition, i have utilized udacity's data collection as well.

2. After collection data, create histogram to view the data distribution. This show huge bias in data having steering angle around ZERO. this led to new function "steering_remove" where buckets having dataset above the average dataset count in all buckets will be considered for undersampling.  Following approached for undersampling
    - Use np.histogram and identify bins & hist(frequency). 
    - Calculatae average dataset count accross the bins
    - Assign bin value to each steering angle in the dataset and group them into *above average bins* and *below average bins* 
    - In all *above average bins* steering angles will be randomly identified and marked for removed to bring downt the record count
    in that specific bin.
    
    ![alt text][image1] ![alt text][image2]
    

This step drastically improved the results.

3. Also, data augmentation was done using image flipping, image translation.

4. Image was cropped and reshaped for better performance as well as for meeting Nvidia model requirements.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in 80/20 ratio. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting by increasing training dataset as well as adding dropout to model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I  have identified locations in track where car often fell away from track. Initiated another set of data collection in recovery mode driving. This help to keep car on track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network built based on Nvidia architecture.

![Nvidia Architecture] [image3]

####3. Creation of the Training Set & Training Process

All data collected are accumulated and following steps are done to train:

1. All images run thru pre-processing step as mentioned above. To re-iterate, below are preprocessing steps
    - Images are cropped, reshaped, smoothened and color channel adjusted.
    - data augumentation using flipping and image translation.
2. For better performance python generator was used for pushing images/steering angles into model during the training process.
3. Performance of training and validation is displayed in end using plot as shown in driving.ipynb 
