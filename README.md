# Action Recognition for Intent Classification </br>

**UC Berkeley MIDS W251**

**Indrani Bose, Brian Ament and Mayukh Dutta**

##  <a id="Contents">Contents
  
[1.0 Introduction](#Introduction)

[2.0 System Overview](#System_Overview)

[3.0 The Model](#Model)

[4.0 Generating Training Data](#Imgen)

[5.0 Training the Model](#Train)

[6.0 Inferencing at the Edge](#Edge)


## <a id="Introduction">1.0 Introduction

### 1.1 A _Very_ Brief Overview of Action Recognition
Human action recognition is a standard Computer Vision problem and has been well studied. The fundamental goal is to analyze a video to identify the actions taking place in the video. Essentially a video has a spatial aspect to it ie. the individual frames and a temporal aspect ie. the ordering of the frames. Some actions (eg. standing, running, etc.) can probably be identified by using just a single frame but for more complex actions(eg. walking vs running, bending vs falling) might require more than 1 frame’s information to identify it correctly. Local temporal information plays an important role in differentiating between such actions. Moreover, for some use cases, local temporal information isn’t sufficient and you might need long duration temporal information to correctly identify the action or classify the video.Deep learning approaches have empirically demonstrated remarkable success in learning image representations for tasks like object recognition, image captioning, and semantic segmentation. Convolutional neural networks have enabled us to efficiently capture the hypothesis of spatial locality of data structure in images through parameter sharing convolutions, and local invariance-building max-pooling neurons. In this project, we would like to explore the impact of deep learning techniques on video tasks, specifically action recognition.

*[Return to contents](#Contents)*

### 1.2 Intent of this Project
We want to explore the possibility of applying deep learning to the task of human action recognition and predict the next actions. We wanted to address the following questions through our project
- In space of autonomous vehichles, this can be used to classifiy the intent of pdestriants cyclists .
- In reatil space , the learning can be transferred to classify the intent of the customer (e.g If the person is a prospective buyer or a shop lifter)

## <a id="System_Overview">2.0 System Overview
  
### 2.1 Use Case
In the current project ,we plan to detect 'unmindful prdestrains/'cyclists'.  On prediction of the next motion we have plan to
- Sound an alarm (can be done on the xavier itself, but we would need a buzzer etc.)
- Save the video clip where the action was detected
- Stream the live feed to an App on the phone or a web application and highlight when a potential action is detected

### 2.2 Assumptions
1. The training videos are of high quality
2. The training data is labelled correctly

### 2.3 System Design
Our solutions consists of the following
1. The Cloud - We are using a R(2D+1) Model trained in AWS . The specifications for the same are 
  - NVIDIA Deep Learning AMI v20.11.0-46a68101-e56b-41cd-8e32-631ac6e5d02b
  -  g4dn.2xlarge

2. The Data - It is trained on Kinetics400 dataset, a benchmark dataset for human-action recognition. The accuracy is reported on the traditional validation split.

3. The Edge Device - We are using Jetson Xavier NX for our inference. Trained models are saved over to Jetson device and used for testing.The USB cam on the xavier will stream in the video feeds and the pre-trained model will predict if there is a 'pedestrian approaching' or a 'cyclist approaching' in the view of the camera. Note: this is different than just detecting if there is a pedestrian in the frame, while driving on the streets, there will be predestrians in the view, our approach here is to detect when the pedestrain is dangerously close to the vehicle or moving in a way that could potenially mean them intercepting the path of the moving vehicle. It's easy for humans to detect such situations since we have plethora of experiences detecting when a situation may develop with the slightest of the hints. 

![System diagram](https://github.com/indr19/Action_Recognition/blob/master/images/W251%20System%20Design.png)

## <a id="Model">3.0 The Model
We use a “(2+1)D” convolutional block, which explicitly factorizes 3D convolution into two separate and successive operations, a 2D spatial convolution and a 1D temporal convolution.
R(2+1)D are ResNets with (2+1)D convolutions. For interpretability, residual connections are omitted.
The first advantage is an additional nonlinear rectification between these two operations. This effectively doubles the number of nonlinearities compared to a network using full 3D convolutions for the same number of parameters, thus rendering the model capable of representing more complex functions.
The second potential benefit is that the decomposition facilitates the optimization, yielding in practice both a lower training loss and a lower testing loss.

### 3.1 Model Architecture 
- Convolutional residual blocks for video
In this section we discuss several spatiotemporal convolutional variants within the framework of residual learning.
Let x denote the input clip of size 3×L× H ×W, where L is the number of frames in the clip, H and W are the frame height and width, and 3 refers to the RGB channels. 
Let z<sub>i</sub> be the tensor computed by the i-th convolutional block in the residual network. In this work we consider only“vanilla” residual blocks (i.e., without bottlenecks), with each block consisting of two convolutional layers with a ReLU activation function after each layer. Then the output of the i-th residual block is given by

![Cov_Res equation](https://github.com/indr19/Action_Recognition/blob/master/images/Conv%20Res.JPG)

where F(; θ<sub>i</sub>) implements the composition of two convolutions parameterized by weights θi and the application of the ReLU functions.

![Model_architecture](https://github.com/indr19/Action_Recognition/blob/master/images/Capture.JPG)

- **Loss Fuction - CrossEntropyLoss**</br>

- **Kinetics400 dataset pretraining parameters**</br>
input size: [3, 16, 112, 112]</br>
input space: RGB</br>
input range: [0, 1]</br>
mean: [0.43216, 0.394666, 0.37645]</br>
std: [0.22803, 0.22145, 0.216989]</br>

- **Hyperparameters**</br>
number of frames per clip = 16</br>
maximum number of clips per video to consider =5 </br>
batch-size =8 </br>
epochs=25</br>
number of data loading workers=16</br>
initial learning rate=.01**</br>
momentum=.0</br>
weight decay = 1e-4**</br>
decrease lr on milestones=[20, 30, 40]</br>
decrease lr by a factor of lr-gamma=0,1</br>
number of warmup epochs=10</br>
number of classes: 400</br>

## <a id="Imgen">4.0 Generating Training Data
  
## 4.1 Generating train data
We relied on videos from Youtube covering pedestrian actions , cyclists and empty roads.
  
## 4.2 Testing with live feed
The jetson xavier was mounted on the dashboard of the car. The USB cam on the xavier will stream in the video feeds and the pre-trained model will predict if there is a 'pedestrian approaching' or a 'cyclist approaching' in the view of the camera.
The frames recieved from the camera are buffered on the Jetson via a sliding window approach. Each frame initiates a new queue that keeps adding frames until a specified number of frames are collected. e.g. if we are going to do inference on a 3 second clip, assuming we are getting 15 fps from the usb cam on the jetson, we will have 45 frames in a queue, which will then be used for inference and then the frames will be discarded. Every second a new queue will be created, which means every 15 frames a new queue is created, at the end of 3 seconds we have 3 queues which the first queue having 45 frames, the 2nd one with 30 frames and the 3rd one with 15 frames. That is the maximum number of frames we will have in memory at a given time. As soon as a queue has 45 frames, we run the prediction and drop the frames.

## <a id="Train">5.0 Training the Model

### 5.1 Results
- Validation Accuracy 1 = 50.909 
- Validation Accuracy 5 = 100.000

### 5.2 Metrics
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/learning%20rate.svg" width="500" height="500" alt="Learning Rate"/>
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/learning%20rate.svg" alt="Your image title" width="400"/>
  
## References
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset - https://arxiv.org/pdf/1705.07750.pdf </br>
A Closer Look at Spatiotemporal Convolutions for Action Recognition - https://arxiv.org/pdf/1711.11248v3.pdf </br>
Video Classification Using 3D ResNet - https://github.com/kenshohara/video-classification-3d-cnn-pytorch </br>
