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

[7.0 E2E Run](#E2E)

## <a id="Introduction">1.0 Introduction

### 1.1 A _Very_ Brief Overview of Action Recognition
Human action recognition is a standard Computer Vision problem and has been well studied. The fundamental goal is to analyze a video to identify the actions taking place in the video. Essentially a video has a spatial aspect to it ie. the individual frames and a temporal aspect ie. the ordering of the frames. Some actions (eg. standing, running, etc.) can probably be identified by using just a single frame but for more complex actions(eg. walking vs running, bending vs falling) might require more than 1 frame’s information to identify it correctly. Local temporal information plays an important role in differentiating between such actions. Moreover, for some use cases, local temporal information isn’t sufficient and you might need long duration temporal information to correctly identify the action or classify the video.Deep learning approaches have empirically demonstrated remarkable success in learning image representations for tasks like object recognition, image captioning, and semantic segmentation. Convolutional neural networks have enabled us to efficiently capture the hypothesis of spatial locality of data structure in images through parameter sharing convolutions, and local invariance-building max-pooling neurons. In this project, we would like to explore the impact of deep learning techniques on video tasks, specifically action recognition.

### 1.2 Evolution of Action Recognition
![Evolution_History](https://github.com/indr19/Action_Recognition/blob/master/images/Evolution%20History.JPG)

**2014**</br>
In 2014, two important breakthrough papers gave deep learning the start in video recognition. Large-scale Video Classification with Convolutional Neural Networks by Karpathy et. al. and Two-Stream Convolutional Networks for Action Recognition in Videos by Simonyan and Zisserman gave rise to the popularity of single stream and two stream networks in action recognition.

**2015**</br>
3D ConvNets were established as the new state of the art in the 2015 research paper Learning Spatiotemporal Features with 3D Convolutional Networks by Du Tran et. al . In this paper, they establish that the 3D convolution net (C3D) with a 3x3x3 kernel is the most effective in learning spatiotemporal features.

**2016**</br>
The focus shifted back to two stream networks. In Convolutional Two-Stream Network Fusion for Video Action Recognition by Zisserman et. al. , the authors tackled how to effectively fuse spatial and temporal data across streams and create multi-level loss that could handle long term temporal dependencies. This network was able to better capture motion and spatial features in distinct subnetworks and beat the state of the art IDT and C3D approaches. The multi-level loss is formed by a spatiotemporal loss at the last fusion layer and a separate temporal loss that is formed from output of the temporal net. 

**2017**</br>
 Zhu et. al. took two stream networks a step forward by introducing a hidden stream that learns optical flow called MotionNet [8]. This end-to-end approach allowed the researchers to skip explicitly computing optical flow. This means that two streams approaches could now be real-time and errors from misprediction could also be propagated into MotionNet for more optimal optical flow features.
 
 **2018**</br>
 Many advances in deep residual learning led to novel architectures like 3DResNet and pseudo-residual C3D (P3D)
 
 **2019...**</br>
 Du Tran et. al. propose channel separated convolution networks (CSN) for the task of action recognition in Video Classification with Channel-Separated Convolutional Networks.The researchers build on the ideas of group convolution and depth-wise convolution that received great success in Xception and MobileNet models.Fundamentally, group convolutions introduce regularisation and less computations by not being fully connected. This network effectively captures spatial and spatiotemporal features in their own distinct layers. The channel separated convolution blocks learns these features distinctly but combines them locally at all stages of convolution. This alleviates the need to perform slow fusion of temporal and spatial two stream networks. 


### 1.3 Intent of this Project
Our project goal is to
- Detect 'unmindful pedestrians/'cyclists' on the road
- Save the video clip where the action was detected
- Stream the live feed to an App on the phone or a web application 
- Sound an alarm when a potential action is detected

This might help solve problems like but not limited to
- Road accidents in self driving cars
- Shoplifting 
- Human translation



*[Return to contents](#Contents)*
## <a id="System_Overview">2.0 System Overview
  
### 2.1 Use Case
In the current project ,we plan to detect 'unmindful prdestrains/'cyclists'.  On prediction of the next motion we have plan to
- Sound an alarm (can be done on the xavier itself, but we would need a buzzer etc.)
- Save the video clip where the action was detected
- Stream the live feed to an App on the phone or a web application and highlight when a potential action is detected

### 2.2 Assumptions
- The training videos are of goos quality and resolution
- All videos are more than 8 secs
- The training data is labelled correctly

### 2.3 Components
We are using a R(2D+1) Model trained  We are using Jetson Xavier NX for our inference. Trained models are saved over to Jetson device and used for testing.The USB cam on the xavier will stream in the video feeds and the pre-trained model will predict if there is a 'pedestrian approaching' or a 'cyclist approaching' in the view of the camera.
Note: this is different than just detecting if there is a pedestrian in the frame, while driving on the streets, there will be predestrians in the view, our approach here is to detect when the pedestrain is dangerously close to the vehicle or moving in a way that could potenially mean them intercepting the path of the moving vehicle. It's easy for humans to detect such situations since we have plethora of experiences detecting when a situation may develop with the slightest of the hints. 

1. The Cloud
    * NVIDIA Deep Learning AMI v20.11.0-46a68101-e56b-41cd-8e32-631ac6e5d02b
    * g4dn.2xlarge
    * Configure the Virtual m/c
      * aws ec2 create-security-group --group-name hw09 --description "FinalProj" --vpc-id vpc-id
      * aws ec2 authorize-security-group-ingress --group-id security_group_id --protocol tcp --port 1-65535 --cidr 0.0.0.0/0
      * aws ec2 run-instances --image-id ami-05637fb3a5183e0d0 --instance-type g4dn.2xlarge --security-group-ids security_group_id --associate-public-ip-address --key-name key --count 1

2. The Data
    * Kinetics400 dataset, a benchmark dataset for human-action recognition. The accuracy is reported on the traditional validation split.
    * Labelled data from Youtube

3. The Edge Device
    * Jetson Xavier NX

### 2.4 System Design
![System diagram](https://github.com/indr19/Action_Recognition/blob/master/images/W251%20System%20Design_Final.jpg)

## <a id="Model">3.0 The Model
We use a “(2+1)D” convolutional block, which explicitly factorizes 3D convolution into two separate and successive operations, a 2D spatial convolution and a 1D temporal convolution.
R(2+1)D are ResNets with (2+1)D convolutions. For interpretability, residual connections are omitted.
The first advantage is an additional nonlinear rectification between these two operations. This effectively doubles the number of nonlinearities compared to a network using full 3D convolutions for the same number of parameters, thus rendering the model capable of representing more complex functions.
The second potential benefit is that the decomposition facilitates the optimization, yielding in practice both a lower training loss and a lower testing loss.

*[Return to contents](#Contents)*

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
number of classes: 3</br>


*[Return to contents](#Contents)*

## <a id="Imgen">4.0 Generating Training Data
  
## 4.1 Generating train data
We relied on videos from Youtube covering 3 classes 
* pedestrians
* cyclists
* no pedestrians
  
## 4.2 Testing with live feed
The jetson xavier was mounted on the dashboard of the car. The USB cam on the xavier will stream in the video feeds and the pre-trained model will predict if there is a 'pedestrian approaching' or a 'cyclist approaching' in the view of the camera.
The frames recieved from the camera are buffered on the Jetson via a sliding window approach. Each frame initiates a new queue that keeps adding frames until a specified number of frames are collected. e.g. if we are going to do inference on a 3 second clip, assuming we are getting 15 fps from the usb cam on the jetson, we will have 45 frames in a queue, which will then be used for inference and then the frames will be discarded. Every second a new queue will be created, which means every 15 frames a new queue is created, at the end of 3 seconds we have 3 queues which the first queue having 45 frames, the 2nd one with 30 frames and the 3rd one with 15 frames. That is the maximum number of frames we will have in memory at a given time. As soon as a queue has 45 frames, we run the prediction and drop the frames.

## <a id="Train">5.0 Training the Model
We built a Docker container to facilitate training in the cloud. The container is built on the base Pytorch containerand facilitates deploying instances to allow simultaneous training of models . The code in this repo is largely a reuse of the pytorch vision video classification code from here https://github.com/pytorch/vision.git
While vision/references/video_classification/train.py in the pytorch repo uses PyAV to process the videos, here we do not use PyAV, we instead use a sequence of image files to create the training dataset. The downloader downloads videos from youtube as a collection of images and also prepares an annotation file.


*[Return to contents](#Contents)*

### 5.1 The Preprocessor
* Prepare the *training list*, the ones we wish to download from YouTube and tag them appropriately
  * Each entry in the video list needs to be of the format:
  {'url':"\<url of the video>", 'category':'\<category>', 'start': \<start seconds>, 'end': \<end seconds>}
  * e.g., the list file should look like, start and end are time in seconds, category is the label which should be known
  [{'url':"\<url>", 'category': "\<cat>", 'start': 506, 'end': 508}, {'url':"\<url>", 'category': "\<cat>", 'start': 123, 'end': 127}]

* Pull docker image & run the cdocker container
  * docker pull mayukhd/torch1.7:videoclassification
  * docker run -it --rm --runtime=nvidia -p 8888:8888 -p 6006:6006 -v ~/Action_Recognition/torchvideoclf:/app mayukhd/torch1.7:videoclassification

* Download the images from YouTube using the downloader utility
  * python3 download.py --train_video_list=<full path to the training list> --dataset_traindir=<full path to where the image sequences for training should be saved> --val_video_list=<full path to the test list> --dataset_valdir=<full path to where the image sequences for validation should be saved>

### 5.2 The Trainer
* Train & validate 
  * The code uses GPU by default, you can change it via the --device parameter when running
  * python3 train.py --train-dir=dataset/train --val-dir=dataset/val --output-dir=checkpoint --pretrained
  * run tensorboard --logdir=runs in another session
  * goto https://<url>:6006 to view the training metrics

### 5.3 The Saver
* Saves the model checkpoints
* The checkpoint file is scped to the edge device for inferencing
  * scp -i key -r user@aws public dns :/home/ubuntu/Action_Recognition/torchvideoclf/checkpoint /.* 

### 5.3 Results
- Validation Accuracy  = 50.909 

### 5.4 Metrics
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/lr.JPG" width="400"/>
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/trin_acc.JPG" width="400"/>
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/train_loss.JPG" width="400"/>
<img src="https://github.com/indr19/Action_Recognition/blob/master/metrics/val_acc.JPG" width="400"/>

*[Return to contents](#Contents)*

## <a id="Edge">6.0 Inferencing at the Edge
Download the docker image that will be used to run the inference on the jetson
  * docker pull mayukhd/jetson_4_1:cp36torch1.7
    * Remember to run the container on Jetson with --device=/dev/video0 flag
    * Change the following in the index.html file to match your Jetsons IP



### 6.1 The Detector
The feed detctor captures the live stream using the USG cam on the xavier and forwards it to the inferencer. The frames recieved from the camera are buffered on the Jetson via a sliding window approach. Each frame initiates a new queue that keeps adding frames until a specified number of frames are collected. e.g. if we are going to do inference on a 3 second clip, assuming we are getting 15 fps from the usb cam on the jetson, we will have 45 frames in a queue, which will then be used for inference and then the frames will be discarded. Every second a new queue will be created, which means every 15 frames a new queue is created, at the end of 3 seconds we have 3 queues which the first queue having 45 frames, the 2nd one with 30 frames and the 3rd one with 15 frames. That is the maximum number of frames we will have in memory at a given time. As soon as a queue has 45 frames, we run the prediction and drop the frames.

* Download the test images 
  * python3 download.py --val_video_list= full path to the test list --dataset_valdir= full path to where the image sequences
* Setup Jetson for inststructions
  *  For setting up jetson please refer to the [Jetson Setup_Instructions](https://github.com/indr19/Action_Recognition/blob/master/README_Jetson_Setup.md)
  
### 6.2 The Inferencer
The inference container runs the model that was trained in the cloud. On receipt of an feed, the container further preprocesses the image, feeds the processed image forward through the network and predicts the class of the video clip. We also provide a measure of accuracy (using the ground truth which is embedded in the file names passed through).
Fetch the checkpoints from the system where you ran your training, e.g., if you ran your training in the cloud you would need to download the checkpoint named 'checkpoint.pth' file which will be in the location specified in --output-dir

* Run test 
  * python3 test.py --test-dir= test image seq. dir --resume-dir= full path to checkpoint file
  
* Test Results
  * Test Accuracy = 68.000 

### 6.3 The Alarm Generator

* Ensure the following before running the app

  * Ensure you have the serial cable plugged in to the Jetson and the screen tool is used to open a session to the Jetson.
  * Using the Ubuntu GUI (use VNC Viewer), create a new Wifi connection on the Xavier of mode = Hotspot
  * Connect the xavier to this new Wifi network
  * Connect your phone/laptop to this Wifi network
  * Both the phone and the xavier will get a 10.42.* address when on this network. You can use this IP to talk to the xavier from the phone.

* Stream video from Jetson to a web browser
Read the frame from the video feed of the camera encode the image in JPEG format convert the encoded image to bytearray format send the bytearray as image/jpeg content type in the http response Content-Type: image/jpeg bytearray(jpeg_encoded_image)
  * yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

* Get the docker image to use on the Jetson, and run the container with access to the camera
sudo docker run -it --rm --runtime=nvidia --device=/dev/video0 -v ~/w251/finalproject/app:/app -p 8888:8888 mayukhd/jetson_4_1:cp36torch1.7

* Install the dependencies 
  * pip3 install -r requirements.txt
  * pip3 install -r requirements_dev_macos.txt

* Run the app
  * python3 livedetect.py --resume-dir=checkpoint.pth
  
* Get prediction on your ios phone
*   Navigate to the http://<jetson ip>:8080/ on your browser
*   Open debug view in your browser, you should see the console printing essages that it is receiving communication from the server via sockets in an async manner.


*[Return to contents](#Contents)*

## References
Base Code from pytorch - https://github.com/pytorch/vision/tree/master/references/video_classification </br>
Kinetics Dataset - https://deepmind.com/research/open-source/kinetics</br>
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset - https://arxiv.org/pdf/1705.07750.pdf </br>
A Closer Look at Spatiotemporal Convolutions for Action Recognition - https://arxiv.org/pdf/1711.11248v3.pdf </br>
Video Classification Using 3D ResNet - https://github.com/kenshohara/video-classification-3d-cnn-pytorch </br>
Large-scale Video Classification with Convolutional Neural Networks - https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf </br>
Awesome Action Recognition - https://github.com/jinwchoi/awesome-action-recognition </br>
Deep Learning Models for Human Activity Recognition - https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/ </br>
Pedestrian and Cyclist Detection and Intent Estimation for Autonomous Vehicles - https://www.mdpi.com/2076-3417/9/11/2335/htm#B48-applsci-09-02335 </br>
Deep Learning Architectures for Action Recognition - https://towardsdatascience.com/deep-learning-architectures-for-action-recognition-83e5061ddf90 </br>
Literature Survey: Human Action Recognition - https://towardsdatascience.com/literature-survey-human-action-recognition-cc7c3818a99a </br>
