# torchvideoclf

### Video classification using pytorch

The code in this repo is largely a reuse of the pytorch vision video classification code from here https://github.com/pytorch/vision.git

While `vision/references/video_classification/train.py` in the pytorch repo uses PyAV to process the videos, here we do not use PyAV, we instead use a sequence of image files to create the training dataset.
The downloader downloads videos from youtube as a collection of images and also prepares an annotation file.

The train.py uses the image collections to prepare the training dataset.

The code in this repo was developed on this docker image *mayukhd/torch1.7:videoclassification*

`docker pull mayukhd/torch1.7:videoclassification`

You should run the below in the above container

#### Steps:

- Prepare the source video list, the ones we wish to download from YouTube and tag them appropriately
    - Each entry in the video list needs to be of the format: 
    
    > `{'url':"\<url of the video>", 'category':'\<category>', 'start': \<start seconds>, 'end': \<end seconds>}`
     
     e.g., the list file should look like, start and end are time in seconds, category is the label which should be known
     
    > `[{'url':"\<url>", 'category': "\<cat>", 'start': 506, 'end': 508},
        {'url':"\<url>", 'category': "\<cat>", 'start': 123, 'end': 127}]`

- Download the images from YouTube using the downloader utility
  - Run this in the container 
  
  `python3 download.py --train_video_list=<full path to the training list> 
  --dataset_traindir=<full path to where the image sequences for training should be saved> 
  --val_video_list=<full path to the test list> 
  --dataset_valdir=<full path to where the image sequences for validation should be saved>`

- Run the train.py to train the model on the images we downloaded
  - The code uses GPU by default, you can change it via the `--device` parameter when running
    
    `python3 train.py --train-dir=dataset/train --val-dir=dataset/val --output-dir=checkpoint --pretrained`
    
    export port 6006 in the container
    
    run `tensorboard --logdir=runs` in another session 
    
    goto https://\<url>:6006 to view the training metrics
    
    ![image](https://user-images.githubusercontent.com/17194414/113135283-92e4df80-923f-11eb-81cd-b0074b34cb3c.png)

- Run a test
    
    We will run our inference on a NVIDIA jetson xavier NX with Jetpack 4.1
    
    Download the docker image that will be used to run the inference on the jetson
    
    `docker pull mayukhd/jetson_4_1:cp36torch1.7`
    
    Fetch the checkpoints from the system where you ran your training, e.g., if you ran your training in the cloud
    you would need to download the checkpoint named 'checkpoint.pth' file which will be in the location specified in --output-dir
    
    Download only the test images 
    `python3 download.py --val_video_list=<full path to the test list> --dataset_valdir=<full path to where the image sequences>`
  
    `python3 test.py --test-dir=\<test image seq. dir> --resume-dir=\<full path to checkpoint file>`
