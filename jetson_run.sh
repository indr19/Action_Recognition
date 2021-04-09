sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker


docker run -it --rm --runtime=nvidia -p 8888:8888 -p 6006:6006 -v ~/DeepLearning/Action_Recognition/torchvideoclf:/app mayukhd/jetson_4_1:cp36torch1.7

python3 download.py --val_video_list=data/videos_test_list.txt --dataset_valdir=/dataset/test

python3 test.py --test-dir=/dataset/test --resume-dir=checkpoint/checkpoint.pth

scp  -r indra@mc ip:C:/checkpoint ~/DeepLearning/.

