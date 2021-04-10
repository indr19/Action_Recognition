#Connect to AWS instance

aws ec2 describe-vpcs 
aws ec2 create-security-group --group-name hw09 --description "FinalProj" --vpc-id vpc-id

aws ec2 run-instances --image-id ami-05637fb3a5183e0d0 --instance-type g4dn.2xlarge --security-group-id sec_grp_id --associate-public-ip-address --key-name key --count 1

aws ec2 authorize-security-group-ingress --group-id sec_grp_id --protocol tcp --port 1-65535 --cidr 0.0.0.0/0

ssh -i key user@public_dns_AWS

docker run -it --rm --runtime=nvidia -p 8888:8888 -p 6006:6006 -v ~/Action_Recognition/torchvideoclf:/app mayukhd/torch1.7:videoclassification

docker exec -it conatiner id sh

python3 download.py --train_video_list=data/videos_train_list.txt --dataset_traindir=/dataset/train --val_video_list=data/videos_val_list.txt --dataset_valdir=/dataset/val

python3 train.py --train-dir=/dataset/train --val-dir=/dataset/val --output-dir=checkpoint --pretrained

tensorboard --logdir=runs

#Install nvida docker
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

#Add the docker repo    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

#Copy model checkpoint to your jetson
scp -r user@remote:src_directory dst_directory

