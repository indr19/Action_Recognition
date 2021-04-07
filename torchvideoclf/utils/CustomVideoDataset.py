
import torch.nn as nn
import torch
from torchvision.transforms import transforms
from utils.ImagesAsFrames import *
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from typing import Optional, List, Iterator, Sized, Union, cast
import json

class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)
class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class VideoClassificationPresetTrain:
    def __init__(self, resize_size, crop_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989),
                 hflip_prob=0.5):
        trans = [
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)

class VideoClassificationPresetEval:
    def __init__(self, resize_size, crop_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose([
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
            transforms.Normalize(mean=mean, std=std),
            transforms.CenterCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])

    def __call__(self, x):
        return self.transforms(x)


class UniformClipSampler(Sampler):
    """
    Sample `num_video_clips_per_video` clips for each video, equally spaced.
    When number of unique clips in the video is fewer than num_video_clips_per_video,
    repeat the clips until `num_video_clips_per_video` clips are collected
    Args:
        video_clips (VideoClips): video clips to sample from
        num_clips_per_video (int): number of clips to be sampled per video
    """
    def __init__(self, video_frames: ImagesAsFrames, num_clips_per_video: int) -> None:
        if not isinstance(video_frames, ImagesAsFrames):
            raise TypeError("Expected video_clips to be an instance of VideoFrames, "
                            "got {}".format(type(video_frames)))
        self.video_frames = video_frames
        self.num_clips_per_video = num_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select num_clips_per_video for each video, uniformly spaced
        for c in self.video_frames.clips:
            length = len(c)
            if length == 0:
                # corner case where video decoding fails
                continue

            sampled = (
                torch.linspace(s, s + length - 1, steps=self.num_clips_per_video)
                .floor()
                .to(torch.int64)
            )
            s += length
            idxs.append(sampled)
        return iter(cast(List[int], torch.cat(idxs).tolist()))

    def __len__(self) -> int:
        return sum(
            self.num_clips_per_video for c in self.video_frames.clips if len(c) > 0
        )


class RandomClipSampler(Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly
    Args:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """
    def __init__(self, video_frames: ImagesAsFrames, max_clips_per_video: int) -> None:
        if not isinstance(video_frames, ImagesAsFrames):
            raise TypeError("Expected video_frames to be an instance of VideoFrames, "
                            "got {}".format(type(video_frames)))
        self.video_frames = video_frames
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_frames.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_frames.clips)

"""
    Use the video frame indices for tracking and load videos only when needed by loader
"""


class VideoDatasetCustom(Dataset):
    """
        Using an root datapath that contains the videos stored as collection of images
        and the annotations file, create the torch Dataset that gives us access to the sequence of video
        frames defined as per our resampling and sequencing strategy
        frames_per_clip: number of frames we want in the sequence we want to process during training, default = 16
        frames_between_clips: number of steps we need between the frames when sampling the sequence, default is 1
        frame_rate: the new desired frame rate, to resample from the current frames to match this new frame rate
        transform: Leave as None for now, ideally we wish to do mean and std normalization here TODO
    """

    def loadannotations(self, dataset_path, annotations_file):
        annotations = []
        file = dataset_path + "/" + annotations_file
        print("Opening annotations file {}".format(file))
        with open(file, "r") as f:
            j = f.read()
            rawjson = json.loads(j)
        for rec in rawjson:
            #change to json read
            #data = l.strip().split(' ')
            # if len(data) == 5:  # we have all information
            #     annot = {'path': dataset_path + "/" + data[0], 'frame_start': data[1], 'frame_end': data[2],
            #              'class': data[3], 'fps': data[4]}
            rec['path'] = dataset_path + "/" + rec['path']
            annotations.append(rec)
        return annotations

    def __init__(self, dataset_path, annotations_file, transform=None):
        self.transform = transform
        self.rootpath = dataset_path
        self.annotations = self.loadannotations(dataset_path, annotations_file)
        self.clips = ImagesAsFrames(self.annotations, frames_per_clip=16, frames_between_clips=1, frame_rate=15)

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return self.clips.num_clips()

    def __getitem__(self, index):
        """ get a video """
        frame, video_idx = self.clips.get_frame(index)
        video = self.annotations[video_idx]
        label = int(video['cls'])
        if self.transform:
            frame = self.transform(frame)
        return frame, label