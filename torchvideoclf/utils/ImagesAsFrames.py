
from utils.ImagesAsVideoDataset import *
from utils.ImageSequenceProcessor import *
import torch
import bisect
import cv2

"""
    The object that holds all the video frames from the entire video list
    This will allow us access to the content of the frames
"""

class ImagesAsFrames(object):
    """
        uses the annotations to populate the clips/frames
        implements a getter to extract actual content for a given frame by acessing the corresponding image file on the disk
    """

    def __init__(self, annotations, frames_per_clip=16, frames_between_clips=1, frame_rate=None):
        """
            frames_between_clips: how many frames we want to skip when loading the frames into the sequences, default=1
        """
        self.annotations = annotations
        self.frames_per_clip = frames_per_clip
        self.frames_between_clips = frames_between_clips
        self.frame_rate = frame_rate

        self.getpts()

        self.compute_clips(frames_per_clip, frames_between_clips, frame_rate)

    def _collate_fn(self, x):
        return x

    """
        Get all the frames in the video arranged in an array
    """

    def getpts(self):
        self.video_pts = []
        self.video_fps = []
        dl = torch.utils.data.DataLoader(
            ImagesAsVideoDataset(self.annotations),
            batch_size=6,
            num_workers=1,
            collate_fn=self._collate_fn  # collate function is must
        )
        for d in dl:
            clips, fps = list(zip(*d))
            clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
            self.video_pts.extend(clips)
            self.video_fps.extend(fps)

    def compute_clips(self, frames_per_clip, step_between_frames, frame_rate):
        self.clips = []
        for v_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = get_clip_sequences_from_video(v_pts, frames_per_clip, step_between_frames, fps, frame_rate)
            self.clips.append(clips)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_frame(self, idx):
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range ({} number of clips)".format(idx, self.num_clips()))
        # get the clip location
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.cumulative_sizes[video_idx - 1]
        video_path = self.annotations[video_idx]
        frame_idxs = self.clips[video_idx][frame_idx]
        # all these frames are basically images in the folder video_path, load them here
        # how many frames are there in the video, simply check the number of images in the path
        if len(frame_idxs) <= 0:
            raise IndexError("frame_idxs not available for idx {}".format(idx))
        num_frames = len(frame_idxs)
        image_name = video_path['path'] + "/" + "img_{}.jpg".format(frame_idxs[-1])
        frame = cv2.imread(image_name)
        height, width, channels = frame.shape
        # initialize tensor
        frames = torch.FloatTensor(num_frames, height, width, channels)
        for idx in range(len(frame_idxs)):
            image_name = video_path['path'] + "/" + "img_{}.jpg".format(frame_idxs[idx])
            frame = cv2.imread(image_name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            # frame = frame.permute(2, 0, 1)
            # this conversion is needed to ensure that the frames can be recognized by PIL image
            frames[idx, :, :, :] = frame.type(torch.uint8)
        return frames.type(torch.uint8), video_idx