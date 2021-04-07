from typing import List
import numpy as np

"""
    The dataset for reading in the frame indices and the fps for each of the videos
"""
class ImagesAsVideoDataset(object):
    """
        A dataset for the videos in the video path/root path of the data
    """
    #each annotation is equivalent to a video path
    def __init__(self, annotations: List[dict]):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
            args:
                idx: the index of the item to get
            get the frame count and fps for the videos in the list,
            this is the frames of the video arranged in a list and fps of the original downloaded video
        """
        annot = self.annotations[idx]
        print(annot)
        frame_cnt = int(annot['end'])
        fps = int(annot['fps'])
        video_frame_idxs = list(np.arange(frame_cnt))
        return video_frame_idxs, fps