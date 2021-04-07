import torch
import math

"""
Process the images from the downloaded folder as video frames and prepare datasets for training
Most of the code here is from the pytorch github
"""


"""
    Compute which frame indices we need to take given the reduced frame rate
    e.g., if you have 100 fps, and we want only 70 fps, we want only 70% of the frames
    i.e., num_frames * 0.7 of the frames will need to be sampled from the video
    if we had 203 frames, we will need to sample 142 from 203 frames, 
    but we need to decide which of the 142 frames from the 203 needs to be chosen.
    This is sampled by multiplying the frame indices for 142 frames (0-142) with a fraction
"""
def sample_frames_from_video(num_frames, original_fps, new_fps):
    step = float(original_fps) / new_fps
    if step.is_integer():
        # optimization: if step is integer, don't need to perform
        # advanced indexing
        step = int(step)
        return slice(None, None, step)
    idxs = torch.arange(num_frames, dtype=torch.float32) * step
    idxs = idxs.floor().to(torch.int64)
    return idxs

"""
    From the resampled frame indices, select frames_per_clip frames with a stride 
    of frames_between_clips
"""
def unfold(tensor, frames_per_clip, frames_between_clips, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors
    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (frames_between_clips * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (frames_per_clip - 1) + 1)) // frames_between_clips + 1, frames_per_clip)
    if new_size[0] < 1:
        new_size = (0, frames_per_clip)
    return torch.as_strided(tensor, new_size, new_stride)

"""
    Compute a tensor pair for the video that will give the sequence of frames from the videos, N clips per sequence
    This also allows us to resample the frames using a new frame rate
"""
def get_clip_sequences_from_video(video_frame_ids, num_frames_in_clip, frames_between_clips, fps, desired_frame_rate):
    """
        video_frame_ids: the frames indices of the video arranged in a 1-d tensor
        num_frames_in_clip: how many frames we want in each sequence of frames
        frames_between_clips: how many frames we want to skip when loading the frames into the sequences, default=1
        fps: original frames per second of the video
        desired_frame_rate: what is the new frames per second we want, we want to resample the frames based on this new value

        return: clips, idxs Tensors, the tensor of clips with the frame indices and the tensor of indices of the frames
        Note: here clips and idxs convey identical information here, so there is a potential to remove one of them in the future
    """
    if fps is None:
        # if for some reason the video doesn't have fps (because doesn't have a video stream)
        # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
        fps = 1
    if desired_frame_rate is None:
        desired_frame_rate = fps
    # create a 1-d tensor of all the frame indices in the video, we do not need any content from the video yet
    # video_frame_ids = torch.arange(original_frame_cnt)
    total_frames = len(video_frame_ids) * (float(desired_frame_rate) / fps)
    idxs = sample_frames_from_video(int(math.floor(total_frames)), fps, desired_frame_rate)
    resampled_video_frame_ids = video_frame_ids[idxs]
    clips = unfold(resampled_video_frame_ids, num_frames_in_clip, frames_between_clips)
    if isinstance(idxs, slice):  # i.e. the quotient was an integer
        idxs = [idxs] * len(clips)
    else:
        idxs = unfold(idxs, num_frames_in_clip, frames_between_clips)
    return clips, idxs