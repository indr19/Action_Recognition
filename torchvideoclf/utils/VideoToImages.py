import os.path
from utils.VideoDownloader import *
import imageio
from PIL import Image
import os
import traceback
import sys
import ast
import json


class annots():
    def __init__(self, path, start, end, cls, fps):
        self.path = path
        self.start = start
        self.end = end
        self.cls = cls
        self.fps = fps
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)
"""
    Downloads videos from youtube and generates images for training
    This routine will download the videos from youtube, rescale them as per the new width provided
    It will also generate the annotations file that will have the TOC for the videos in the downloaded path, annotated with the classes
    The class ids are automatically derived from the labels provided in the video list
"""
def YoutubeVideoToImages(video_list_path, clip_len=None, save_images=True, save_videos=False, rootpath=None, videopath=None):
    """
        Arg: video_list is a dictionary of url (a youtube video link) and the category: string having the format
             {url, start_seconds, end_seconds, category/class/label} of the video
             Optional: clip_len: int, length of the clip in seconds, default = 30, used when end_seconds are not provided
                       save_images: save the images from the frames to directory named after the category, default = True
                       save_videos: save the videos to a directory in videopath, default = False
                       rootpath: the directory for the training images
                       videopath: the directory for the extracted videos
             Returns: None, saves an annotations file with the path to the images for videos, the start frame, end frame, the class and the FPS
                      Saves the video as images in the rootpath / <category> / <video id: numeric, integer corresponding to the frame count>
                      The naming of the images should be strictly monotonically increasing integer starting from 0 to the frame_count.
    """
    annotationsfile = "annotations.txt"
    annotations = []
    clsses = {}
    free_ids = 0
    defaultcliplen = 30  # seconds
    if clip_len is not None:
        cliplen = int(clip_len)
    else:
        cliplen = defaultcliplen
    if rootpath is None:
        raise ValueError("rootpath must be provided.")
    vid_idx = 0
    # clean up old annotations file
    annotationsfilepath = rootpath + "/" + annotationsfile
    try:
        os.remove(annotationsfilepath)
    except:
        print("Error removing the annotations file {}".format(annotationsfilepath))
    #open the video list
    print("Opening video list {}".format(video_list_path))
    with open(video_list_path, "r") as f:
        video_list = f.read()
    res = ast.literal_eval(video_list)
    print("Fetching {} videos from list {} ".format(len(res), video_list_path))
    for vl in res:
        try:
            # get the metadata for the video
            if 'url' in vl:
                url = vl['url']
            else:
                raise Exception("No video url provided, skipping...")
            m3u8, width, height, fps = getproperties(url)
            # get the start and end frame requirements for the video, if provided
            if 'start' in vl:
                start = int(vl['start'])
            else:
                start = 0
            if 'end' in vl:
                end = int(vl['end'])
            else:
                end = cliplen
            # get the category of the video
            if 'category' in vl:
                category = vl['category']
            else:
                raise Exception("No category specified for the video {}".format(url))
            # get the class id for the category
            if category not in clsses:
                clsses[category] = free_ids
                free_ids += 1
            class_id = clsses[category]
            print("Class id is {}".format(class_id))
            print("Downloading from {} to {} of length {}".format(start, end, end - start))
            print("Sending width {} to {}".format(width, width / 4))
            out, prop = downloadvideo(m3u8, fps, start, end - start, width / 4)
            width, height = getsize(prop)
            print("New size {} and {}".format(width, height))
            # the numpy array of the video
            video = getvideo(out, height, width)
            if videopath is not None:
                pathtovideo = os.path.join(videopath + "/" + category)
                if not os.path.exists(pathtovideo):
                    os.makedirs(pathtovideo)
                fullpath = pathtovideo + "/" + "video{}.mp4".format(vid_idx)
                print("Saving video to {}".format(fullpath))
                imageio.mimwrite(fullpath, video, fps=fps)
            if rootpath is not None:
                # create a folder to save the images from the video, some arbritrary naming
                # we will use the index of the video
                pathtoimages = os.path.join(rootpath + "/" + category + "/" + str(vid_idx))
                if not os.path.exists(pathtoimages):
                    os.makedirs(pathtoimages)
                for idx in range(video.shape[0]):
                    fullpath = pathtoimages + "/" + "img_{}.jpg".format(idx)
                    im = Image.fromarray(video[idx])
                    im.save(fullpath)
                frame_cnt = video.shape[0]
                print("Saved {} images in {}".format(frame_cnt, pathtoimages))
                path = "{}/{}".format(category, vid_idx)
                an = annots(path=path, start=0, end=frame_cnt, cls=class_id, fps=fps)
                annotations.append(an)
                # write the image paths to the annotations file, path, start frame, end frame, class id, fps
                # with open(annotationsfilepath, 'a') as filetowrite:
                #     filetowrite.write("{}/{} {} {} {} {}\n".format(category, vid_idx, 0, frame_cnt, class_id, fps))

            vid_idx += 1
        except Exception as e:
            print(e)
            traceback.print_exception(*sys.exc_info())
            print("Error in processing the video at {}".format(vl['url']))
            continue

    with open(annotationsfilepath, 'a') as f:
        f.write('[')
        cntr = 0
        num_ele = len(annotations)
        for an in annotations:
            f.write(an.toJSON())
            if cntr < num_ele - 1:
                f.write(',')
            cntr += 1
        f.write(']')