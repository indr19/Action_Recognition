import numpy as np
import youtube_dl
import re
import ffmpeg


"""
    Exract the properties of the video on youtube given the Url
"""
def getproperties(url):
    """
        url: the url of the video on youtube
    """
    m3u8, width, height, fps = None, None, None, None
    try:
        yl = youtube_dl.YoutubeDL({'format': 'best'})
        quality = yl.extract_info(url, download=False)
        m3u8 = quality['url']
        width = quality['width']
        height = quality['height']
        fps = quality['fps']
    except Exception as e:
        print("Error downloading the metadata for the video url {}, error {}".format(url, e.__class__))
        raise Exception("Metadata fetch error")
    return m3u8, width, height, fps
""" Get the size of the video frames from the metadata of the downloaded stream
    This has been tested to work with the metadata from the youtube video streams only
"""
def getsize(prop):
    """
    Args:
        a byte array
    Returns:
        (int, int): width and height of the video frames
    """
    grp = None
    width = None
    height = None
    #get the part of the bytes after rgb24
    p = re.compile(rb'rgb24\s*(.*)')
    try:
        m = p.search(prop)
        if m:
            grp = m.group(1)
        p = re.compile(rb'((\d+)(\s|x|\.)(\d+))')
        m = p.search(grp)
        try:
            if m:
                width = int(m.group(2))
                height = int(m.group(4))
        except:
            print("Error when extracting the size from the downloaded video stream!")
    except:
        print("Error parsing the properties of the downloaded video")
        raise Exception("Get size error")
    return width, height


"""
    Download the video using the video stream link, fps and a provided width for scaling the frames
"""
def downloadvideo(m3u8, fps, startsecs, durationsecs, width):
    try:
        print("Getting video for width {}".format(width))
        out, prop = (
                ffmpeg
                .input(m3u8, ss=startsecs, t=durationsecs) #use this instead of trim
                #.trim(start_frame=stfr, end_frame=enfr) #buggy, does not work!
                .filter('fps', fps=fps, round='up')
                .filter('scale', width, -1)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet = True))
    except Exception as e:
        print("Error reading the video stream, error was {}".format(e.__class__))
        raise Exception("Download error")
    return out, prop

"""
    From video bytearray buffer to numpy array
"""
def getvideo(out, width, height):
    """
        out: the bytearray buffer
        width: width of video
        height: height of video
    """
    try:
        video = (np.frombuffer(out, np.uint8).reshape([-1, width, height, 3]))
    except:
        print("Error reshaping video byte array to numpy array")
        raise Exception("Fetch video error")
    return video