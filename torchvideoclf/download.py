from utils.VideoToImages import *
import os
import errno

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def main(args):

    if args.clip_len:
        makedir(args.clip_len)


    #if the training list is provided
    if args.train_video_list:
        if os.path.exists(args.train_video_list):
            if args.video_traindir:
                makedir(args.video_traindir)
            if args.dataset_traindir:
                makedir(args.dataset_traindir)
            print("Downloading image sequences for training to {}".format(args.dataset_traindir))
            if args.save_video_clips:
                video_save_dir = args.video_traindir
            else:
                video_save_dir = None
            YoutubeVideoToImages(args.train_video_list, clip_len=args.clip_len, videopath=video_save_dir,
                                 rootpath=args.dataset_traindir)

    # if the val list is provided
    if args.val_video_list:
        if os.path.exists(args.val_video_list):
            if args.video_valdir:
                makedir(args.video_valdir)
            if args.video_testdir:
                makedir(args.video_testdir)
            print("Downloading image sequences for validation to {}".format(args.dataset_valdir))
            if args.save_video_clips:
                video_save_dir = args.video_traindir
            else:
                video_save_dir = None
            YoutubeVideoToImages(args.val_video_list, clip_len=args.clip_len, videopath=video_save_dir,
                                 rootpath=args.dataset_valdir)

    # if the test list is provided
    if args.test_video_list:
        if os.path.exists(args.test_video_list):
            if args.dataset_valdir:
                makedir(args.dataset_valdir)
            if args.dataset_test:
                makedir(args.dataset_test)
            print("Downloading image sequences for test to {}".format(args.dataset_testdir))
            if args.save_video_clips:
                video_save_dir = args.video_traindir
            else:
                video_save_dir = None
            YoutubeVideoToImages(args.train_video_list, clip_len=args.clip_len, videopath=video_save_dir,
                                 rootpath=args.dataset_testdir)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Video downloader from youtube')
    parser.add_argument('--train_video_list', help='name of train dir')
    parser.add_argument('--val_video_list', help='name of val dir')
    parser.add_argument('--test_video_list', help='name of val dir')
    parser.add_argument('--video_traindir', help='name of val dir')
    parser.add_argument('--video_valdir', help='name of val dir')
    parser.add_argument('--video_testdir', help='name of val dir')
    parser.add_argument('--dataset_traindir', help='path where to save')
    parser.add_argument('--dataset_valdir', help='path where to save')
    parser.add_argument('--dataset_testdir', help='path where to save')
    parser.add_argument('--clip_len', default='5', help='default video clip seconds')
    parser.add_argument(
        "--save-video-clips",
        dest="save_video_clips",
        help="if we want the downloader to save the video clips",
        action="store_true",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)