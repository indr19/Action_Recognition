from aiohttp import web
import socketio
import cv2
import asyncio
from utils.VideoToImages import *
from utils.CustomVideoDataset import *
from utils.WarmupMultiStepLR import *
from utils.MetricLogger import *

import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
from torch import nn
from collections import deque

"""
We will be using a socketIO server written in python and a javascript based socketIO client, to the versions must be compatible
python-socketio==4.6.0
javascript: https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.2.0/socket.io.js
as per this https://python-socketio.readthedocs.io/en/latest/intro.html#what-is-socket-io
python-socketio 4.x should be used with javascript sockeIO 1.x and 2.x
"""
connection_flag = 0
#global variable, the model that is used to make predictions
prediction_model = None
"""
    The dataset of the frames from the carmera used for prediction
"""
class FrameDataset():

    def __init__(self, max_frames):
        self.framequeue = deque()
        self.max_frames = max_frames

    def add_frame(self, frame):
        if self.num_frames() > self.max_frames:
            self.framequeue.popleft()
        self.framequeue.append(frame)

    def num_frames(self):
        return len(self.framequeue)

    def get_frames(self):
        return [self.framequeue.popleft() for i in range(self.num_frames())]

    def get_as_dataset(self, min_frames=16):
        transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
        frames = self.get_frames()
        height, width, channels = frames[0].shape
        framestensor = torch.FloatTensor(min_frames, height, width, channels)
        #frame = cv2.imread(image_name)
        print("Frame count = {}".format(len(frames)))
        for idx in range(min_frames-1):
            frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            framestensor[idx, :, :, :] = frame.type(torch.uint8)
        return transform_eval(framestensor.type(torch.uint8))
        # (H x W x C) to (C x H x W)
        # frame = frame.permute(2, 0, 1)
        # this conversion is needed to ensure that the frames can be recognized by PIL image
        #frames[idx, :, :, :] = frame.type(torch.uint8)

    #return frames.type(torch.uint8), video_idx
        # transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
        # dataset_test = VideoDatasetCustom(args.test_dir, "annotations.txt", transform=transform_eval)
        #
        # test_sampler = UniformClipSampler(dataset_test.clips, args.clips_per_video)
        #
        # data_loader_test = torch.utils.data.DataLoader(
        #     dataset_test, batch_size=args.batch_size,
        #     sampler=test_sampler, num_workers=args.workers,
        #     pin_memory=True, collate_fn=collate_fn)


all_frames = FrameDataset(max_frames=150) #at 15 fps, 10 seconds
#code from here https://python-socketio.readthedocs.io/en/latest/intro.html#what-is-socket-io
#and here https://tutorialedge.net/python/python-socket-io-tutorial/
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

cap = cv2.VideoCapture(0)

def gen_frames():
    global all_frames
    while True:
        success, frame = cap.read()  # read a frame from the camera
        all_frames.add_frame(frame)
        if not success:
            break
        else:
            frame = cv2.resize(frame, (480, 320))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


async def video_feed(request):
    response = web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    for frame in gen_frames():
        await asyncio.sleep(0.1)
        await response.write(frame)
    return response


async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('connect')
async def connect_handler(sid, environ):
    global connection_flag
    print("New connection from {}".format(sid))
    connection_flag = 1
    await sio.emit('welcome', "connected")

@sio.event
def disconnect(sid):
    global connection_flag
    print('disconnect ', sid)
    connection_flag = 0



async def send_message():
    global connection_flag
    global all_frames
    try:
        print("Background task started...")
        await asyncio.sleep(1)
        print("Wait till a client connects...")
        while connection_flag == 0:
            await asyncio.sleep(1)
            pass
        print("Waiting 2 seconds..")
        await asyncio.sleep(2)
        while True:
            #print("Now emitting: {}".format(i))
            if all_frames.num_frames() >= 16:
                frames = all_frames.get_as_dataset()
                #print("shape {}".format(frames.shape))
                m = prediction_model.get_model()
                frames = frames[None, :]
                output = m(frames)
                #print("output shape {}".format(output.shape))
                _, pred = output.topk(5, 1, True, True)
                #print("prediction = {}".format(pred))
                p = torch.nn.functional.softmax(output, dim=1)
                # print("output {}".format(output))
                # print("prediction probabilities = {}".format(p))
                vals, preds = p.topk(5, 1, True, True)
                msg = "Best class = {}, best prob = {}".format(preds[0][0], vals[0][0])
                await sio.emit('feedback', msg)
            await asyncio.sleep(0.3)

    finally:
        print("Background task exiting!")

# @sio.on('message')
# async def print_message(sid, message):
#     print("Socket ID: {}".format(sid))
#     print(message)
#     await sio.emit('message', message[::-1])

app.router.add_get('/', index)
app.router.add_get('/videostream', video_feed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Video classification training using image sequences')
    parser.add_argument('--workers', default=4, help='number of workers')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume-dir', default='checkpoint.pth', help='path where the model checkpoint is saved')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    return args

class Model():
    def __init__(self, args):
        self.model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
        self.device = torch.device(args.device)
        self.model.to(self.device)
        if args.resume_dir:
            if not os.path.exists(args.resume_dir):
                raise OSError("Checkpoint file does not exist!")
            else:
                checkpoint_file = args.resume_dir
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                self.model.load_state_dict(checkpoint['model'])

    def get_model(self):
        return self.model





if __name__ == '__main__':
    #global prediction_model
    args = parse_args()
    print("Loading the model for prediction...")
    prediction_model = Model(args)
    # device = torch.device(args.device)
    # # load the pretrained weights for the known model
    # model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
    # model.to(device)

    # if args.resume_dir:
    #     if not os.path.exists(args.resume_dir):
    #         raise OSError("Checkpoint file does not exist!")
    #     else:
    #         checkpoint_file = args.resume_dir
    #         checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'])
    # print("Model loaded from checkpoint!")

    sio.start_background_task(target=lambda: send_message())
    web.run_app(app)
    cap.release()