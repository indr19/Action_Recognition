from utils.VideoToImages import *
from utils.CustomVideoDataset import *
from utils.WarmupMultiStepLR import *
from utils.MetricLogger import *
import errno
import os

import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
from torch import nn



def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def collate_fn(batch):
    return default_collate(batch)

def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for video, target in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg

def main(args):

    device = torch.device(args.device)

    transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
    dataset_test = VideoDatasetCustom(args.test_dir, "annotations.txt", transform=transform_eval)

    test_sampler = UniformClipSampler(dataset_test.clips, args.clips_per_video)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)

    #load the pretrained weights for the known model
    model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
    model.to(device)

    #load the pretrained model from the checkpoint file
    if args.resume_dir:
        if not os.path.exists(args.resume_dir):
            raise OSError("Checkpoint file does not exist!")
        else:
            checkpoint_file = args.resume_dir
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

    criterion = nn.CrossEntropyLoss()

    print("Start testing")
    start_time = time.time()
    evaluate(model, criterion, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Video classification training using image sequences')
    parser.add_argument('--test-dir', default='dataset_train', help='name of train dir')
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

if __name__ == "__main__":
    args = parse_args()
    main(args)


