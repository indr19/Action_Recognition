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
from torch.utils.tensorboard import SummaryWriter



def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def collate_fn(batch):
    return default_collate(batch)

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, writer):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))
    running_loss = 0.0
    running_accuracy = 0.0
    header = 'Epoch: [{}]'.format(epoch)
    cntr = 0
    for video, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        output = model(video)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        running_loss += loss.item()
        running_accuracy += acc1.item()
        if cntr % 10 == 9: #average loss over the accumulated mini-batch
            writer.add_scalar('training loss',
                              running_loss / 10,
                              epoch * len(data_loader) + cntr)
            writer.add_scalar('learning rate',
                              optimizer.param_groups[0]["lr"],
                              epoch * len(data_loader) + cntr)
            writer.add_scalar('training accuracy',
                              running_accuracy / 10,
                              epoch * len(data_loader) + cntr)
            running_loss = 0.0
            running_accuracy = 0.0
        cntr = cntr + 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, epoch, criterion, data_loader, device, writer):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    cntr = 0
    running_accuracy = 0.0
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
            running_accuracy += acc1.item()
            if cntr % 10 == 9:  # average loss over the accumulated mini-batch
                writer.add_scalar('validation accuracy',
                                  running_accuracy / 10,
                                  epoch * len(data_loader) + cntr)
                running_accuracy = 0.0
            cntr += 1
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg

def main(args):
    if args.output_dir:
        makedir(args.output_dir)
    print(args)
    device = torch.device(args.device)

    transform_train = VideoClassificationPresetTrain((128, 171), (112, 112))
    dataset_train = VideoDatasetCustom(args.train_dir, "annotations.txt", transform=transform_train)

    transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
    dataset_eval = VideoDatasetCustom(args.val_dir, "annotations.txt", transform=transform_eval)

    train_sampler = RandomClipSampler(dataset_train.clips, args.clips_per_video)
    test_sampler = UniformClipSampler(dataset_eval.clips, args.clips_per_video)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)

    model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    print("Start training")
    writer = SummaryWriter('runs/vc_experiment_1')
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq, writer)
        evaluate(model, epoch, criterion, data_loader_eval, device=device, writer=writer)

        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            print("Saving checkpoint to {}".format(os.path.join(args.output_dir, 'checkpoint.pth')))
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Video classification training using image sequences')
    parser.add_argument('--train-dir', default='dataset_train', help='name of train dir')
    parser.add_argument('--val-dir', default='dataset_test', help='name of val dir')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save the model checkpoint')
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


