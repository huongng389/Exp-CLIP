import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.Exp_CLIP import ExpCLIP_Train, ExpCLIP_Cached
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
from models.clip import clip
from models.BLIP2_T5 import *
from models.Text import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--batch-size-test-image', type=int, default=512)
parser.add_argument('--batch-size-test-video', type=int, default=48)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--job-id', type=str)
parser.add_argument('--instruction', type=str)
parser.add_argument('--load-model', type=str)
parser.add_argument('--extra-checkpoint-dir', type=str, default=None,
                     help='If set, also copy each epoch checkpoint here (e.g. a mounted Google Drive path), '
                          'so progress survives a Colab runtime loss.')
parser.add_argument('--train-data', type=str,
                     default='/data/EECS-IoannisLab/datasets/Static_FER_Datasets/CAERS_Face/train',
                     help='ImageFolder-style directory of training face images (e.g. CAER-S train).')
parser.add_argument('--sample-weights', type=str, default=None,
                     help='JSON {image path relative to --train-data: weight}; weights each sample in the '
                          'contrastive loss (EXP-001). Images not in the file get weight 1.')
parser.add_argument('--cache-clip', type=str, default=None,
                     help='npz from tools/cache_clip_features.py. With --cache-targets, train on cached '
                          'CLIP features and BLIP-2 targets instead of images (EXP-003): no CLIP/BLIP-2 forward.')
parser.add_argument('--cache-targets', type=str, default=None,
                     help='Directory of chunk_*.npz from tools/precompute_au_weights.py stage target (pool all).')
parser.add_argument('--save-epochs', nargs='+', type=int, default=[],
                     help='Also keep the head after these epochs (1-based) as <job-id>-e<N>-model.pth.')
parser.add_argument('--resume', type=str, default=None,
                     help='Path to a <job-id>-resume.pth file to continue training from the epoch after it.')
args = parser.parse_args()

random.seed(args.seed)  
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now()
train_time = now.strftime("%y-%m-%d %H:%M")
print("Training date: ", train_time)
job_id = args.job_id

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')


def main():

    log_txt_path = './log/' + job_id + '-log.txt'
    log_curve_path = './log/' + job_id + '-log.png'
    checkpoint_path = './checkpoint/' + job_id
    resume_path = checkpoint_path + '-resume.pth'
    train_data_file_path = args.train_data
    recorder = RecorderMeter(args.epochs)

    # create model and load pre_trained parameters
    model = ExpCLIP_Cached(args) if args.cache_clip else ExpCLIP_Train(args)

    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "projection_head" in name:
            param.requires_grad = True  

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # define optimizer
    optimizer = torch.optim.SGD([{"params": model.module.projection_head.parameters(), "lr": args.lr}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if args.cache_clip:
        sample_weights = None
        if args.sample_weights:
            with open(args.sample_weights) as f:
                sample_weights = json.load(f)
        train_dataset = CachedPairs(args.cache_clip, args.cache_targets, sample_weights)
        info = 'Cached training: {} pairs ({} CLIP features without a target skipped), weights mean {:.3f}'.format(
            len(train_dataset), train_dataset.n_skipped, train_dataset.sample_weights.mean().item())
        print(info)
        with open(log_txt_path, 'a') as f:
            f.write(info + '\n')
    elif args.sample_weights:
        with open(args.sample_weights) as f:
            sample_weights = json.load(f)
        train_dataset = WeightedImageFolder(train_data_file_path, train_transform, sample_weights)
        n_found = sum(os.path.relpath(p, train_data_file_path) in sample_weights for p, _ in train_dataset.samples)
        w = torch.tensor(train_dataset.sample_weights)
        info = 'Sample weights: {} / {} images found in {} (others = 1), mean {:.3f}, min {:.3f}'.format(
            n_found, len(train_dataset), args.sample_weights, w.mean().item(), w.min().item())
        print(info)
        with open(log_txt_path, 'a') as f:
            f.write(info + '\n')
    else:
        train_dataset = datasets.ImageFolder(train_data_file_path, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0 if args.cache_clip else args.workers,
                                               pin_memory=True, 
                                               drop_last=True)

    # resume projection head, optimizer, scheduler, curve and RNG states from a previous run
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        model.module.projection_head.load_state_dict(state['projection_head'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        n = min(start_epoch, args.epochs)
        recorder.epoch_losses[:n] = state['epoch_losses'][:n]
        recorder.epoch_accuracy[:n] = state['epoch_accuracy'][:n]
        recorder.current_epoch = n
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        torch.cuda.set_rng_state_all(state['cuda_state'])
        print('Resumed from {} (finished epoch {}), continuing at epoch {}'.format(args.resume, state['epoch'], start_epoch))
        with open(log_txt_path, 'a') as f:
            f.write('Resumed from ' + args.resume + ' at epoch ' + str(start_epoch) + '\n')

    for epoch in range(start_epoch, args.epochs):

        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')      
            
        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)
        scheduler.step()

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc)
        recorder.plot_curve(log_curve_path)
        print('The train accuracy: {:.3f}'.format(train_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(train_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')

        # save a checkpoint after every epoch so progress survives an interrupted run
        epoch_checkpoint_name = checkpoint_path + '-model.pth'
        torch.save(model.module.projection_head.state_dict(), epoch_checkpoint_name)
        torch.save({'epoch': epoch,
                    'projection_head': model.module.projection_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch_losses': recorder.epoch_losses,
                    'epoch_accuracy': recorder.epoch_accuracy,
                    'random_state': random.getstate(),
                    'numpy_state': np.random.get_state(),
                    'torch_state': torch.get_rng_state(),
                    'cuda_state': torch.cuda.get_rng_state_all()}, resume_path)
        if args.extra_checkpoint_dir:
            os.makedirs(args.extra_checkpoint_dir, exist_ok=True)
            shutil.copy2(epoch_checkpoint_name, args.extra_checkpoint_dir)
            shutil.copy2(resume_path, args.extra_checkpoint_dir)
            shutil.copy2(log_txt_path, args.extra_checkpoint_dir)
        if epoch + 1 in args.save_epochs:
            kept_name = checkpoint_path + '-e' + str(epoch + 1) + '-model.pth'
            shutil.copy2(epoch_checkpoint_name, kept_name)
            if args.extra_checkpoint_dir:
                shutil.copy2(kept_name, args.extra_checkpoint_dir)

    #  save model and conduct zero-shot prediction
    checkpoint_name = checkpoint_path + '-model.pth'
    torch.save(model.module.projection_head.state_dict(), checkpoint_name)
    if args.extra_checkpoint_dir:
        os.makedirs(args.extra_checkpoint_dir, exist_ok=True)
        shutil.copy2(checkpoint_name, args.extra_checkpoint_dir)
        shutil.copy2(log_txt_path, args.extra_checkpoint_dir)


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()

    for i, (inputs, sample_w) in enumerate(train_loader):

        # compute output
        if args.cache_clip:
            clip_features, target_features = inputs
            n = clip_features.size(0)
            logit_scale, image_features, text_features = model(clip_features.cuda(), target_features.cuda())
        else:
            images = inputs.cuda()
            n = images.size(0)
            logit_scale, image_features, text_features = model(image=images)
        target = torch.arange(n).cuda()
        
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        if args.sample_weights:
            # weighted InfoNCE: sample i weights both its image->text and text->image terms
            sample_w = sample_w.float().cuda()
            loss_vision = (F.cross_entropy(logits_per_image, target, reduction='none') * sample_w).sum() / sample_w.sum()
            loss_text = (F.cross_entropy(logits_per_text, target, reduction='none') * sample_w).sum() / sample_w.sum()
        else:
            loss_vision = criterion(logits_per_image, target)
            loss_text = criterion(logits_per_text, target)
        
        loss = 0.5 * loss_vision + 0.5 * loss_text
        
        # measure accuracy and record loss
        acc1, _ = accuracy(logits_per_image, target, topk=(1, 5))
        losses.update(loss.item(), n)
        top1.update(acc1[0], n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
            
    return top1.avg, losses.avg


class WeightedImageFolder(datasets.ImageFolder):
    """ImageFolder that returns (image, sample weight) instead of (image, class label)."""
    def __init__(self, root, transform, weights):
        super().__init__(root, transform)
        self.sample_weights = [float(weights.get(os.path.relpath(p, root), 1.0)) for p, _ in self.samples]

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, self.sample_weights[index]


class CachedPairs(torch.utils.data.Dataset):
    """(CLIP feature, BLIP-2 target) pairs cached on disk, matched by image path (EXP-003).
    Returns ((clip feature, target), sample weight); weights default to 1."""
    def __init__(self, clip_npz, target_dir, weights=None):
        import glob
        clip = np.load(clip_npz)
        chunks = sorted(glob.glob(os.path.join(target_dir, 'chunk_*.npz')))
        targets = {}
        for c in chunks:
            d = np.load(c)
            targets.update(zip(d['paths'], d['all']))
        keep = [i for i, p in enumerate(clip['paths']) if p in targets]
        self.n_skipped = len(clip['paths']) - len(keep)
        paths = [clip['paths'][i] for i in keep]
        self.clip_features = torch.from_numpy(clip['feats'][keep])
        self.targets = torch.from_numpy(np.stack([targets[p] for p in paths]))
        self.sample_weights = torch.tensor([float((weights or {}).get(p, 1.0)) for p in paths])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (self.clip_features[index], self.targets[index]), self.sample_weights[index]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
