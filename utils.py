import random
import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from randaugment import rand_augment_transform, GaussianBlur_simclr

# Custom Dataset Loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, use_randaug=False):
        super(CustomDataset, self).__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_randaug = use_randaug
        if self.use_randaug:
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
            normalize = self.transform.transforms[-1] # get normalize layer
            self.aug1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                ], p=1.0),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
                transforms.ToTensor(),
                normalize
            ])
            self.aug2 = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
                ], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur_simclr([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        print("Loading dataset...")
        self.load_dataset()

    def load_dataset(self):
        self.data = ImageFolder(self.dataset_path)

    def do_transform(self, img):
        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                img = self.aug1(img)
            else:
                img = self.aug2(img)
        else:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.do_transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class BalancedSoftmax(_Loss):
    def __init__(self, samples_per_class):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(samples_per_class)

    def balanced_softmax_loss(self, labels, logits, sample_per_class, reduction="mean"):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        sample_per_class: A int tensor of size [no of classes].
        reduction: string. One of "none", "mean", "sum"
        Returns:
        loss: A float tensor. Balanced Softmax Loss.
        """
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
        return loss

    def forward(self, input, label, reduction='mean'):
        return self.balanced_softmax_loss(label, input, self.sample_per_class, reduction)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, rank):
        self.reset()
        self.rank = rank

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

    def all_reduce(self):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.rank)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def add_to_confusion_matrix(confusion_matrix, output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def get_per_class_results(confusion_matrix):
    per_class_info = []
    for i in range(confusion_matrix.shape[0]):
        per_class_info.append(round((confusion_matrix[i,i] / confusion_matrix[i,:].sum()).item()*100, 3))
    return per_class_info

def make_deterministic(random_seed=42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_ckpt(epoch, model, per_class_results, run_name):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'per_class_results': per_class_results
    }
    torch.save(states, os.path.join(f"{run_name}.pth"))

def load_ckpt(model, run_name):
    print(f"Loading checkpoint {run_name}")
    saved = torch.load(f"{run_name}.pth", map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    return model