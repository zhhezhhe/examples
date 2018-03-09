import argparse
import os
import shutil
import time
import io
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from image_load import ImageFolder as ImageFolder
from MyResnet import MyResnet as MyResnet
from PIL import Image


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--image_dir', default='', type=str, metavar='DIR',
                    help='path to dataset')

parser.add_argument('--result_json', default='', type=str,
                    help='path to output json')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-num_classes', '--num_classes', default=10, type=int, metavar='N',
                    help='num classes for finetune (default: 10)')

parser.add_argument('--epochs', default=64, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-in', '--inference', dest='inference', action='store_true',
                    help='inference model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')

best_prec1 = 0


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def forward(filename, model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img = default_loader(filename)
    Transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    img = Transforms(img)
    img = torch.unsqueeze(img, 0)
    model.eval()
    input_var = torch.autograd.Variable(img, volatile=True)
    output = model(input_var)
    _, pred = output.data.topk(1, 1, True, True)
    classid = torch.squeeze(pred)
    return classid.cpu().numpy()[0]



def main():
    global args, best_prec1
    args = parser.parse_args()
    num_classes = args.num_classes

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    original_model = models.__dict__[args.arch]()
    model = MyResnet(original_model, args.arch, num_classes)


    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    class_to_idx = {
        'football_field': 0,
        'beach': 1,
        'badminton_court': 2,
        'snow': 3,
        'ward': 4,
        'basketball_court': 5,
        'golf_course': 6,
        'ice_rink': 7,
        'pool': 8,
        'classroom': 9}
    id_to_class = {}
    for key in class_to_idx:
        id_to_class[class_to_idx[key]] = key

    result_dic = {}
    result_dic['id_to_class'] = id_to_class
    num = 0
    for root, dirs, files in os.walk(args.image_dir):
        for name in files:
            if name.endswith('.jpg') and name not in result_dic.keys():
                num += 1
                filename = os.path.join(root, name)
                classid = forward(filename, model)
                result_dic[name] = classid
                if num % 100 == 0:
                    print(num)
                    print(len(result_dic)-1)
                    print(name)
                    print("--"*30)

    with io.open(args.result_json, 'w', encoding='utf-8') as fd:
        fd.write(unicode(json.dumps(result_dic,
                                    ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))))

    print("process %d images"%(len(result_dic)-1))




if __name__ == '__main__':
    main()
