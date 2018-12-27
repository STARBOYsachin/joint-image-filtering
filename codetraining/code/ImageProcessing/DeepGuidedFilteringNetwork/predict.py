import os
import argparse

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from skimage.io import imsave
from skimage.color import gray2rgb
from torch.autograd import Variable

from dataset import PreSuDataset
from module import DeepGuidedFilter, DeepGuidedFilterAdvanced

def tensor_to_img(tensor, transpose=False):
    im = np.asarray(np.clip(np.squeeze(tensor.numpy()) * 255, 0, 255), dtype=np.uint8)
    if transpose:
        im = im.transpose((1, 2, 0))

    return im

parser = argparse.ArgumentParser(description='Predict with Deep Guided Filtering Networks')
parser.add_argument('--img_path',    type=str,   required=True,                      help='IMG_PATH')
parser.add_argument('--type',        type=str,   required=True,                      help='TYPE')
parser.add_argument('--model_type',  type=str,   required=True,                      help='MODEL_TYPE')

parser.add_argument('--gpu',         type=int,   default=0,                          help='GPU')
parser.add_argument('--r',           type=int,   default=1,                          help='R')
parser.add_argument('--eps',         type=float, default=1e-8,                       help='EPS')
parser.add_argument('--low_size',    type=int,   default=64,                         help='LOW_SIZE')
args = parser.parse_args()

model2name = {
    'guided_filter': 'lr',
    'deep_guided_filter': 'hr',
    'deep_guided_filter_advanced': 'hr_ad'
}

# Test Images
img_list = []
if args.img_path is not None:
    img_list.append(args.img_path)
assert len(img_list) > 0

model_path = os.path.join('/data/models', args.type, '{}_net_latest.pth'.format(model2name[args.model_type]))

# Model
if args.model_type in ['guided_filter', 'deep_guided_filter']:
    model = DeepGuidedFilter(args.r, args.eps)
elif args.model_type == 'deep_guided_filter_advanced':
    model = DeepGuidedFilterAdvanced(args.r, args.eps)
else:
    print('Not a valid model!')
    exit(-1)

if args.model_type in ['deep_guided_filter', 'deep_guided_filter_advanced']:
    model.load_state_dict(torch.load(model_path))
elif args.model_type == 'guided_filter':
    model.init_lr(model_path)
else:
    print('Not a valid model!')
    exit(-1)

# data set
test_data = PreSuDataset(img_list, low_size=args.low_size)

# GPU
if args.gpu >= 0:
    with torch.cuda.device(args.gpu):
        model.cuda()

# test
i_bar = tqdm(total=len(test_data), desc='#Images')
for idx, imgs in enumerate(test_data):
    name = os.path.basename(test_data.get_path(idx))

    lr_x, hr_x = imgs[1].unsqueeze(0), imgs[0].unsqueeze(0)
    if args.gpu >= 0:
        with torch.cuda.device(args.gpu):
            lr_x = lr_x.cuda()
            hr_x = hr_x.cuda()
    imgs = model(Variable(lr_x), Variable(hr_x)).data.cpu()

    for img in imgs:
        img = tensor_to_img(img, transpose=True)
        if args.type == 'style_transfer':
            img = gray2rgb(img.mean(axis=2).astype(img.dtype))
        imsave('/results/output.jpg', img)
        
        input_img = np.asarray(Image.open(test_data.get_path(idx)).convert('RGB'))
        w, h, _ = input_img.shape
        if w > h:
            out = np.hstack([input_img, img])
        else:
            out = np.vstack([input_img, img])
        imsave('/results/in_out.jpg', out)

    i_bar.update()