import argparse
import urllib.request

from subprocess import call

parser = argparse.ArgumentParser(description='MAIN APP')
parser.add_argument('task',       type=str)
parser.add_argument('model_type', type=str)

parser.add_argument('image_type', type=str)
parser.add_argument('image_path', type=str)
parser.add_argument('image_url',  type=str)

parser.add_argument('gpu',        type=int)
parser.add_argument('r',          type=int)
parser.add_argument('eps',        type=float)

parser.add_argument('low_size',   type=int)

parser.add_argument('optional',   type=str)
args = parser.parse_args()

if args.image_type == 'URL':
    args.image_path = '/results/input.jpg'
    urllib.request.urlretrieve(args.image_url, args.image_path)
    
args_str = ' --gpu {}'.format(args.gpu)
if args.r > 0:
    args_str += ' --r {}'.format(args.r)
if args.eps >= 0:
    args_str += ' --eps {}'.format(args.eps)

else:
    command = ' '.join(['python -u /code/ImageProcessing/DeepGuidedFilteringNetwork/predict.py',
                        '--type', args.task,
                        '--model_type', args.model_type, 
                        '--img_path', args.image_path,
                        '--low_size', str(args.low_size), args_str])
                        
print(command)
call(command, shell=True)
