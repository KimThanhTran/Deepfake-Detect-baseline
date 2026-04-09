import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks import build_detector
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
                'ForenSynths': { 'dataroot'   : 'd:/WorkSpace/NPR-DeepfakeDetection/dataset/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : 'd:/WorkSpace/NPR-DeepfakeDetection/dataset/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

         'DiffusionForensics': { 'dataroot'   : 'd:/WorkSpace/NPR-DeepfakeDetection/dataset/DiffusionForensics/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

        'UniversalFakeDetect': { 'dataroot'   : 'd:/WorkSpace/NPR-DeepfakeDetection/dataset/UniversalFakeDetect/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

checkpoint = torch.load(opt.model_path, map_location='cpu')
model_type = checkpoint.get('model_type', opt.model_type) if isinstance(checkpoint, dict) else opt.model_type
model = build_detector(model_type=model_type, num_classes=1)
# Check if checkpoint has 'model' key (full checkpoint dict) or is direct state_dict
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# Handle DataParallel saved models (remove 'module.' prefix if present)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt, device)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
