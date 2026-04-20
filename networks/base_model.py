# from pix2pix
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.start_epoch = 0
        self.resume_state = {}
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device("cuda:0" if opt.gpu_ids else "cpu")

    def save_networks(self, epoch, extra_state=None):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        checkpoint = {
            'model': self.model.state_dict(),
            'total_steps': self.total_steps,
            'epoch': self.start_epoch,
            'model_type': getattr(self.opt, 'model_type', 'npr'),
        }
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer'] = self.optimizer.state_dict()
        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, save_path)
        print(f'Saving model {save_path}')

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        if hasattr(checkpoint, '_metadata'):
            del checkpoint._metadata

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_state = checkpoint['model']
            total_steps = checkpoint.get('total_steps', 0)
            resume_epoch = checkpoint.get('epoch', 0)
            optimizer_state = checkpoint.get('optimizer')
            self.resume_state = {
                k: v for k, v in checkpoint.items()
                if k not in {'model', 'optimizer'}
            }
        else:
            # Backward compatibility with older checkpoints that stored only model.state_dict().
            model_state = checkpoint
            total_steps = 0
            resume_epoch = 0
            optimizer_state = None
            self.resume_state = {}

        self.model.load_state_dict(model_state)
        self.total_steps = total_steps
        self.start_epoch = resume_epoch

        if self.isTrain and not self.opt.new_optim and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
