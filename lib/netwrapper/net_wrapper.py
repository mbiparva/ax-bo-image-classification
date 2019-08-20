import os
from utils.config import cfg

import torch
import torch.nn as nn
# import torch.nn.init as nn_init
from utils.miscellaneous import StepLRestart
import torch.optim as optim
from models.alexnet import alexnet


# noinspection PyTypeChecker
class NetWrapper(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.core_net = None
        self.criterion, self.optimizer, self.scheduler = None, None, None

        self.create_load(device)

        self.setup_optimizer()

    def create_load(self, device):
        if cfg.PRETRAINED_MODE == 'Custom':
            self.create_net()
            self.load(cfg.PT_PATH)
        else:
            self.create_net()

        self.core_net = self.core_net.to(device)

    def create_net(self):
        if cfg.NET_ARCH == 'alexnet':
            self.core_net = alexnet()
        # elif cfg.NET_ARCH == 'vggnet':
        #     self.core_net = vgg16_bn()
        # elif cfg.NET_ARCH == 'resnet':
        #     self.core_net = resnet50()
        else:
            raise NotImplementedError

    def setup_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()

        print('creating optimizer with the following parameters',
              cfg.TRAIN.LR, cfg.TRAIN.WEIGHT_DECAY, cfg.TRAIN.MOMENTUM)

        self.optimizer = optim.SGD(params=self.parameters(),
                                   lr=cfg.TRAIN.LR,
                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                   momentum=cfg.TRAIN.MOMENTUM,
                                   nesterov=cfg.TRAIN.NESTEROV)

        if cfg.TRAIN.SCHEDULER_MODE:
            if cfg.TRAIN.SCHEDULER_TYPE == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'step_restart':
                self.scheduler = StepLRestart(self.optimizer, step_size=4, restart_size=8, gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'multi':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=cfg.TRAIN.SCHEDULER_MULTI_MILESTONE,
                                                                gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'lambda':
                def lr_lambda(e): return 1 if e < 5 else .5 if e < 10 else .1 if e < 15 else .01
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10,
                                                                      cooldown=0,
                                                                      verbose=True)
            else:
                raise NotImplementedError

    def schedule_step(self, metric=None):
        if cfg.TRAIN.SCHEDULER_MODE:
            if cfg.TRAIN.SCHEDULER_TYPE in ['step', 'step_restart', 'multi', 'lambda']:
                self.scheduler.step()
            if cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler.step(metric.meters['loss'].avg)

    def save(self, file_path, e):
        torch.save(self.state_dict(), os.path.join(file_path, '{:03d}.pth'.format(e)))

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def forward(self, x):
        return self.core_net(x)

    def loss_update(self, p, a, step=True):
        loss = self.criterion(p, a)

        if step:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
