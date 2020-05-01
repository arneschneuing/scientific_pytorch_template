import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from src.core_components.metric_tracker import MetricTracker
from src.data_handlers.data_loader import InfiniteDataLoader


def build_dataloaders(cfg):
    """
    Build required dataloaders from config.
    :param cfg: config dict
    :return: dict containing PyTorch data loaders with key options: ["train",
    "val", "test"]
    """

    train_loader = InfiniteDataLoader(
        torchvision.datasets.MNIST('./tmp/files', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),
        batch_size=cfg['batch_size'], shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./tmp/files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),
        batch_size=cfg['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./tmp/files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),
        batch_size=cfg['batch_size'], shuffle=True)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


def build_model(cfg):
    """
    Build model from config dict.
    :param cfg: config dict
    :return: PyTorch nn.Module model
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return Net()


def build_criterion(cfg):
    """
    Build loss object from config dict.
    :param cfg: config dict
    :return: PyTorch nn.Module object
    """

    return nn.CrossEntropyLoss()


def build_lr_scheduler(cfg, optimizer):
    if cfg.get('LR_Scheduler') is None:
        return None
    else:
        step_size = cfg['step_size']
        gamma = cfg['gamma']
        return StepLR(optimizer, step_size=step_size, gamma=gamma,
                      last_epoch=-1)


def build_metric_trackers(cfg):
    return {'train': MetricTracker(), 'eval': MetricTracker()}


def build_optimizer(cfg, params):
    """
    Build optimizer from config dict.
    :param cfg: config dict
    :param params: network parameters to be optimized
    :return:
    """

    learning_rate = cfg.get('learning_rate', None)
    return optim.SGD(params, lr=learning_rate)


if __name__ == '__main__':

    # Create config
    cfg = {'batch_size': 64, 'Optimizer': {'learning_rate': 0.01}}

    # Build data loaders
    data_loaders = build_dataloaders(cfg=cfg)

    print(f'Dataset Size: {len(data_loaders["train"].dataset)}')

    # Build network
    model = build_model(cfg=cfg)

    # Build criterion
    criterion = build_criterion(cfg=cfg)

    # Build optimizer
    optimizer = build_optimizer(cfg=cfg, params=model.parameters())

    # Build metric tracker
    metric_tracker = MetricTracker()

    # Iterate over dataset
    for batch_id, (batch_data, batch_labels) in \
            enumerate(data_loaders['train']):

        optimizer.zero_grad()

        print(f'Batch ID: {batch_id + 1}')

        # print(f'Batch Input: {batch_data.size()}')
        # print(f'Batch Label: {batch_labels.size()}')

        # Perform forward pass
        batch_output = model(batch_data)

        # print(f'Batch Output: {batch_output.size()}')

        # Compute loss
        loss = criterion(batch_output, batch_labels)

        # print(f'Loss: {loss}')

        # Update metric tracker
        metric_tracker.update(batch_output, batch_labels, loss)

        print(f'Accuracy: {metric_tracker.get_metrics()}')

        # Perform backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

